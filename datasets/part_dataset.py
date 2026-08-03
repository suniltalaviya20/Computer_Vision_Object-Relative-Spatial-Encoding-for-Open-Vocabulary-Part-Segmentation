from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from utils.geometry import create_relative_coordinate_maps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PascalPartQueryDataset(Dataset):
    """Dataset for text-conditioned object-part segmentation.

    Each manifest record produces:

    - RGB image
    - binary parent-object mask
    - binary requested-part mask
    - object-relative horizontal map U
    - object-relative vertical map V
    - object and part names
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 448,
    ) -> None:
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.is_absolute():
            self.manifest_path = (
                PROJECT_ROOT / self.manifest_path
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        self.records: list[dict[str, Any]] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        if not self.records:
            raise RuntimeError(
                f"Manifest is empty: {self.manifest_path}"
            )

        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_label_map(path: Path) -> np.ndarray:
        """Load a semantic annotation as a 2-D integer array."""

        label_image = Image.open(path)
        label_array = np.asarray(label_image)

        if label_array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D label map at {path}, "
                f"received shape {label_array.shape}."
            )

        return label_array.astype(np.int64)

    @staticmethod
    def _resize_binary_mask(
        mask: np.ndarray,
        image_size: int,
        preserve_small_regions: bool = False,
    ) -> Tensor:
        """Resize a binary mask.

        Nearest-neighbour interpolation preserves exact class boundaries,
        but very small parts can disappear when downsampling.

        For part masks, BOX interpolation followed by a > 0 threshold
        preserves pixels whenever any source-mask area contributes to the
        resized pixel.
        """

        mask_image = Image.fromarray(
            mask.astype(np.uint8) * 255,
            mode="L",
        )

        original_height, original_width = mask.shape

        is_downsampling = (
            original_height > image_size
            or original_width > image_size
        )

        if preserve_small_regions and is_downsampling:
            resampling_method = Image.Resampling.BOX
        else:
            resampling_method = Image.Resampling.NEAREST

        resized_mask = mask_image.resize(
            (image_size, image_size),
            resample=resampling_method,
        )

        resized_array = np.asarray(
            resized_mask,
            dtype=np.uint8,
        )

        if preserve_small_regions and is_downsampling:
            # Any source-mask contribution keeps the destination pixel.
            binary_array = resized_array > 0
        else:
            binary_array = resized_array >= 128

        return torch.from_numpy(
            binary_array.copy()
        ).float()

    @staticmethod
    def _resolve_path(relative_path: str) -> Path:
        path = PROJECT_ROOT / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Referenced dataset file does not exist: {path}"
            )

        return path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        image_path = self._resolve_path(
            record["image_path"]
        )

        object_annotation_path = self._resolve_path(
            record["object_annotation_path"]
        )

        part_annotation_path = self._resolve_path(
            record["part_annotation_path"]
        )

        # Load RGB input.
        image = Image.open(image_path).convert("RGB")

        # Load full semantic label maps.
        object_labels = self._load_label_map(
            object_annotation_path
        )

        part_labels = self._load_label_map(
            part_annotation_path
        )

        if object_labels.shape != part_labels.shape:
            raise ValueError(
                f"Object and part annotation shapes differ for "
                f"{record['sample_id']}: "
                f"{object_labels.shape} vs {part_labels.shape}"
            )

        object_id = int(record["object_id"])
        part_id = int(record["part_id"])

        # Convert semantic maps into binary masks for this query.
        object_mask_array = object_labels == object_id
        part_mask_array = part_labels == part_id

        if not np.any(object_mask_array):
            raise RuntimeError(
                f"Parent-object mask is empty for "
                f"{record['sample_id']}."
            )

        if not np.any(part_mask_array):
            raise RuntimeError(
                f"Part mask is empty for {record['sample_id']}."
            )

        # A requested part should belong to the selected parent object.
        # This also removes small annotation inconsistencies.
        part_mask_array = (
            part_mask_array & object_mask_array
        )

        if not np.any(part_mask_array):
            raise RuntimeError(
                f"Part mask has no overlap with its parent object for "
                f"{record['sample_id']}."
            )

        # Resize the RGB image.
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

        # Convert image to float tensor [3, H, W] in [0, 1].
        image_tensor = TF.pil_to_tensor(image).float() / 255.0

        # Resize masks using nearest neighbour.
        object_mask = self._resize_binary_mask(
            mask=object_mask_array,
            image_size=self.image_size,
            preserve_small_regions=False,
        )

        part_mask = self._resize_binary_mask(
            mask=part_mask_array,
            image_size=self.image_size,
            preserve_small_regions=True,
        )

        # Ensure resized part pixels remain inside the resized object.
        # First check the independently resized masks.
        if object_mask.sum().item() == 0:
            raise RuntimeError(
                f"Object mask disappeared after resizing for "
                f"{record['sample_id']}."
            )

        if part_mask.sum().item() == 0:
            raise RuntimeError(
                f"Part mask disappeared during resizing for "
                f"{record['sample_id']}."
            )

        # The object and part masks use different resizing methods:
        # object -> nearest neighbour
        # part   -> coverage-preserving BOX interpolation
        #
        # Tiny boundary differences may therefore cause the resized part to
        # fall slightly outside the resized object mask.
        overlap_mask = part_mask * object_mask

        if overlap_mask.sum().item() > 0:
            # Keep only the region belonging to the resized parent object.
            part_mask = overlap_mask
        else:
            # The original masks overlapped before resizing, so a lack of overlap
            # here is a resizing artefact. Preserve the tiny part and minimally
            # expand the resized object mask to contain it.
            object_mask = torch.maximum(
                object_mask,
                part_mask,
            )
        # Generate coordinates after resizing, because they must match
        # the final training resolution.
        u_map, v_map = create_relative_coordinate_maps(
            object_mask
        )

        return {
            "image": image_tensor,
            "object_mask": object_mask.unsqueeze(0),
            "part_mask": part_mask.unsqueeze(0),
            "u_map": u_map.unsqueeze(0),
            "v_map": v_map.unsqueeze(0),
            "sample_id": record["sample_id"],
            "image_stem": record["image_stem"],
            "object_id": object_id,
            "object_name": record["object_name"],
            "part_id": part_id,
            "part_name": record["part_name"],
            "full_part_name": record["full_part_name"],
            "evaluation_split": record["evaluation_split"],
        }