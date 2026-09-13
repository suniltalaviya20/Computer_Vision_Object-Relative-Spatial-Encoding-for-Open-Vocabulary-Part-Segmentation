from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PascalPart116Dataset(Dataset):
    """Pure query-level Pascal-Part-116 dataset.

    This loader preserves the original image resolution.
    It does not resize, normalize, augment, or calculate
    geometric features.
    """

    def __init__(
        self,
        split: str = "train",
        processed_dir: str | Path = (
            "data/processed"
        ),
        transform: Callable[
            [dict[str, Any]],
            dict[str, Any],
        ] | None = None,
    ) -> None:
        processed_dir = Path(
            processed_dir
        )

        if not processed_dir.is_absolute():
            processed_dir = (
                PROJECT_ROOT
                / processed_dir
            )

        self.manifest_path = (
            processed_dir
            / f"{split}.json"
        )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}\n"
                "Run python scripts/"
                "prepare_dataset.py first."
            )

        records = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            records,
            list,
        ):
            raise ValueError(
                "Manifest must contain "
                "a JSON list"
            )

        if not records:
            raise ValueError(
                f"Manifest is empty: "
                f"{self.manifest_path}"
            )

        self.split = split
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _resolve_path(
        path_value: str,
    ) -> Path:
        path = Path(path_value)

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset file not "
                f"found: {path}"
            )

        return path

    @staticmethod
    def _load_label_map(
        path: Path,
    ) -> np.ndarray:
        with Image.open(path) as annotation:
            labels = np.asarray(
                annotation
            )

        if labels.ndim != 2:
            raise ValueError(
                f"Expected a 2-D label map "
                f"at {path}, got "
                f"{labels.shape}"
            )

        return labels.astype(
            np.int64,
            copy=False,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        record = self.records[index]

        image_path = self._resolve_path(
            record["image_path"]
        )

        object_annotation_path = (
            self._resolve_path(
                record[
                    "object_annotation_path"
                ]
            )
        )

        part_annotation_path = (
            self._resolve_path(
                record[
                    "part_annotation_path"
                ]
            )
        )

        with Image.open(
            image_path
        ) as image_file:
            image_array = np.asarray(
                image_file.convert("RGB")
            )

        object_labels = (
            self._load_label_map(
                object_annotation_path
            )
        )

        part_labels = (
            self._load_label_map(
                part_annotation_path
            )
        )

        image_shape = (
            image_array.shape[:2]
        )

        if (
            image_shape
            != object_labels.shape
            or object_labels.shape
            != part_labels.shape
        ):
            raise ValueError(
                f"Spatial dimensions differ "
                f"for {record['sample_id']}: "
                f"image={image_shape}, "
                f"objects="
                f"{object_labels.shape}, "
                f"parts="
                f"{part_labels.shape}"
            )

        object_id = int(
            record["object_id"]
        )

        part_id = int(
            record["part_id"]
        )

        object_mask_array = (
            object_labels
            == object_id
        )

        raw_part_mask_array = (
            part_labels
            == part_id
        )

        # Remove any part pixels that do not
        # belong to the expected parent object.
        part_mask_array = (
            raw_part_mask_array
            & object_mask_array
        )

        if not object_mask_array.any():
            raise RuntimeError(
                f"Empty object mask for "
                f"{record['sample_id']}"
            )

        if not part_mask_array.any():
            raise RuntimeError(
                f"Empty part mask for "
                f"{record['sample_id']}"
            )

        image_tensor = (
            torch.from_numpy(
                image_array
                .transpose(2, 0, 1)
                .copy()
            )
            .to(torch.uint8)
        )

        object_labels_tensor = (
            torch.from_numpy(
                object_labels.copy()
            )
            .long()
        )

        part_labels_tensor = (
            torch.from_numpy(
                part_labels.copy()
            )
            .long()
        )

        object_mask_tensor = (
            torch.from_numpy(
                object_mask_array.copy()
            )
            .bool()
        )

        raw_part_mask_tensor = (
            torch.from_numpy(
                raw_part_mask_array.copy()
            )
            .bool()
        )

        part_mask_tensor = (
            torch.from_numpy(
                part_mask_array.copy()
            )
            .bool()
        )

        sample = {
            "sample_id": (
                record["sample_id"]
            ),
            "image_id": (
                record["image_id"]
            ),
            "source_split": (
                record["source_split"]
            ),

            # Original dataset values.
            "image": image_tensor,
            "object_labels": (
                object_labels_tensor
            ),
            "part_labels": (
                part_labels_tensor
            ),

            # Query-specific masks.
            "object_mask": (
                object_mask_tensor
            ),
            "raw_part_mask": (
                raw_part_mask_tensor
            ),
            "part_mask": (
                part_mask_tensor
            ),

            # IDs and names.
            "object_id": object_id,
            "object_name": (
                record["object_name"]
            ),
            "part_id": part_id,
            "part_name": (
                record["part_name"]
            ),
            "full_part_name": (
                record["full_part_name"]
            ),
            "query": record["query"],
            "evaluation_split": (
                record[
                    "evaluation_split"
                ]
            ),

            # File information.
            "image_path": str(
                image_path
            ),
            "object_annotation_path": str(
                object_annotation_path
            ),
            "part_annotation_path": str(
                part_annotation_path
            ),
            "original_height": (
                image_array.shape[0]
            ),
            "original_width": (
                image_array.shape[1]
            ),
        }

        if self.transform is not None:
            sample = self.transform(
                sample
            )

        return sample