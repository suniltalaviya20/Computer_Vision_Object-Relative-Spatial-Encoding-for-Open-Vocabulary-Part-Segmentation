from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.geometry import create_relative_coordinate_maps


DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "Pascal-Part-116"

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pascal_part_116_categories.json"
)

IMAGE_DIR = DATASET_ROOT / "images" / "train"
OBJECT_DIR = DATASET_ROOT / "annotations_detectron2_obj" / "train"
PART_DIR = DATASET_ROOT / "annotations_detectron2_part" / "train"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def valid_files(directory: Path) -> list[Path]:
    """Return non-hidden image files from a directory."""

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            and not path.name.startswith(".")
            and not path.name.startswith("._")
        )
    )


def build_stem_map(directory: Path) -> dict[str, Path]:
    """Map file stem to file path."""

    return {
        path.stem: path
        for path in valid_files(directory)
    }


def find_matching_sample(
    requested_stem: str | None = None,
) -> tuple[str, Path, Path, Path]:
    """Find an image with matching object and part annotations."""

    image_map = build_stem_map(IMAGE_DIR)
    object_map = build_stem_map(OBJECT_DIR)
    part_map = build_stem_map(PART_DIR)

    common_stems = sorted(
        set(image_map)
        & set(object_map)
        & set(part_map)
    )

    if not common_stems:
        raise RuntimeError(
            "No matching image/object/part files were found."
        )

    if requested_stem is not None:
        if requested_stem not in common_stems:
            raise ValueError(
                f"No complete sample found for stem: {requested_stem}"
            )

        stem = requested_stem
    else:
        stem = common_stems[0]

    return (
        stem,
        image_map[stem],
        object_map[stem],
        part_map[stem],
    )


def load_label_map(path: Path) -> np.ndarray:
    """Load a semantic annotation as a 2-D integer label map."""

    image = Image.open(path)
    array = np.asarray(image)

    if array.ndim == 2:
        return array.astype(np.int64)

    if array.ndim == 3 and array.shape[2] >= 3:
        # Convert each RGB colour into one integer label.
        rgb = array[..., :3].astype(np.int64)

        packed = (
            (rgb[..., 0] << 16)
            + (rgb[..., 1] << 8)
            + rgb[..., 2]
        )

        return packed

    raise ValueError(
        f"Unsupported annotation shape {array.shape} for {path}"
    )


def resize_label_map(
    label_map: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Resize labels using nearest-neighbour interpolation."""

    pil_image = Image.fromarray(label_map.astype(np.int32), mode="I")

    resized = pil_image.resize(
        (target_width, target_height),
        resample=Image.Resampling.NEAREST,
    )

    return np.asarray(resized).astype(np.int64)


def get_valid_labels(label_map: np.ndarray) -> list[int]:
    """Return semantic labels while excluding ignore/background pixels."""

    # Pascal-Part uses 255 as the ignore label.
    # If an annotation is stored as RGB, white becomes 16777215
    # after our RGB packing operation.
    ignored_values = {
        255,
        16777215,
    }

    return [
        int(value)
        for value in np.unique(label_map)
        if int(value) not in ignored_values
    ]


def select_largest_object(
    object_labels: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select the largest annotated parent-object region."""

    valid_labels = get_valid_labels(object_labels)

    if not valid_labels:
        raise RuntimeError(
            "The object annotation contains no usable object labels."
        )

    selected_label = max(
        valid_labels,
        key=lambda label: int(
            np.count_nonzero(object_labels == label)
        ),
    )

    object_mask = object_labels == selected_label

    return selected_label, object_mask


def select_part_inside_object(
    part_labels: np.ndarray,
    object_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select the part label with the largest overlap with the object."""

    valid_labels = get_valid_labels(part_labels)

    if not valid_labels:
        raise RuntimeError(
            "The part annotation contains no usable part labels."
        )

    overlap_counts = {
        label: int(
            np.count_nonzero(
                (part_labels == label) & object_mask
            )
        )
        for label in valid_labels
    }

    selected_label = max(
        overlap_counts,
        key=overlap_counts.get,
    )

    if overlap_counts[selected_label] == 0:
        raise RuntimeError(
            "No part annotation overlaps the selected object."
        )

    part_mask = (
        (part_labels == selected_label)
        & object_mask
    )

    return selected_label, part_mask


def overlay_mask(
    axis: plt.Axes,
    image: np.ndarray,
    mask: np.ndarray,
    title: str,
) -> None:
    """Show an RGB image with a binary mask overlay."""

    axis.imshow(image)

    masked_overlay = np.ma.masked_where(
        ~mask,
        mask.astype(float),
    )

    axis.imshow(
        masked_overlay,
        alpha=0.55,
        vmin=0,
        vmax=1,
    )

    axis.set_title(title)
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Optional image stem, for example 2008_000002",
    )

    args = parser.parse_args()

    stem, image_path, object_path, part_path = (
        find_matching_sample(args.stem)
    )

    rgb_image = np.asarray(
        Image.open(image_path).convert("RGB")
    )

    image_height, image_width = rgb_image.shape[:2]

    object_labels = load_label_map(object_path)
    part_labels = load_label_map(part_path)

    if object_labels.shape != (image_height, image_width):
        print(
            "Resizing object annotation:",
            object_labels.shape,
            "->",
            (image_height, image_width),
        )

        object_labels = resize_label_map(
            object_labels,
            image_width,
            image_height,
        )

    if part_labels.shape != (image_height, image_width):
        print(
            "Resizing part annotation:",
            part_labels.shape,
            "->",
            (image_height, image_width),
        )

        part_labels = resize_label_map(
            part_labels,
            image_width,
            image_height,
        )

    object_label, object_mask = select_largest_object(
        object_labels
    )

    part_label, part_mask = select_part_inside_object(
            part_labels,
            object_mask,
        )

    metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

    object_record = metadata["objects"][object_label]
    part_record = metadata["object_parts"][part_label]

    object_name = object_record["name"]
    part_name = part_record["part_name"]
    full_part_name = part_record["full_name"]

    if part_record["object_name"] != object_name:
        print(
            "Warning: selected object and part class do not match:",
            object_name,
            full_part_name,
        )

    object_mask_tensor = torch.from_numpy(
        object_mask.copy()
    ).float()

    u_map, v_map = create_relative_coordinate_maps(
        object_mask_tensor
    )

    u_array = u_map.cpu().numpy()
    v_array = v_map.cpu().numpy()

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "real_sample"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 9),
    )

    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    overlay_mask(
        axes[0, 1],
        rgb_image,
        object_mask,
        f"Parent object: {object_name} — ID {object_label}"
    )

    overlay_mask(
        axes[0, 2],
        rgb_image,
        part_mask,
        f"Selected part: {part_name} — ID {part_label}",
    )

    image_u = axes[1, 0].imshow(
        u_array,
        vmin=0,
        vmax=1,
    )

    axes[1, 0].set_title(
        "Object-relative horizontal map U"
    )
    axes[1, 0].axis("off")

    figure.colorbar(
        image_u,
        ax=axes[1, 0],
        fraction=0.046,
    )

    image_v = axes[1, 1].imshow(
        v_array,
        vmin=0,
        vmax=1,
    )

    axes[1, 1].set_title(
        "Object-relative vertical map V"
    )
    axes[1, 1].axis("off")

    figure.colorbar(
        image_v,
        ax=axes[1, 1],
        fraction=0.046,
    )

    axes[1, 2].imshow(rgb_image)

    axes[1, 2].imshow(
        np.ma.masked_where(
            ~object_mask,
            object_mask,
        ),
        alpha=0.25,
    )

    axes[1, 2].imshow(
        np.ma.masked_where(
            ~part_mask,
            part_mask,
        ),
        alpha=0.75,
    )

    axes[1, 2].set_title(
        "Object and part relationship"
    )
    axes[1, 2].axis("off")

    figure.suptitle(
        f"Pascal-Part sample: {stem}",
        fontsize=16,
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / f"{stem}_visualization.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print()
    print("Sample stem:", stem)
    print("Image:", image_path)
    print("Object annotation:", object_path)
    print("Part annotation:", part_path)
    print("Image shape:", rgb_image.shape)
    print("Selected object label:", object_label)
    print("Selected part label:", part_label)
    print(
        "Object pixels:",
        int(object_mask.sum()),
    )
    print(
        "Part pixels:",
        int(part_mask.sum()),
    )
    print(
        "Part contained in object:",
        bool(np.all(part_mask <= object_mask)),
    )
    print("Saved visualization:", output_path)

    print("Object name:", object_name)
    print("Generic part query:", part_name)
    print("Full object-part class:", full_part_name)
    print("Evaluation split:", object_record["split"])
    

if __name__ == "__main__":
    main()