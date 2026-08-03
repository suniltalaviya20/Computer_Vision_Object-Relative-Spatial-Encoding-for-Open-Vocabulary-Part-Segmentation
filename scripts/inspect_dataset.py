from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "Pascal-Part-116"


def find_first_file(directory: Path) -> Path:
    """Return the first regular file found recursively."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())

    if not files:
        raise FileNotFoundError(f"No files found inside {directory}")

    return files[0]


def inspect_image(path: Path, name: str) -> None:
    """Print basic information about an image or annotation."""
    image = Image.open(path)
    array = np.asarray(image)

    print(f"\n{name}")
    print("-" * len(name))
    print("Path:", path)
    print("PIL mode:", image.mode)
    print("Shape:", array.shape)
    print("Data type:", array.dtype)
    print("Minimum value:", array.min())
    print("Maximum value:", array.max())

    unique_values = np.unique(array)

    print("Number of unique values:", len(unique_values))
    print("First unique values:", unique_values[:30])


def main() -> None:
    image_path = find_first_file(DATASET_ROOT / "images" / "train")
    object_annotation_path = find_first_file(
        DATASET_ROOT / "annotations_detectron2_obj" / "train"
    )
    part_annotation_path = find_first_file(
        DATASET_ROOT / "annotations_detectron2_part" / "train"
    )

    inspect_image(image_path, "RGB image")
    inspect_image(object_annotation_path, "Object annotation")
    inspect_image(part_annotation_path, "Part annotation")


if __name__ == "__main__":
    main()