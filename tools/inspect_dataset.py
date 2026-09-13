from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from datasets import (  # noqa: E402
    PascalPart116Dataset,
)


VALID_SPLITS = (
    "train",
    "validation",
    "test",
    "train_seen",
    "validation_seen",
    "test_seen",
    "test_unseen",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an original "
            "Pascal-Part-116 sample."
        )
    )

    parser.add_argument(
        "--split",
        choices=VALID_SPLITS,
        default="train",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Specific dataset index. "
            "A random index is used "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed."
        ),
    )

    args = parser.parse_args()

    dataset = PascalPart116Dataset(
        split=args.split
    )

    if args.index is None:
        index = random.Random(
            args.seed
        ).randrange(
            len(dataset)
        )
    else:
        index = args.index

    if not (
        0
        <= index
        < len(dataset)
    ):
        raise IndexError(
            f"Index {index} is outside "
            f"dataset length "
            f"{len(dataset)}"
        )

    sample = dataset[index]

    print()
    print("Pascal-Part-116 sample")
    print("=" * 50)
    print(
        "Manifest split:   ",
        args.split,
    )
    print(
        "Dataset index:    ",
        index,
    )
    print(
        "Sample ID:        ",
        sample["sample_id"],
    )
    print(
        "Image ID:         ",
        sample["image_id"],
    )
    print(
        "Original size:    ",
        sample["original_width"],
        "x",
        sample["original_height"],
    )
    print(
        "Object:           ",
        sample["object_id"],
        sample["object_name"],
    )
    print(
        "Part:             ",
        sample["part_id"],
        sample["part_name"],
    )
    print(
        "Full part name:   ",
        sample["full_part_name"],
    )
    print(
        "Text query:       ",
        sample["query"],
    )
    print(
        "Evaluation split: ",
        sample["evaluation_split"],
    )
    print(
        "Object pixels:    ",
        int(
            sample[
                "object_mask"
            ].sum()
        ),
    )
    print(
        "Raw part pixels:  ",
        int(
            sample[
                "raw_part_mask"
            ].sum()
        ),
    )
    print(
        "Target pixels:    ",
        int(
            sample[
                "part_mask"
            ].sum()
        ),
    )
    print(
        "Image path:       ",
        sample["image_path"],
    )

    image = (
        sample["image"]
        .permute(1, 2, 0)
        .numpy()
    )

    object_labels = (
        sample["object_labels"]
        .numpy()
    )

    part_labels = (
        sample["part_labels"]
        .numpy()
    )

    object_mask = (
        sample["object_mask"]
        .numpy()
    )

    part_mask = (
        sample["part_mask"]
        .numpy()
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10),
    )

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(
        f"Original RGB image\n"
        f"{sample['image_id']}"
    )

    axes[0, 1].imshow(
        np.ma.masked_equal(
            object_labels,
            255,
        ),
        cmap="tab20",
    )
    axes[0, 1].set_title(
        "Complete object labels"
    )

    axes[0, 2].imshow(
        np.ma.masked_equal(
            part_labels,
            255,
        ),
        cmap="nipy_spectral",
    )
    axes[0, 2].set_title(
        "Complete part labels"
    )

    axes[1, 0].imshow(
        object_mask,
        cmap="Blues",
    )
    axes[1, 0].set_title(
        f"Object "
        f"{sample['object_id']}: "
        f"{sample['object_name']}"
    )

    axes[1, 1].imshow(
        part_mask,
        cmap="Reds",
    )
    axes[1, 1].set_title(
        f"Part "
        f"{sample['part_id']}: "
        f"{sample['part_name']}"
    )

    axes[1, 2].imshow(image)

    axes[1, 2].imshow(
        np.ma.masked_where(
            ~object_mask,
            object_mask,
        ),
        cmap="Blues",
        alpha=0.35,
    )

    axes[1, 2].imshow(
        np.ma.masked_where(
            ~part_mask,
            part_mask,
        ),
        cmap="Reds",
        alpha=0.75,
    )

    axes[1, 2].set_title(
        "Object (blue), part (red)"
    )

    for axis in axes.flat:
        axis.axis("off")

    figure.suptitle(
        f"Query: "
        f"{sample['full_part_name']}",
        fontsize=16,
    )

    figure.tight_layout()

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "dataset_inspection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"{args.split}_"
            f"{index}_"
            f"{sample['image_id']}_"
            f"part_"
            f"{sample['part_id']}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        "Visualization saved to:"
    )
    print(output_path)


if __name__ == "__main__":
    main()