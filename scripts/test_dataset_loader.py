from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset


def main() -> None:
    dataset = PascalPartQueryDataset(
        manifest_path="data/manifests/train.json",
        image_size=448,
    )

    print("Dataset records:", len(dataset))

    sample = dataset[0]

    print("\nFirst sample")
    print("-" * 40)

    print("Sample ID:", sample["sample_id"])
    print("Object:", sample["object_name"])
    print("Part query:", sample["part_name"])
    print("Full part name:", sample["full_part_name"])
    print("Evaluation split:", sample["evaluation_split"])

    print("\nTensor shapes")
    print("-" * 40)

    print("Image:", tuple(sample["image"].shape))
    print(
        "Object mask:",
        tuple(sample["object_mask"].shape),
    )
    print(
        "Part mask:",
        tuple(sample["part_mask"].shape),
    )
    print("U map:", tuple(sample["u_map"].shape))
    print("V map:", tuple(sample["v_map"].shape))

    print("\nTensor ranges")
    print("-" * 40)

    print(
        "Image:",
        float(sample["image"].min()),
        "to",
        float(sample["image"].max()),
    )

    print(
        "Object mask:",
        float(sample["object_mask"].min()),
        "to",
        float(sample["object_mask"].max()),
    )

    print(
        "Part mask:",
        float(sample["part_mask"].min()),
        "to",
        float(sample["part_mask"].max()),
    )

    print(
        "U map:",
        float(sample["u_map"].min()),
        "to",
        float(sample["u_map"].max()),
    )

    print(
        "V map:",
        float(sample["v_map"].min()),
        "to",
        float(sample["v_map"].max()),
    )

    outside_part_pixels = (
        sample["part_mask"]
        * (1.0 - sample["object_mask"])
    ).sum()

    print(
        "Part pixels outside object:",
        float(outside_part_pixels),
    )

    # Test batching.
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch shapes")
    print("-" * 40)

    print("Images:", tuple(batch["image"].shape))
    print(
        "Object masks:",
        tuple(batch["object_mask"].shape),
    )
    print(
        "Part masks:",
        tuple(batch["part_mask"].shape),
    )
    print("U maps:", tuple(batch["u_map"].shape))
    print("V maps:", tuple(batch["v_map"].shape))
    print("Part names:", batch["part_name"])

    # Visualize the first sample.
    image = sample["image"].permute(1, 2, 0).numpy()
    object_mask = sample["object_mask"][0].numpy()
    part_mask = sample["part_mask"][0].numpy()
    u_map = sample["u_map"][0].numpy()
    v_map = sample["v_map"][0].numpy()

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "dataset_loader_test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("RGB image")

    axes[1].imshow(object_mask, cmap="gray")
    axes[1].set_title(
        f"Object: {sample['object_name']}"
    )

    axes[2].imshow(part_mask, cmap="gray")
    axes[2].set_title(
        f"Part: {sample['part_name']}"
    )

    image_u = axes[3].imshow(
        u_map,
        vmin=0,
        vmax=1,
    )
    axes[3].set_title("Relative U")
    figure.colorbar(
        image_u,
        ax=axes[3],
        fraction=0.046,
    )

    image_v = axes[4].imshow(
        v_map,
        vmin=0,
        vmax=1,
    )
    axes[4].set_title("Relative V")
    figure.colorbar(
        image_v,
        ax=axes[4],
        fraction=0.046,
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        sample["full_part_name"],
        fontsize=15,
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / f"{sample['image_stem']}_loader_test.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nSaved visualization:", output_path)


if __name__ == "__main__":
    main()