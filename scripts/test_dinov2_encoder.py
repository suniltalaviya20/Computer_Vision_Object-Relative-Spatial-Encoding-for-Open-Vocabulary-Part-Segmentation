from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.dinov2_encoder import FrozenDINOv2Encoder


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = PascalPartQueryDataset(
        manifest_path="data/manifests/train.json",
        image_size=224,
    )

    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"].to(device)

    print("\nLoaded batch")
    print("-" * 40)
    print("Images:", tuple(images.shape))
    print("Image minimum:", float(images.min()))
    print("Image maximum:", float(images.max()))
    print("Part names:", list(batch["part_name"]))

    encoder = FrozenDINOv2Encoder(
        model_name="dinov2_vits14",
    ).to(device)

    encoder.eval()

    with torch.no_grad():
        features = encoder(images)

    print("\nDINOv2 model")
    print("-" * 40)
    print("Patch size:", encoder.patch_size)
    print("Output dimension:", encoder.output_dim)

    print("\nFeature tensors")
    print("-" * 40)
    print("Input images:", tuple(images.shape))
    print("DINO features:", tuple(features.shape))
    print("Feature minimum:", float(features.min()))
    print("Feature maximum:", float(features.max()))
    print("Feature mean:", float(features.mean()))
    print("Feature standard deviation:", float(features.std()))

    total_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )

    print("\nParameters")
    print("-" * 40)
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)

    if trainable_parameters != 0:
        raise RuntimeError(
            "DINOv2 should be completely frozen, but some "
            "parameters are trainable."
        )

    expected_height = (
        images.shape[-2] // encoder.patch_size
    )

    expected_width = (
        images.shape[-1] // encoder.patch_size
    )

    expected_shape = (
        images.shape[0],
        encoder.output_dim,
        expected_height,
        expected_width,
    )

    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Expected feature shape {expected_shape}, "
            f"received {tuple(features.shape)}."
        )

    # Average magnitude across channels to produce one spatial map.
    activation_map = (
        features[0]
        .abs()
        .mean(dim=0, keepdim=True)
        .unsqueeze(0)
    )

    # Upsample only for visualization.
    activation_map_large = F.interpolate(
        activation_map,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    image = (
        batch["image"][0]
        .permute(1, 2, 0)
        .numpy()
    )

    object_mask = (
        batch["object_mask"][0, 0]
        .numpy()
    )

    part_mask = (
        batch["part_mask"][0, 0]
        .numpy()
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "dinov2_test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Input image")

    axes[1].imshow(object_mask, cmap="gray")
    axes[1].set_title(
        f"Object: {batch['object_name'][0]}"
    )

    axes[2].imshow(part_mask, cmap="gray")
    axes[2].set_title(
        f"Part: {batch['part_name'][0]}"
    )

    activation_plot = axes[3].imshow(
        activation_map_large
        .detach()
        .cpu()
        .numpy(),
    )

    axes[3].set_title(
        "DINOv2 feature magnitude"
    )

    figure.colorbar(
        activation_plot,
        ax=axes[3],
        fraction=0.046,
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    output_path = (
        output_dir
        / "dinov2_features.png"
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("\nAll DINOv2 checks passed.")
    print("Saved visualization:", output_path)


if __name__ == "__main__":
    main()