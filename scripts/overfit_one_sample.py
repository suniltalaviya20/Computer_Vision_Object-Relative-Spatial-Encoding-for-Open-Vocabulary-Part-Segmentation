from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.decoder import TinyPartDecoder
from utils.losses import combined_segmentation_loss


def calculate_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Calculate binary mask Intersection over Union."""

    predictions = (
        torch.sigmoid(logits) >= threshold
    )

    targets_binary = targets >= 0.5

    intersection = (
        predictions & targets_binary
    ).sum().float()

    union = (
        predictions | targets_binary
    ).sum().float()

    if union == 0:
        return 1.0

    return float(intersection / union)


def prepare_model_input(
    sample: dict,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate image and geometric channels."""

    image = sample["image"].unsqueeze(0).to(device)

    object_mask = (
        sample["object_mask"]
        .unsqueeze(0)
        .to(device)
    )

    u_map = (
        sample["u_map"]
        .unsqueeze(0)
        .to(device)
    )

    v_map = (
        sample["v_map"]
        .unsqueeze(0)
        .to(device)
    )

    return torch.cat(
        [
            image,
            object_mask,
            u_map,
            v_map,
        ],
        dim=1,
    )


def save_visualization(
    sample: dict,
    logits: torch.Tensor,
    output_path: Path,
) -> None:
    image = (
        sample["image"]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    object_mask = (
        sample["object_mask"][0]
        .cpu()
        .numpy()
    )

    target_mask = (
        sample["part_mask"][0]
        .cpu()
        .numpy()
    )

    probability = (
        torch.sigmoid(logits)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_mask = probability >= 0.5

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(19, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Input image")

    axes[1].imshow(
        object_mask,
        cmap="gray",
    )
    axes[1].set_title(
        f"Object: {sample['object_name']}"
    )

    axes[2].imshow(
        target_mask,
        cmap="gray",
    )
    axes[2].set_title(
        f"Target: {sample['part_name']}"
    )

    probability_plot = axes[3].imshow(
        probability,
        vmin=0,
        vmax=1,
    )

    axes[3].set_title(
        "Predicted probability"
    )

    figure.colorbar(
        probability_plot,
        ax=axes[3],
        fraction=0.046,
    )

    axes[4].imshow(
        predicted_mask,
        cmap="gray",
    )
    axes[4].set_title(
        "Thresholded prediction"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        sample["full_part_name"],
        fontsize=15,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--containment-weight",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    dataset = PascalPartQueryDataset(
        manifest_path="data/manifests/train.json",
        image_size=args.image_size,
    )

    sample = dataset[args.sample_index]

    print("Sample ID:", sample["sample_id"])
    print("Object:", sample["object_name"])
    print("Part:", sample["part_name"])
    print("Full class:", sample["full_part_name"])

    model_input = prepare_model_input(
        sample,
        device,
    )

    target_mask = (
        sample["part_mask"]
        .unsqueeze(0)
        .to(device)
    )

    object_mask = (
        sample["object_mask"]
        .unsqueeze(0)
        .to(device)
    )

    print("Model input shape:", tuple(model_input.shape))
    print("Target shape:", tuple(target_mask.shape))

    model = TinyPartDecoder(
        input_channels=model_input.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    losses: list[float] = []
    iou_values: list[float] = []

    model.train()

    for step in range(1, args.steps + 1):
        logits = model(model_input)

        loss, components = combined_segmentation_loss(
            logits=logits,
            targets=target_mask,
            object_mask=object_mask,
            containment_weight=args.containment_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_iou = calculate_iou(
            logits,
            target_mask,
        )

        losses.append(components["total"])
        iou_values.append(current_iou)

        if (
            step == 1
            or step % 25 == 0
            or step == args.steps
        ):
            print(
                f"Step {step:04d} | "
                f"Total {components['total']:.4f} | "
                f"BCE {components['bce']:.4f} | "
                f"Dice {components['dice']:.4f} | "
                f"Contain {components['containment']:.4f} | "
                f"IoU {current_iou:.4f}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(model_input)

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "overfit_one_sample"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        output_dir
        / "prediction.png"
    )

    save_visualization(
        sample=sample,
        logits=final_logits,
        output_path=prediction_path,
    )

    figure, axis = plt.subplots(
        figsize=(7, 4),
    )

    axis.plot(losses)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Total loss")
    axis.set_title("One-sample overfitting loss")
    axis.grid(True, alpha=0.3)

    figure.tight_layout()

    loss_path = output_dir / "loss_curve.png"

    figure.savefig(
        loss_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    final_iou = calculate_iou(
        final_logits,
        target_mask,
    )

    checkpoint_path = (
        output_dir
        / "tiny_decoder.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sample_id": sample["sample_id"],
            "image_size": args.image_size,
            "final_iou": final_iou,
        },
        checkpoint_path,
    )

    print()
    print("Final IoU:", f"{final_iou:.4f}")
    print("Prediction:", prediction_path)
    print("Loss curve:", loss_path)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()