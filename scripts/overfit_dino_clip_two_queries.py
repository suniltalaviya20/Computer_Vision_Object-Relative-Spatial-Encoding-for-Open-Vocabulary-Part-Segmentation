from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.dino_clip_part_model import DINOCLIPPartModel
from utils.losses import combined_segmentation_loss


def calculate_iou_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> list[float]:
    """Calculate binary IoU separately for every batch item."""

    predictions = torch.sigmoid(logits) >= threshold
    targets_binary = targets >= 0.5

    iou_values: list[float] = []

    for prediction, target in zip(
        predictions,
        targets_binary,
    ):
        intersection = (
            prediction & target
        ).sum().float()

        union = (
            prediction | target
        ).sum().float()

        if union == 0:
            iou_values.append(1.0)
        else:
            iou_values.append(
                float(intersection / union)
            )

    return iou_values


def find_good_query_pair(
    dataset: PascalPartQueryDataset,
) -> tuple[int, int]:
    """Select two large, distinct parts from the same object.

    Selecting reasonably large parts makes the first debugging
    experiment easier because DINOv2 initially operates on a
    relatively coarse patch grid.
    """

    grouped_indices: dict[
        tuple[str, int],
        list[int],
    ] = defaultdict(list)

    for index, record in enumerate(dataset.records):
        key = (
            record["image_stem"],
            int(record["object_id"]),
        )

        grouped_indices[key].append(index)

    best_pair: tuple[int, int] | None = None
    best_score = -1

    for indices in grouped_indices.values():
        sorted_indices = sorted(
            indices,
            key=lambda index: int(
                dataset.records[index].get(
                    "part_pixels",
                    0,
                )
            ),
            reverse=True,
        )

        for first_position, first_index in enumerate(
            sorted_indices
        ):
            first_record = dataset.records[first_index]

            for second_index in sorted_indices[
                first_position + 1:
            ]:
                second_record = dataset.records[
                    second_index
                ]

                if (
                    first_record["part_name"]
                    == second_record["part_name"]
                ):
                    continue

                first_pixels = int(
                    first_record.get(
                        "part_pixels",
                        0,
                    )
                )

                second_pixels = int(
                    second_record.get(
                        "part_pixels",
                        0,
                    )
                )

                # Prefer a pair where both parts are reasonably large.
                score = min(
                    first_pixels,
                    second_pixels,
                )

                if score > best_score:
                    best_score = score
                    best_pair = (
                        first_index,
                        second_index,
                    )

    if best_pair is None:
        raise RuntimeError(
            "Could not find two different parts belonging "
            "to the same object instance."
        )

    return best_pair


def stack_samples(
    samples: list[dict],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [
            sample[key]
            for sample in samples
        ]
    ).to(device)


def save_predictions(
    samples: list[dict],
    logits: torch.Tensor,
    output_path: Path,
) -> None:
    probabilities = (
        torch.sigmoid(logits)
        .detach()
        .cpu()
    )

    number_of_samples = len(samples)

    figure, axes = plt.subplots(
        number_of_samples,
        5,
        figsize=(18, 4 * number_of_samples),
    )

    if number_of_samples == 1:
        axes = axes[None, :]

    for row, sample in enumerate(samples):
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
            probabilities[row, 0]
            .numpy()
        )

        predicted_mask = probability >= 0.5

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Input image\n{sample['object_name']}"
        )

        axes[row, 1].imshow(
            object_mask,
            cmap="gray",
        )
        axes[row, 1].set_title(
            "Parent-object mask"
        )

        axes[row, 2].imshow(
            target_mask,
            cmap="gray",
        )
        axes[row, 2].set_title(
            f"Target\nQuery: {sample['part_name']}"
        )

        probability_plot = axes[row, 3].imshow(
            probability,
            vmin=0,
            vmax=1,
        )

        axes[row, 3].set_title(
            "Predicted probability"
        )

        figure.colorbar(
            probability_plot,
            ax=axes[row, 3],
            fraction=0.046,
        )

        axes[row, 4].imshow(
            predicted_mask,
            cmap="gray",
        )
        axes[row, 4].set_title(
            "Thresholded prediction"
        )

        for column in range(5):
            axes[row, column].axis("off")

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
        default=400,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--containment-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
        image_size=args.image_size,
    )

    first_index, second_index = find_good_query_pair(
        dataset
    )

    samples = [
        dataset[first_index],
        dataset[second_index],
    ]

    part_names = [
        sample["part_name"]
        for sample in samples
    ]

    print("\nSelected samples")
    print("-" * 60)

    for index, sample in zip(
        [first_index, second_index],
        samples,
    ):
        record = dataset.records[index]

        print(
            f"Index {index} | "
            f"{sample['full_part_name']} | "
            f"part pixels: {record.get('part_pixels', 'unknown')}"
        )

    images = stack_samples(
        samples,
        "image",
        device,
    )

    object_masks = stack_samples(
        samples,
        "object_mask",
        device,
    )

    target_masks = stack_samples(
        samples,
        "part_mask",
        device,
    )

    u_maps = stack_samples(
        samples,
        "u_map",
        device,
    )

    v_maps = stack_samples(
        samples,
        "v_map",
        device,
    )

    print("\nInput tensors")
    print("-" * 60)
    print("Images:", tuple(images.shape))
    print("Object masks:", tuple(object_masks.shape))
    print("Target masks:", tuple(target_masks.shape))
    print("U maps:", tuple(u_maps.shape))
    print("V maps:", tuple(v_maps.shape))
    print("Part names:", part_names)

    model = DINOCLIPPartModel(
        dino_model_name="dinov2_vits14",
        clip_model_name="ViT-B/32",
        visual_channels=128,
        text_channels=32,
    ).to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print("\nParameters")
    print("-" * 60)
    print(
        "Trainable parameters:",
        trainable_parameter_count,
    )
    print(
        "Frozen parameters:",
        frozen_parameter_count,
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    loss_history: list[float] = []
    iou_history: list[list[float]] = []

    model.train()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_names=part_names,
        )

        loss, components = combined_segmentation_loss(
            logits=logits,
            targets=target_masks,
            object_mask=object_masks,
            containment_weight=(
                args.containment_weight
            ),
        )

        loss.backward()
        optimizer.step()

        current_ious = calculate_iou_per_sample(
            logits,
            target_masks,
            threshold=args.threshold,
        )

        loss_history.append(
            components["total"]
        )

        iou_history.append(
            current_ious
        )

        if (
            step == 1
            or step % 25 == 0
            or step == args.steps
        ):
            iou_text = " | ".join(
                f"{part_name}: {iou:.3f}"
                for part_name, iou in zip(
                    part_names,
                    current_ious,
                )
            )

            print(
                f"Step {step:04d} | "
                f"Total {components['total']:.4f} | "
                f"BCE {components['bce']:.4f} | "
                f"Dice {components['dice']:.4f} | "
                f"Contain {components['containment']:.4f} | "
                f"{iou_text}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_names=part_names,
        )

    final_ious = calculate_iou_per_sample(
        final_logits,
        target_masks,
        threshold=args.threshold,
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "overfit_dino_clip_two_queries"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        output_dir
        / "predictions.png"
    )

    save_predictions(
        samples=samples,
        logits=final_logits,
        output_path=prediction_path,
    )

    loss_figure, loss_axis = plt.subplots(
        figsize=(7, 4),
    )

    loss_axis.plot(loss_history)
    loss_axis.set_xlabel("Training step")
    loss_axis.set_ylabel("Total loss")
    loss_axis.set_title(
        "DINOv2–CLIP two-query overfitting"
    )
    loss_axis.grid(
        True,
        alpha=0.3,
    )

    loss_figure.tight_layout()

    loss_path = (
        output_dir
        / "loss_curve.png"
    )

    loss_figure.savefig(
        loss_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(loss_figure)

    iou_figure, iou_axis = plt.subplots(
        figsize=(7, 4),
    )

    for query_index, part_name in enumerate(
        part_names
    ):
        values = [
            step_values[query_index]
            for step_values in iou_history
        ]

        iou_axis.plot(
            values,
            label=part_name,
        )

    iou_axis.set_xlabel("Training step")
    iou_axis.set_ylabel("IoU")
    iou_axis.set_ylim(0, 1)
    iou_axis.set_title(
        "IoU during two-query overfitting"
    )
    iou_axis.grid(
        True,
        alpha=0.3,
    )
    iou_axis.legend()

    iou_figure.tight_layout()

    iou_path = (
        output_dir
        / "iou_curve.png"
    )

    iou_figure.savefig(
        iou_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(iou_figure)

    # Save only the trainable components.
    # This avoids storing another copy of the large frozen models.
    checkpoint_path = (
        output_dir
        / "trainable_components.pt"
    )

    torch.save(
        {
            "visual_projection_state_dict": (
                model.visual_projection.state_dict()
            ),
            "text_projection_state_dict": (
                model.text_projection.state_dict()
            ),
            "decoder_state_dict": (
                model.decoder.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "dino_model_name": "dinov2_vits14",
            "clip_model_name": "ViT-B/32",
            "visual_channels": 128,
            "text_channels": 32,
            "image_size": args.image_size,
            "part_names": part_names,
            "sample_ids": [
                sample["sample_id"]
                for sample in samples
            ],
            "final_ious": final_ious,
        },
        checkpoint_path,
    )

    print("\nFinal results")
    print("-" * 60)

    for sample, iou in zip(
        samples,
        final_ious,
    ):
        print(
            f"{sample['full_part_name']}: "
            f"IoU {iou:.4f}"
        )

    print("\nSaved outputs")
    print("-" * 60)
    print("Predictions:", prediction_path)
    print("Loss curve:", loss_path)
    print("IoU curve:", iou_path)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()