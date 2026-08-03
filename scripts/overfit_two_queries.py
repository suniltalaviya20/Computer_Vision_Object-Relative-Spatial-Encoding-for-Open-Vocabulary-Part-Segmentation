from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.query_conditioned_model import QueryConditionedPartModel
from utils.losses import combined_segmentation_loss


VOCABULARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "part_vocabulary.json"
)


def calculate_iou_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> list[float]:
    predictions = (
        torch.sigmoid(logits) >= threshold
    )

    targets = targets >= 0.5

    values: list[float] = []

    for prediction, target in zip(
        predictions,
        targets,
    ):
        intersection = (
            prediction & target
        ).sum().float()

        union = (
            prediction | target
        ).sum().float()

        if union == 0:
            values.append(1.0)
        else:
            values.append(
                float(intersection / union)
            )

    return values


def find_two_queries_for_same_object(
    dataset: PascalPartQueryDataset,
) -> tuple[int, int]:
    """Find one object having two different annotated parts."""

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

    for indices in grouped_indices.values():
        part_names = {
            dataset.records[index]["part_name"]
            for index in indices
        }

        if len(part_names) < 2:
            continue

        first_index = indices[0]
        first_part = dataset.records[
            first_index
        ]["part_name"]

        for second_index in indices[1:]:
            second_part = dataset.records[
                second_index
            ]["part_name"]

            if second_part != first_part:
                return first_index, second_index

    raise RuntimeError(
        "Could not find an object with two different parts."
    )


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
        4,
        figsize=(14, 4 * number_of_samples),
    )

    if number_of_samples == 1:
        axes = axes[None, :]

    for row, sample in enumerate(samples):
        image = (
            sample["image"]
            .permute(1, 2, 0)
            .numpy()
        )

        target = (
            sample["part_mask"][0]
            .numpy()
        )

        probability = (
            probabilities[row, 0]
            .numpy()
        )

        prediction = probability >= 0.5

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Image\nObject: {sample['object_name']}"
        )

        axes[row, 1].imshow(
            target,
            cmap="gray",
        )

        axes[row, 1].set_title(
            f"Target\nQuery: {sample['part_name']}"
        )

        probability_plot = axes[row, 2].imshow(
            probability,
            vmin=0,
            vmax=1,
        )

        axes[row, 2].set_title(
            "Predicted probability"
        )

        figure.colorbar(
            probability_plot,
            ax=axes[row, 2],
            fraction=0.046,
        )

        axes[row, 3].imshow(
            prediction,
            cmap="gray",
        )

        axes[row, 3].set_title(
            "Thresholded prediction"
        )

        for column in range(4):
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
        default=800,
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

    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    vocabulary = json.loads(
        VOCABULARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    part_to_index = vocabulary[
        "part_to_index"
    ]

    dataset = PascalPartQueryDataset(
        manifest_path="data/manifests/train.json",
        image_size=args.image_size,
    )

    first_index, second_index = (
        find_two_queries_for_same_object(
            dataset
        )
    )

    samples = [
        dataset[first_index],
        dataset[second_index],
    ]

    print("\nSelected samples:")

    for sample in samples:
        print(
            sample["sample_id"],
            "|",
            sample["full_part_name"],
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

    part_masks = stack_samples(
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

    part_indices = torch.tensor(
        [
            part_to_index[
                sample["part_name"]
            ]
            for sample in samples
        ],
        dtype=torch.long,
        device=device,
    )

    print("\nTensor shapes:")
    print("Images:", tuple(images.shape))
    print(
        "Object masks:",
        tuple(object_masks.shape),
    )
    print(
        "Target masks:",
        tuple(part_masks.shape),
    )
    print(
        "Part indices:",
        tuple(part_indices.shape),
    )

    model = QueryConditionedPartModel(
        number_of_parts=vocabulary[
            "number_of_parts"
        ],
        text_channels=16,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    model.train()

    loss_history: list[float] = []

    for step in range(1, args.steps + 1):
        logits = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_indices=part_indices,
        )

        loss, components = (
            combined_segmentation_loss(
                logits=logits,
                targets=part_masks,
                object_mask=object_masks,
                containment_weight=(
                    args.containment_weight
                ),
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(
            components["total"]
        )

        iou_values = calculate_iou_per_sample(
            logits,
            part_masks,
        )

        if (
            step == 1
            or step % 50 == 0
            or step == args.steps
        ):
            iou_text = ", ".join(
                f"{samples[index]['part_name']}: "
                f"{iou:.3f}"
                for index, iou in enumerate(
                    iou_values
                )
            )

            print(
                f"Step {step:04d} | "
                f"Loss {components['total']:.4f} | "
                f"{iou_text}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_indices=part_indices,
        )

    final_ious = calculate_iou_per_sample(
        final_logits,
        part_masks,
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "overfit_two_queries"
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

    figure, axis = plt.subplots(
        figsize=(7, 4),
    )

    axis.plot(loss_history)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Total loss")
    axis.set_title(
        "Two-query overfitting loss"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    loss_path = (
        output_dir
        / "loss_curve.png"
    )

    figure.savefig(
        loss_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    checkpoint_path = (
        output_dir
        / "query_conditioned_model.pt"
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "part_to_index": part_to_index,
            "sample_ids": [
                sample["sample_id"]
                for sample in samples
            ],
            "final_ious": final_ious,
            "image_size": args.image_size,
        },
        checkpoint_path,
    )

    print("\nFinal results:")

    for sample, iou in zip(
        samples,
        final_ious,
    ):
        print(
            f"{sample['part_name']}: "
            f"IoU {iou:.4f}"
        )

    print("Predictions:", prediction_path)
    print("Loss curve:", loss_path)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()