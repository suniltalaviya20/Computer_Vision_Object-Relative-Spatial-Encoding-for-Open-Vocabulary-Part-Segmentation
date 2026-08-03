from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.part_dataset import PascalPartQueryDataset
from models.dino_clip_part_model import DINOCLIPPartModel
from utils.losses import combined_segmentation_loss
from utils.metrics import calculate_segmentation_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def create_loader(
    manifest_path: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PascalPartQueryDataset(
        manifest_path=manifest_path,
        image_size=image_size,
    )

    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }

    if num_workers > 0:
        loader_arguments["persistent_workers"] = True
        loader_arguments["prefetch_factor"] = 2

    return DataLoader(**loader_arguments)


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    DINOCLIPPartModel,
    dict[str, Any],
]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # The checkpoint was generated locally by this project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_config = checkpoint["model_config"]

    model = DINOCLIPPartModel(
        dino_model_name=model_config[
            "dino_model_name"
        ],
        clip_model_name=model_config[
            "clip_model_name"
        ],
        visual_channels=int(
            model_config["visual_channels"]
        ),
        text_channels=int(
            model_config["text_channels"]
        ),
        ablation_mode=model_config[
            "ablation_mode"
        ],
    )

    model.visual_projection.load_state_dict(
        checkpoint[
            "visual_projection_state_dict"
        ]
    )

    model.text_projection.load_state_dict(
        checkpoint[
            "text_projection_state_dict"
        ]
    )

    model.decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_split(
    model: DINOCLIPPartModel,
    manifest_path: str | Path,
    split_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    containment_weight: float,
    amp_enabled: bool,
) -> dict[str, Any]:
    loader = create_loader(
        manifest_path=manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "containment_loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "outside_probability_ratio": 0.0,
    }

    sample_count = 0

    progress = tqdm(
        loader,
        desc=f"Evaluate {split_name}",
    )

    for batch in progress:
        non_blocking = device.type == "cuda"

        images = batch["image"].to(
            device,
            non_blocking=non_blocking,
        )

        object_masks = batch["object_mask"].to(
            device,
            non_blocking=non_blocking,
        )

        target_masks = batch["part_mask"].to(
            device,
            non_blocking=non_blocking,
        )

        u_maps = batch["u_map"].to(
            device,
            non_blocking=non_blocking,
        )

        v_maps = batch["v_map"].to(
            device,
            non_blocking=non_blocking,
        )

        part_names = list(batch["part_name"])
        current_batch_size = images.shape[0]

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                image=images,
                object_mask=object_masks,
                u_map=u_maps,
                v_map=v_maps,
                part_names=part_names,
            )

            _, loss_components = (
                combined_segmentation_loss(
                    logits=logits,
                    targets=target_masks,
                    object_mask=object_masks,
                    containment_weight=(
                        containment_weight
                    ),
                )
            )

        metrics = calculate_segmentation_metrics(
            logits=logits,
            targets=target_masks,
            threshold=threshold,
        )

        probabilities = torch.sigmoid(
            logits.float()
        )

        outside_probability = (
            probabilities
            * (1.0 - object_masks)
        ).sum(dim=(1, 2, 3))

        total_probability = probabilities.sum(
            dim=(1, 2, 3)
        )

        outside_ratio = (
            outside_probability
            / (total_probability + 1e-6)
        ).mean()

        totals["loss"] += (
            loss_components["total"]
            * current_batch_size
        )

        totals["bce"] += (
            loss_components["bce"]
            * current_batch_size
        )

        totals["dice_loss"] += (
            loss_components["dice"]
            * current_batch_size
        )

        totals["containment_loss"] += (
            loss_components["containment"]
            * current_batch_size
        )

        totals["iou"] += (
            metrics["iou"]
            * current_batch_size
        )

        totals["dice"] += (
            metrics["dice"]
            * current_batch_size
        )

        totals[
            "outside_probability_ratio"
        ] += (
            float(outside_ratio)
            * current_batch_size
        )

        sample_count += current_batch_size

        progress.set_postfix(
            iou=f"{metrics['iou']:.3f}",
            dice=f"{metrics['dice']:.3f}",
        )

    if sample_count == 0:
        raise RuntimeError(
            f"The {split_name} dataset is empty."
        )

    averaged_results = {
        name: value / sample_count
        for name, value in totals.items()
    }

    return {
        "split": split_name,
        "manifest": str(manifest_path),
        "samples": sample_count,
        **averaged_results,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained part-segmentation model "
            "on seen and unseen test sets."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/training/full/best.pt"
        ),
    )

    parser.add_argument(
        "--seen-manifest",
        default=(
            "data/manifests/test_seen.json"
        ),
    )

    parser.add_argument(
        "--unseen-manifest",
        default=(
            "data/manifests/test_unseen.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/training/full/"
            "test_results.json"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    output_path = resolve_path(
        args.output
    )

    print("Device:", device)
    print("AMP enabled:", amp_enabled)
    print("Checkpoint:", checkpoint_path)

    model, checkpoint = load_trained_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    arguments = checkpoint.get(
        "arguments",
        {},
    )

    image_size = int(
        arguments.get("image_size", 224)
    )

    mode = checkpoint[
        "model_config"
    ]["ablation_mode"]

    containment_weight = (
        float(
            arguments.get(
                "containment_weight",
                0.1,
            )
        )
        if mode == "full"
        else 0.0
    )

    print("Training mode:", mode)
    print("Image size:", image_size)
    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch"),
    )
    print(
        "Best validation IoU:",
        checkpoint.get(
            "best_validation_iou"
        ),
    )

    seen_results = evaluate_split(
        model=model,
        manifest_path=args.seen_manifest,
        split_name="seen",
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        threshold=args.threshold,
        containment_weight=(
            containment_weight
        ),
        amp_enabled=amp_enabled,
    )

    unseen_results = evaluate_split(
        model=model,
        manifest_path=args.unseen_manifest,
        split_name="unseen",
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        threshold=args.threshold,
        containment_weight=(
            containment_weight
        ),
        amp_enabled=amp_enabled,
    )

    results = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "training_mode": mode,
        "best_validation_iou": (
            checkpoint.get(
                "best_validation_iou"
            )
        ),
        "threshold": args.threshold,
        "seen": seen_results,
        "unseen": unseen_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nResults")
    print("=" * 68)

    for split_results in [
        seen_results,
        unseen_results,
    ]:
        print(
            f"{split_results['split'].upper():8s} | "
            f"samples {split_results['samples']:5d} | "
            f"IoU {split_results['iou']:.4f} | "
            f"Dice {split_results['dice']:.4f} | "
            f"outside ratio "
            f"{split_results['outside_probability_ratio']:.4f}"
        )

    print("=" * 68)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()  