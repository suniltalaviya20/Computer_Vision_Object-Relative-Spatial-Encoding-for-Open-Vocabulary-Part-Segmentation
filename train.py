from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent

from datasets.part_dataset import PascalPartQueryDataset
from models.dino_clip_part_model import (
    ABLATION_SETTINGS,
    DINOCLIPPartModel,
)
from utils.losses import combined_segmentation_loss
from utils.metrics import calculate_segmentation_metrics


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker a reproducible seed."""

    del worker_id

    worker_seed = (
        torch.initial_seed() % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def limited_loader(
    loader: DataLoader,
    max_batches: int,
) -> Iterator[dict[str, Any]]:
    """Iterate through a loader with an optional batch limit."""

    for batch_index, batch in enumerate(loader):
        if (
            max_batches > 0
            and batch_index >= max_batches
        ):
            break

        yield batch


def number_of_batches(
    loader: DataLoader,
    max_batches: int,
) -> int:
    if max_batches > 0:
        return min(
            len(loader),
            max_batches,
        )

    return len(loader)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Move tensor values to the selected device."""

    non_blocking = device.type == "cuda"

    return {
        "images": batch["image"].to(
            device,
            non_blocking=non_blocking,
        ),
        "object_masks": batch["object_mask"].to(
            device,
            non_blocking=non_blocking,
        ),
        "target_masks": batch["part_mask"].to(
            device,
            non_blocking=non_blocking,
        ),
        "u_maps": batch["u_map"].to(
            device,
            non_blocking=non_blocking,
        ),
        "v_maps": batch["v_map"].to(
            device,
            non_blocking=non_blocking,
        ),
        "part_names": list(batch["part_name"]),
    }


def create_data_loader(
    dataset: PascalPartQueryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Construct a training or validation DataLoader."""

    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        loader_arguments["persistent_workers"] = True
        loader_arguments["prefetch_factor"] = 2

    return DataLoader(
        **loader_arguments
    )


def train_one_epoch(
    model: DINOCLIPPartModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    containment_weight: float,
    threshold: float,
    gradient_clip: float,
    max_batches: int,
    epoch: int,
) -> dict[str, float]:
    """Train the model for one epoch."""

    model.train()

    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "containment": 0.0,
        "iou": 0.0,
        "dice": 0.0,
    }

    sample_count = 0

    progress = tqdm(
        limited_loader(
            loader,
            max_batches,
        ),
        total=number_of_batches(
            loader,
            max_batches,
        ),
        desc=f"Train {epoch:03d}",
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    for raw_batch in progress:
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        batch_size = batch[
            "images"
        ].shape[0]

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                image=batch["images"],
                object_mask=batch["object_masks"],
                u_map=batch["u_maps"],
                v_map=batch["v_maps"],
                part_names=batch["part_names"],
            )

            loss, components = (
                combined_segmentation_loss(
                    logits=logits,
                    targets=batch["target_masks"],
                    object_mask=batch[
                        "object_masks"
                    ],
                    containment_weight=(
                        containment_weight
                    ),
                )
            )

        scaler.scale(loss).backward()

        if gradient_clip > 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=gradient_clip,
            )

        scaler.step(optimizer)
        scaler.update()

        metrics = (
            calculate_segmentation_metrics(
                logits=logits,
                targets=batch["target_masks"],
                threshold=threshold,
            )
        )

        totals["loss"] += (
            components["total"] * batch_size
        )

        totals["bce"] += (
            components["bce"] * batch_size
        )

        totals["dice_loss"] += (
            components["dice"] * batch_size
        )

        totals["containment"] += (
            components["containment"]
            * batch_size
        )

        totals["iou"] += (
            metrics["iou"] * batch_size
        )

        totals["dice"] += (
            metrics["dice"] * batch_size
        )

        sample_count += batch_size

        progress.set_postfix(
            loss=f"{components['total']:.4f}",
            iou=f"{metrics['iou']:.3f}",
            dice=f"{metrics['dice']:.3f}",
        )

    if sample_count == 0:
        raise RuntimeError(
            "The training DataLoader produced no samples."
        )

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


@torch.no_grad()
def validate(
    model: DINOCLIPPartModel,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    containment_weight: float,
    threshold: float,
    max_batches: int,
    epoch: int,
) -> dict[str, float]:
    """Evaluate the model on the validation set."""

    model.eval()

    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "containment": 0.0,
        "iou": 0.0,
        "dice": 0.0,
    }

    sample_count = 0

    progress = tqdm(
        limited_loader(
            loader,
            max_batches,
        ),
        total=number_of_batches(
            loader,
            max_batches,
        ),
        desc=f"Valid {epoch:03d}",
    )

    for raw_batch in progress:
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        batch_size = batch[
            "images"
        ].shape[0]

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                image=batch["images"],
                object_mask=batch["object_masks"],
                u_map=batch["u_maps"],
                v_map=batch["v_maps"],
                part_names=batch["part_names"],
            )

            loss, components = (
                combined_segmentation_loss(
                    logits=logits,
                    targets=batch["target_masks"],
                    object_mask=batch[
                        "object_masks"
                    ],
                    containment_weight=(
                        containment_weight
                    ),
                )
            )

        del loss

        metrics = (
            calculate_segmentation_metrics(
                logits=logits,
                targets=batch["target_masks"],
                threshold=threshold,
            )
        )

        totals["loss"] += (
            components["total"] * batch_size
        )

        totals["bce"] += (
            components["bce"] * batch_size
        )

        totals["dice_loss"] += (
            components["dice"] * batch_size
        )

        totals["containment"] += (
            components["containment"]
            * batch_size
        )

        totals["iou"] += (
            metrics["iou"] * batch_size
        )

        totals["dice"] += (
            metrics["dice"] * batch_size
        )

        sample_count += batch_size

        progress.set_postfix(
            loss=f"{components['total']:.4f}",
            iou=f"{metrics['iou']:.3f}",
            dice=f"{metrics['dice']:.3f}",
        )

    if sample_count == 0:
        raise RuntimeError(
            "The validation DataLoader produced no samples."
        )

    return {
        name: value / sample_count
        for name, value in totals.items()
    }


def build_checkpoint(
    model: DINOCLIPPartModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_iou: float,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Create a checkpoint without duplicating frozen encoders."""

    return {
        "epoch": epoch,
        "best_validation_iou": (
            best_validation_iou
        ),
        "model_config": model.get_config(),
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
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "scaler_state_dict": scaler.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": (
            validation_metrics
        ),
        "arguments": vars(arguments),
    }


def save_json(
    data: Any,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_epoch_summary(
    epoch: int,
    learning_rate: float,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    print()
    print("=" * 72)
    print(
        f"Epoch {epoch:03d} | "
        f"Learning rate {learning_rate:.6g}"
    )

    print(
        "Train | "
        f"loss {train_metrics['loss']:.4f} | "
        f"IoU {train_metrics['iou']:.4f} | "
        f"Dice {train_metrics['dice']:.4f}"
    )

    print(
        "Valid | "
        f"loss {validation_metrics['loss']:.4f} | "
        f"IoU {validation_metrics['iou']:.4f} | "
        f"Dice {validation_metrics['dice']:.4f}"
    )

    print("=" * 72)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the object-relative open-vocabulary "
            "part-segmentation model."
        )
    )

    parser.add_argument(
        "--mode",
        choices=sorted(ABLATION_SETTINGS),
        default="full",
    )

    parser.add_argument(
        "--train-manifest",
        default="data/manifests/train.json",
    )

    parser.add_argument(
        "--validation-manifest",
        default=(
            "data/manifests/validation.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
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
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--visual-channels",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--text-channels",
        type=int,
        default=32,
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
        "--gradient-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help=(
            "Limit training batches for debugging. "
            "Use 0 for the complete loader."
        ),
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
        help=(
            "Limit validation batches for debugging. "
            "Use 0 for the complete loader."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.image_size % 14 != 0:
        raise ValueError(
            "--image-size must be divisible by 14 "
            "for DINOv2 ViT-S/14."
        )

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("Device:", device)
    print("AMP enabled:", amp_enabled)
    print("Ablation mode:", args.mode)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    containment_weight = (
        args.containment_weight
        if args.mode == "full"
        else 0.0
    )

    print(
        "Containment weight:",
        containment_weight,
    )

    train_dataset = PascalPartQueryDataset(
        manifest_path=args.train_manifest,
        image_size=args.image_size,
    )

    validation_dataset = (
        PascalPartQueryDataset(
            manifest_path=(
                args.validation_manifest
            ),
            image_size=args.image_size,
        )
    )

    print(
        "Training records:",
        len(train_dataset),
    )

    print(
        "Validation records:",
        len(validation_dataset),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
    )

    validation_loader = create_data_loader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = DINOCLIPPartModel(
        dino_model_name="dinov2_vits14",
        clip_model_name="ViT-B/32",
        visual_channels=args.visual_channels,
        text_channels=args.text_channels,
        ablation_mode=args.mode,
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
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )

    output_directory = resolve_project_path(
        args.output_dir
    ) / args.mode

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configuration = {
        **vars(args),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "effective_containment_weight": (
            containment_weight
        ),
        "trainable_parameters": (
            trainable_parameter_count
        ),
        "frozen_parameters": (
            frozen_parameter_count
        ),
        "model_config": model.get_config(),
    }

    save_json(
        configuration,
        output_directory / "config.json",
    )

    history: list[dict[str, Any]] = []
    best_validation_iou = -1.0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            containment_weight=(
                containment_weight
            ),
            threshold=args.threshold,
            gradient_clip=args.gradient_clip,
            max_batches=args.max_train_batches,
            epoch=epoch,
        )

        validation_metrics = validate(
            model=model,
            loader=validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            containment_weight=(
                containment_weight
            ),
            threshold=args.threshold,
            max_batches=(
                args.max_validation_batches
            ),
            epoch=epoch,
        )

        scheduler.step(
            validation_metrics["iou"]
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print_epoch_summary(
            epoch=epoch,
            learning_rate=current_learning_rate,
            train_metrics=train_metrics,
            validation_metrics=(
                validation_metrics
            ),
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": (
                current_learning_rate
            ),
            "train": train_metrics,
            "validation": validation_metrics,
        }

        history.append(epoch_record)

        save_json(
            history,
            output_directory
            / "history.json",
        )

        is_best = (
            validation_metrics["iou"]
            > best_validation_iou
        )

        if is_best:
            best_validation_iou = (
                validation_metrics["iou"]
            )

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_validation_iou=(
                best_validation_iou
            ),
            train_metrics=train_metrics,
            validation_metrics=(
                validation_metrics
            ),
            arguments=args,
        )

        torch.save(
            checkpoint,
            output_directory / "last.pt",
        )

        if is_best:
            torch.save(
                checkpoint,
                output_directory / "best.pt",
            )

            print(
                "Saved new best checkpoint "
                f"with validation IoU "
                f"{best_validation_iou:.4f}"
            )

    print()
    print("Training complete.")
    print(
        "Best validation IoU:",
        f"{best_validation_iou:.4f}",
    )
    print(
        "Outputs:",
        output_directory,
    )


if __name__ == "__main__":
    main()