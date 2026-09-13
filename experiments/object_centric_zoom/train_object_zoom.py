from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import ObjectCentricDataset

from src.features.dino_features import (
    get_device,
    load_dino_model,
)

from src.features.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.part_query_alignment.alignment_model import (
    PartQueryAlignmentSegmenter,
)

from src.part_query_alignment.alignment_loss import (
    compute_total_alignment_loss,
)

from src.features.metrics import (
    dice_score,
    iou_score,
)


MODE = "alignment_query_gated_uvd"

SEED = 42

IMAGE_SIZE = 224

CROP_CONTEXT_RATIO = 0.15

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16

NUM_WORKERS = 4

EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

VISUAL_DIM = 128
TEXT_COMMON_DIM = 128
TEXT_DECODER_DIM = 32

TEMPERATURE = 0.10

MASK_THRESHOLD = 0.5


DEVICE = get_device()

USE_AMP = torch.cuda.is_available()


OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "object_zoom"
)


def seed_everything(
    seed=42,
):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def encode_queries(
    clip_model,
    tokenizer,
    queries,
    device,
):
    features = []

    for query in queries:
        feature = extract_clip_features(
            clip_model,
            tokenizer,
            query,
            device=device,
        )

        features.append(
            feature
        )

    return torch.cat(
        features,
        dim=0,
    )


def get_trainable_state(
    model,
):
    state = {
        "visual_projection":
            model.visual_projection.state_dict(),

        "text_projection":
            model.text_projection.state_dict(),

        "text_decoder_projection":
            model.text_decoder_projection.state_dict(),

        "decoder":
            model.decoder.state_dict(),
    }

    if model.geometry_gate is not None:
        state["geometry_gate"] = (
            model.geometry_gate.state_dict()
        )

    return state


def run_epoch(
    model,
    loader,
    clip_model,
    tokenizer,
    optimizer=None,
    scaler=None,
):
    training = (
        optimizer is not None
    )

    if training:
        model.train()

    else:
        model.eval()


    total_loss = 0.0

    total_seg_loss = 0.0

    total_align_loss = 0.0

    total_iou = 0.0

    total_dice = 0.0

    total_samples = 0


    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        images = batch[
            "crop_image"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        object_masks = batch[
            "crop_object_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        targets = batch[
            "crop_part_mask"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        relative_u = batch[
            "crop_relative_u"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        relative_v = batch[
            "crop_relative_v"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        boundary_d = batch[
            "crop_boundary_d"
        ].to(
            DEVICE,
            non_blocking=True,
        )


        text_embeddings = encode_queries(
            clip_model,
            tokenizer,
            batch["query"],
            DEVICE,
        )


        batch_size = (
            images.shape[0]
        )


        if training:
            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(
            training
        ):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                if MODE in {
                    "alignment_fixed_uvd",
                    "alignment_query_gated_uvd",
                }:
                    logits, aux = model(
                        images,
                        text_embeddings,
                        object_masks,
                        relative_u,
                        relative_v,
                        boundary_d,
                    )
                else:
                    logits, aux = model(
                        images,
                        text_embeddings,
                        object_masks,
                        relative_u,
                        relative_v,
                    )


                losses = (
                    compute_total_alignment_loss(
                        mode=MODE,
                        logits=logits,
                        aux=aux,
                        part_mask=targets,
                        object_mask=object_masks,
                    )
                )


                loss = losses[
                    "total"
                ]


            if training:
                if scaler is not None:
                    scaler.scale(
                        loss
                    ).backward()


                    scaler.unscale_(
                        optimizer
                    )


                    torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter
                            in model.parameters()
                            if parameter.requires_grad
                        ),
                        max_norm=1.0,
                    )


                    scaler.step(
                        optimizer
                    )


                    scaler.update()


                else:
                    loss.backward()


                    torch.nn.utils.clip_grad_norm_(
                        (
                            parameter
                            for parameter
                            in model.parameters()
                            if parameter.requires_grad
                        ),
                        max_norm=1.0,
                    )


                    optimizer.step()


        with torch.no_grad():
            batch_iou = iou_score(
                logits,
                targets,
                threshold=MASK_THRESHOLD,
            )


            batch_dice = dice_score(
                logits,
                targets,
                threshold=MASK_THRESHOLD,
            )


        total_loss += (
            losses["total"].item()
            * batch_size
        )


        total_seg_loss += (
            losses["seg_total"].item()
            * batch_size
        )


        total_align_loss += (
            losses["align_total"].item()
            * batch_size
        )


        total_iou += (
            batch_iou.item()
            * batch_size
        )


        total_dice += (
            batch_dice.item()
            * batch_size
        )


        total_samples += (
            batch_size
        )


        if (
            batch_index % 50 == 0
            or batch_index == len(loader)
        ):
            phase = (
                "Train"
                if training
                else "Validation"
            )


            print(
                f"{phase} "
                f"{batch_index}/"
                f"{len(loader)} "
                f"Loss: "
                f"{loss.item():.4f} "
                f"IoU: "
                f"{batch_iou.item():.4f} "
                f"Dice: "
                f"{batch_dice.item():.4f}"
            )


    return {
        "loss":
            total_loss
            / total_samples,

        "seg_loss":
            total_seg_loss
            / total_samples,

        "align_loss":
            total_align_loss
            / total_samples,

        "iou":
            total_iou
            / total_samples,

        "dice":
            total_dice
            / total_samples,
    }


def main():
    seed_everything(
        SEED
    )


    print(
        "Device:",
        DEVICE,
    )


    print(
        "AMP:",
        USE_AMP,
    )


    print(
        "Mode:",
        MODE,
    )


    print(
        "Crop context:",
        CROP_CONTEXT_RATIO,
    )


    experiment_dir = (
        OUTPUT_ROOT
        / MODE
    )


    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    train_dataset = ObjectCentricDataset(
        split="train_seen",
        image_size=IMAGE_SIZE,
        context_ratio=CROP_CONTEXT_RATIO,
    )


    validation_dataset = (
        ObjectCentricDataset(
            split="validation_seen",
            image_size=IMAGE_SIZE,
            context_ratio=CROP_CONTEXT_RATIO,
        )
    )


    print(
        "Training samples:",
        len(train_dataset),
    )


    print(
        "Validation samples:",
        len(validation_dataset),
    )


    pin_memory = (
        DEVICE.type == "cuda"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )


    print()
    print(
        "Loading DINOv2..."
    )


    dino_model = load_dino_model(
        device=DEVICE
    )


    print(
        "Loading CLIP..."
    )


    clip_model, tokenizer = (
        load_clip_model(
            device=DEVICE
        )
    )


    print(
        "Creating crop model..."
    )


    model = PartQueryAlignmentSegmenter(
        dino_encoder=dino_model,
        mode=MODE,
        visual_dim=VISUAL_DIM,
        text_common_dim=TEXT_COMMON_DIM,
        text_decoder_dim=TEXT_DECODER_DIM,
        temperature=TEMPERATURE,
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )


    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


    print(
        "Trainable parameters:",
        trainable_parameters,
    )


    config = {
        "mode":
            MODE,

        "seed":
            SEED,

        "image_size":
            IMAGE_SIZE,

        "crop_context_ratio":
            CROP_CONTEXT_RATIO,

        "epochs":
            EPOCHS,

        "train_batch_size":
            TRAIN_BATCH_SIZE,

        "val_batch_size":
            VAL_BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "visual_dim":
            VISUAL_DIM,

        "text_common_dim":
            TEXT_COMMON_DIM,

        "text_decoder_dim":
            TEXT_DECODER_DIM,

        "temperature":
            TEMPERATURE,

        "mask_threshold":
            MASK_THRESHOLD,

        "dino":
            "dinov2_vits14",

        "clip":
            "ViT-B-32/openai",
    }


    config_path = (
        experiment_dir
        / "config.json"
    )


    config_path.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )


    history = []

    best_validation_iou = -1.0


    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        start_time = time.time()


        print()
        print(
            f"Epoch "
            f"{epoch}/"
            f"{EPOCHS}"
        )


        train_metrics = run_epoch(
            model,
            train_loader,
            clip_model,
            tokenizer,
            optimizer=optimizer,
            scaler=scaler,
        )


        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                clip_model,
                tokenizer,
                optimizer=None,
                scaler=None,
            )


        scheduler.step(
            validation_metrics[
                "iou"
            ]
        )


        elapsed = (
            time.time()
            - start_time
        )


        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )


        history_row = {
            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_seg_loss":
                train_metrics[
                    "seg_loss"
                ],

            "train_align_loss":
                train_metrics[
                    "align_loss"
                ],

            "train_iou":
                train_metrics[
                    "iou"
                ],

            "train_dice":
                train_metrics[
                    "dice"
                ],

            "val_loss":
                validation_metrics[
                    "loss"
                ],

            "val_seg_loss":
                validation_metrics[
                    "seg_loss"
                ],

            "val_align_loss":
                validation_metrics[
                    "align_loss"
                ],

            "val_iou":
                validation_metrics[
                    "iou"
                ],

            "val_dice":
                validation_metrics[
                    "dice"
                ],

            "learning_rate":
                current_lr,

            "seconds":
                elapsed,
        }


        history.append(
            history_row
        )


        print()
        print(
            f"[{MODE}] "
            f"Epoch "
            f"{epoch:02d}/"
            f"{EPOCHS} | "
            f"train IoU "
            f"{train_metrics['iou']:.4f} | "
            f"val IoU "
            f"{validation_metrics['iou']:.4f} | "
            f"val Dice "
            f"{validation_metrics['dice']:.4f} | "
            f"{elapsed:.1f}s"
        )


        checkpoint = {
            "epoch":
                epoch,

            "mode":
                MODE,

            "model_state":
                get_trainable_state(
                    model
                ),

            "optimizer_state":
                optimizer.state_dict(),

            "scheduler_state":
                scheduler.state_dict(),

            "val_metrics":
                validation_metrics,

            "config":
                config,
        }


        torch.save(
            checkpoint,
            experiment_dir
            / "last.pt",
        )


        if (
            validation_metrics[
                "iou"
            ]
            > best_validation_iou
        ):
            best_validation_iou = (
                validation_metrics[
                    "iou"
                ]
            )


            torch.save(
                checkpoint,
                experiment_dir
                / "best.pt",
            )


            print(
                "Saved new best checkpoint."
            )


        history_path = (
            experiment_dir
            / "history.json"
        )


        history_path.write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding="utf-8",
        )


    print()
    print(
        "Object-centric training completed."
    )


    print(
        "Best validation IoU:",
        best_validation_iou,
    )


if __name__ == "__main__":
    main()
    