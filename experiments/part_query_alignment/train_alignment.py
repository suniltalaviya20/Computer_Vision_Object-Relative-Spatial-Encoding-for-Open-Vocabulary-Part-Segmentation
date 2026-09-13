from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.alignment_dataset import AlignmentDataset
from src.features.dino_features import get_device, load_dino_model
from src.features.clip_features import load_clip_model
from src.part_query_alignment.alignment_model import PartQueryAlignmentSegmenter
from src.part_query_alignment.alignment_loss import compute_total_alignment_loss


MODE = "alignment_relative_uv"

SEED = 42
IMAGE_SIZE = 224
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VISUAL_DIM = 128
TEXT_COMMON_DIM = 128
TEXT_DECODER_DIM = 32
MASK_THRESHOLD = 0.5
ALIGNMENT_TEMPERATURE = 0.10

DEVICE = get_device()
USE_AMP = torch.cuda.is_available()

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "part_query_alignment"
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextEmbeddingCache:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}

    @torch.no_grad()
    def _encode_missing(self, queries):
        missing = sorted({q for q in queries if q not in self.cache})
        if not missing:
            return

        prompts = [f"a photo of the {q} of an object" for q in missing]
        tokens = self.tokenizer(prompts).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = F.normalize(embeddings.float(), dim=-1)

        for query, embedding in zip(missing, embeddings):
            self.cache[query] = embedding.detach().cpu()

    def __call__(self, queries):
        self._encode_missing(queries)
        batch = torch.stack([self.cache[q] for q in queries])
        return batch.to(self.device, non_blocking=True)


@torch.no_grad()
def batch_iou(logits, targets, threshold=0.5, eps=1e-6):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    predictions = predictions.flatten(1)
    targets = targets.flatten(1)
    intersection = (predictions & targets).sum(dim=1).float()
    union = (predictions | targets).sum(dim=1).float()
    return (intersection + eps) / (union + eps)


@torch.no_grad()
def batch_dice(logits, targets, threshold=0.5, eps=1e-6):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    predictions = predictions.flatten(1)
    targets = targets.flatten(1)
    intersection = (predictions & targets).sum(dim=1).float()
    denominator = (predictions.sum(dim=1) + targets.sum(dim=1)).float()
    return (2 * intersection + eps) / (denominator + eps)


@torch.no_grad()
def batch_alignment_iou(
    alignment_logits,
    part_mask,
    object_mask,
    threshold=0.5,
    eps=1e-6,
):
    output_size = alignment_logits.shape[-2:]
    target_low = F.adaptive_max_pool2d(part_mask, output_size=output_size)
    valid_low = F.adaptive_max_pool2d(object_mask, output_size=output_size)
    target_low = target_low * valid_low

    prediction = torch.sigmoid(alignment_logits) >= threshold
    prediction = prediction & (valid_low >= 0.5)
    target = target_low >= 0.5

    prediction = prediction.flatten(1)
    target = target.flatten(1)

    intersection = (prediction & target).sum(dim=1).float()
    union = (prediction | target).sum(dim=1).float()
    return (intersection + eps) / (union + eps)


def get_trainable_state(model):
    return {
        "visual_projection": model.visual_projection.state_dict(),
        "text_projection": model.text_projection.state_dict(),
        "text_decoder_projection": model.text_decoder_projection.state_dict(),
        "decoder": model.decoder.state_dict(),
    }


def run_epoch(
    model,
    mode,
    loader,
    text_encoder,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "seg_loss": 0.0,
        "align_loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "alignment_iou": 0.0,
    }
    total_samples = 0

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(DEVICE, non_blocking=True)
        objects = batch["object_mask"].to(DEVICE, non_blocking=True)
        targets = batch["part_mask"].to(DEVICE, non_blocking=True)
        relative_u = batch["relative_u"].to(DEVICE, non_blocking=True)
        relative_v = batch["relative_v"].to(DEVICE, non_blocking=True)
        text_embeddings = text_encoder(batch["query"])

        batch_size = images.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits, aux = model(
                    images,
                    text_embeddings,
                    objects,
                    relative_u,
                    relative_v,
                )
                losses = compute_total_alignment_loss(
                    mode,
                    logits,
                    aux,
                    targets,
                    objects,
                )

            if training:
                if scaler is not None:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    optimizer.step()

        with torch.no_grad():
            ious = batch_iou(logits, targets, MASK_THRESHOLD)
            dices = batch_dice(logits, targets, MASK_THRESHOLD)
            alignment_ious = batch_alignment_iou(
                aux["alignment_logits"],
                targets,
                objects,
            )

        totals["loss"] += losses["total"].item() * batch_size
        totals["seg_loss"] += losses["seg_total"].item() * batch_size
        totals["align_loss"] += losses["align_total"].item() * batch_size
        totals["iou"] += ious.sum().item()
        totals["dice"] += dices.sum().item()
        totals["alignment_iou"] += alignment_ious.sum().item()
        total_samples += batch_size

        if batch_index % 50 == 0 or batch_index == len(loader):
            phase = "Train" if training else "Validation"
            print(
                f"{phase} {batch_index}/{len(loader)} "
                f"Loss: {losses['total'].item():.4f} "
                f"IoU: {ious.mean().item():.4f} "
                f"Dice: {dices.mean().item():.4f} "
                f"AlignIoU: {alignment_ious.mean().item():.4f}"
            )

    return {key: value / total_samples for key, value in totals.items()}


def main():
    if MODE not in {
        "mask_baseline",
        "alignment_mask",
        "alignment_relative_uv",
    }:
        raise ValueError(f"Unknown mode: {MODE}")

    seed_everything(SEED)

    print("Device:", DEVICE)
    print("AMP:", USE_AMP)
    print("Mode:", MODE)

    experiment_dir = OUTPUT_ROOT / MODE
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AlignmentDataset("train_seen", IMAGE_SIZE)
    val_dataset = AlignmentDataset("validation_seen", IMAGE_SIZE)

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print("Loading DINOv2...")
    dino = load_dino_model(device=DEVICE)

    print("Loading CLIP...")
    clip_model, tokenizer = load_clip_model(device=DEVICE)
    text_encoder = TextEmbeddingCache(clip_model, tokenizer, DEVICE)

    print("Creating alignment model...")
    model = PartQueryAlignmentSegmenter(
        dino_encoder=dino,
        mode=MODE,
        visual_dim=VISUAL_DIM,
        text_common_dim=TEXT_COMMON_DIM,
        text_decoder_dim=TEXT_DECODER_DIM,
        temperature=ALIGNMENT_TEMPERATURE,
    ).to(DEVICE)

    trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Trainable parameters:", trainable_parameters)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    config = {
        "mode": MODE,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "alignment_temperature": ALIGNMENT_TEMPERATURE,
        "alignment_loss_weight": 0.20,
        "visual_dim": VISUAL_DIM,
        "text_common_dim": TEXT_COMMON_DIM,
        "text_decoder_dim": TEXT_DECODER_DIM,
        "dino": "dinov2_vits14",
        "clip": "ViT-B-32/openai",
    }

    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    history = []
    best_val_iou = -1.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        train_metrics = run_epoch(
            model,
            MODE,
            train_loader,
            text_encoder,
            optimizer=optimizer,
            scaler=scaler,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                MODE,
                val_loader,
                text_encoder,
            )

        scheduler.step(val_metrics["iou"])
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "train_alignment_iou": train_metrics["alignment_iou"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_alignment_iou": val_metrics["alignment_iou"],
            "learning_rate": current_lr,
            "seconds": elapsed,
        }
        history.append(row)

        print(
            f"[{MODE}] Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train IoU {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val IoU {val_metrics['iou']:.4f} | "
            f"val Dice {val_metrics['dice']:.4f} | "
            f"align IoU {val_metrics['alignment_iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "mode": MODE,
            "model_state": get_trainable_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        }

        torch.save(checkpoint, experiment_dir / "last.pt")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(checkpoint, experiment_dir / "best.pt")
            print("Saved new best checkpoint.")

        (experiment_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print()
    print("Training completed successfully.")
    print("Best validation IoU:", best_val_iou)


if __name__ == "__main__":
    main()
