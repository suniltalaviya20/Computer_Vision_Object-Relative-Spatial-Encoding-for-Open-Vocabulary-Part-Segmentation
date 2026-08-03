from __future__ import annotations

import torch
from torch import Tensor


def calculate_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Calculate mean binary IoU and Dice over a batch."""

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have identical shapes. "
            f"Received {tuple(logits.shape)} and "
            f"{tuple(targets.shape)}."
        )

    probabilities = torch.sigmoid(logits)

    predictions = probabilities >= threshold
    targets_binary = targets >= 0.5

    dimensions = tuple(
        range(1, predictions.ndim)
    )

    intersection = (
        predictions & targets_binary
    ).sum(dim=dimensions).float()

    union = (
        predictions | targets_binary
    ).sum(dim=dimensions).float()

    predicted_pixels = predictions.sum(
        dim=dimensions
    ).float()

    target_pixels = targets_binary.sum(
        dim=dimensions
    ).float()

    iou = torch.where(
        union > 0,
        intersection / (union + epsilon),
        torch.ones_like(union),
    )

    dice_denominator = (
        predicted_pixels + target_pixels
    )

    dice = torch.where(
        dice_denominator > 0,
        (2.0 * intersection)
        / (dice_denominator + epsilon),
        torch.ones_like(dice_denominator),
    )

    return {
        "iou": float(iou.mean().detach().cpu()),
        "dice": float(dice.mean().detach().cpu()),
    }