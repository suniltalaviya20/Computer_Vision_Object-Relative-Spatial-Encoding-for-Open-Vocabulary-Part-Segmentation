from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def binary_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Pixel-wise binary cross-entropy using raw logits."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits and targets must have the same shape, "
            f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
        )

    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
    )


def dice_loss(
    logits: Tensor,
    targets: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    """Soft Dice loss for binary segmentation."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits and targets must have the same shape, "
            f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
        )

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice_score.mean()


def containment_loss(
    logits: Tensor,
    object_mask: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Penalize predicted part probability outside the parent object."""

    if logits.shape != object_mask.shape:
        raise ValueError(
            f"Logits and object mask must have the same shape, "
            f"received {tuple(logits.shape)} "
            f"and {tuple(object_mask.shape)}."
        )

    probabilities = torch.sigmoid(logits)

    outside_probability = (
        probabilities * (1.0 - object_mask)
    ).sum(dim=(1, 2, 3))

    total_probability = probabilities.sum(
        dim=(1, 2, 3)
    )

    loss_per_sample = outside_probability / (
        total_probability + epsilon
    )

    return loss_per_sample.mean()


def combined_segmentation_loss(
    logits: Tensor,
    targets: Tensor,
    object_mask: Tensor,
    containment_weight: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Combine BCE, Dice, and containment losses."""

    bce = binary_cross_entropy_loss(
        logits,
        targets,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    containment = containment_loss(
        logits,
        object_mask,
    )

    total = (
        bce
        + dice
        + containment_weight * containment
    )

    components = {
        "total": float(total.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "containment": float(containment.detach()),
    }

    return total, components