import torch
import torch.nn.functional as F


def dice_loss(
    logits,
    targets,
    smooth=1.0,
):
    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(
        start_dim=1
    )

    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        probabilities
        * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return 1.0 - dice.mean()


def segmentation_loss(
    logits,
    targets,
):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    total = bce + dice

    return total, bce, dice


def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1.0,
):
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    )

    targets = (
        targets >= 0.5
    )

    predictions = predictions.flatten(
        start_dim=1
    )

    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        predictions
        & targets
    ).sum(dim=1).float()

    denominator = (
        predictions.sum(dim=1).float()
        + targets.sum(dim=1).float()
    )

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )

    return dice.mean()


def iou_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1.0,
):
    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    )

    targets = (
        targets >= 0.5
    )

    predictions = predictions.flatten(
        start_dim=1
    )

    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        predictions
        & targets
    ).sum(dim=1).float()

    union = (
        predictions
        | targets
    ).sum(dim=1).float()

    iou = (
        intersection
        + smooth
    ) / (
        union
        + smooth
    )

    return iou.mean()