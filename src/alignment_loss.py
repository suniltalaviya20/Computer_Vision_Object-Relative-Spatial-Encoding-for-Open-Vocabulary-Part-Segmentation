import torch
import torch.nn.functional as F

from src.metrics import (
    segmentation_loss,
)


ALIGNMENT_LOSS_WEIGHT = 0.20


def create_alignment_targets(
    part_mask,
    object_mask,
    output_size,
):
    target_low = F.adaptive_max_pool2d(
        part_mask,
        output_size=output_size,
    )

    valid_low = F.adaptive_max_pool2d(
        object_mask,
        output_size=output_size,
    )

    target_low = (
        target_low
        * valid_low
    )

    return (
        target_low,
        valid_low,
    )


def masked_alignment_bce(
    logits,
    target,
    valid_mask,
):
    loss_map = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    )

    weighted = (
        loss_map
        * valid_mask
    )

    return (
        weighted.sum()
        /
        valid_mask.sum().clamp_min(
            1.0
        )
    )


def masked_alignment_dice_loss(
    logits,
    target,
    valid_mask,
    eps=1e-6,
):
    probability = (
        torch.sigmoid(
            logits
        )
        * valid_mask
    )

    target = (
        target
        * valid_mask
    )

    probability = (
        probability.flatten(1)
    )

    target = (
        target.flatten(1)
    )

    intersection = (
        probability
        * target
    ).sum(
        dim=1
    )

    denominator = (
        probability.sum(
            dim=1
        )
        + target.sum(
            dim=1
        )
    )

    dice = (
        2.0 * intersection
        + eps
    ) / (
        denominator
        + eps
    )

    return (
        1.0 - dice
    ).mean()


def alignment_loss(
    alignment_logits,
    part_mask,
    object_mask,
):
    output_size = (
        alignment_logits.shape[
            -2:
        ]
    )

    (
        target_low,
        valid_low,
    ) = create_alignment_targets(
        part_mask,
        object_mask,
        output_size,
    )

    bce = masked_alignment_bce(
        alignment_logits,
        target_low,
        valid_low,
    )

    dice = (
        masked_alignment_dice_loss(
            alignment_logits,
            target_low,
            valid_low,
        )
    )

    return {
        "align_total":
            bce + dice,

        "align_bce":
            bce,

        "align_dice":
            dice,

        "target_low":
            target_low,

        "valid_low":
            valid_low,
    }


def compute_total_alignment_loss(
    mode,
    logits,
    aux,
    part_mask,
    object_mask,
):
    (
        seg_total,
        seg_bce,
        seg_dice,
    ) = segmentation_loss(
        logits,
        part_mask,
    )

    align = alignment_loss(
        aux[
            "alignment_logits"
        ],
        part_mask,
        object_mask,
    )

    if mode == "mask_baseline":
        total = seg_total

    else:
        total = (
            seg_total
            + ALIGNMENT_LOSS_WEIGHT
            * align[
                "align_total"
            ]
        )

    return {
        "total":
            total,

        "seg_total":
            seg_total,

        "seg_bce":
            seg_bce,

        "seg_dice":
            seg_dice,

        "align_total":
            align[
                "align_total"
            ],

        "align_bce":
            align[
                "align_bce"
            ],

        "align_dice":
            align[
                "align_dice"
            ],

        "alignment_target":
            align[
                "target_low"
            ],

        "alignment_valid":
            align[
                "valid_low"
            ],
    }