import torch
import torch.nn.functional as F

from src.preprocessing import (
    get_resize_info,
)


def paste_crop_prediction(
    crop_prediction,
    original_height,
    original_width,
    crop_x1,
    crop_y1,
    crop_side,
):
    if crop_prediction.ndim == 2:
        crop_prediction = (
            crop_prediction.unsqueeze(0)
        )

    if crop_prediction.ndim == 3:
        crop_prediction = (
            crop_prediction.unsqueeze(0)
        )

    resized_crop = F.interpolate(
        crop_prediction.float(),
        size=(
            crop_side,
            crop_side,
        ),
        mode="bilinear",
        align_corners=False,
    )

    resized_crop = (
        resized_crop.squeeze(0)
    )

    full_prediction = torch.zeros(
        (
            1,
            original_height,
            original_width,
        ),
        dtype=resized_crop.dtype,
    )

    crop_x2 = (
        crop_x1
        + crop_side
    )

    crop_y2 = (
        crop_y1
        + crop_side
    )

    source_x1 = max(
        0,
        -crop_x1,
    )

    source_y1 = max(
        0,
        -crop_y1,
    )

    source_x2 = min(
        crop_side,
        original_width
        - crop_x1,
    )

    source_y2 = min(
        crop_side,
        original_height
        - crop_y1,
    )

    destination_x1 = max(
        0,
        crop_x1,
    )

    destination_y1 = max(
        0,
        crop_y1,
    )

    destination_x2 = min(
        original_width,
        crop_x2,
    )

    destination_y2 = min(
        original_height,
        crop_y2,
    )

    if (
        source_x1 < source_x2
        and
        source_y1 < source_y2
        and
        destination_x1 < destination_x2
        and
        destination_y1 < destination_y2
    ):
        full_prediction[
            :,
            destination_y1:
            destination_y2,
            destination_x1:
            destination_x2,
        ] = resized_crop[
            :,
            source_y1:
            source_y2,
            source_x1:
            source_x2,
        ]

    return full_prediction


def preprocess_prediction_to_full_view(
    full_prediction,
    target_size=224,
):
    if full_prediction.ndim == 2:
        full_prediction = (
            full_prediction.unsqueeze(0)
        )

    _, height, width = (
        full_prediction.shape
    )

    resize_info = get_resize_info(
        height,
        width,
        target_size,
    )

    resized = F.interpolate(
        full_prediction.unsqueeze(0),
        size=(
            resize_info["new_h"],
            resize_info["new_w"],
        ),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    padded = F.pad(
        resized,
        (
            resize_info["left"],
            resize_info["right"],
            resize_info["top"],
            resize_info["bottom"],
        ),
        value=0.0,
    )

    return padded


def project_crop_prediction_to_full_view(
    crop_prediction,
    original_height,
    original_width,
    crop_x1,
    crop_y1,
    crop_side,
    target_size=224,
):
    full_prediction = (
        paste_crop_prediction(
            crop_prediction,
            original_height,
            original_width,
            crop_x1,
            crop_y1,
            crop_side,
        )
    )

    full_view_prediction = (
        preprocess_prediction_to_full_view(
            full_prediction,
            target_size=target_size,
        )
    )

    return full_view_prediction