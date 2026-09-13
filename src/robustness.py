import torch
import torch.nn.functional as F

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


ROTATION_ANGLES = [
    0,
    15,
    30,
    45,
    90,
]


MASK_NOISE_CONDITIONS = [
    "clean",
    "erode_5",
    "erode_11",
    "dilate_5",
    "dilate_11",
    "shift_5",
    "shift_15",
]


SHIFT_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
]


def rotate_sample(
    display_image,
    model_image,
    object_mask,
    part_mask,
    angle,
):
    rotated_display = TF.rotate(
        display_image,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    rotated_model = TF.rotate(
        model_image,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    rotated_object = TF.rotate(
        object_mask.float(),
        angle=angle,
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    ) > 0.5

    rotated_part = TF.rotate(
        part_mask.float(),
        angle=angle,
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    ) > 0.5

    return (
        rotated_display,
        rotated_model,
        rotated_object,
        rotated_part,
    )


def dilate_mask(
    mask,
    kernel_size,
):
    assert kernel_size % 2 == 1

    x = (
        mask.float()
        .unsqueeze(0)
    )

    dilated = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    return (
        dilated.squeeze(0)
        > 0.5
    )


def erode_mask(
    mask,
    kernel_size,
):
    assert kernel_size % 2 == 1

    x = (
        mask.float()
        .unsqueeze(0)
    )

    inverted = (
        1.0 - x
    )

    dilated_background = F.max_pool2d(
        inverted,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    eroded = (
        1.0
        - dilated_background
    )

    return (
        eroded.squeeze(0)
        > 0.5
    )


def shift_mask(
    mask,
    dx,
    dy,
):
    _, height, width = mask.shape

    output = torch.zeros_like(
        mask
    )

    src_x_start = max(
        0,
        -dx,
    )

    src_x_end = min(
        width,
        width - dx,
    )

    dst_x_start = max(
        0,
        dx,
    )

    dst_x_end = min(
        width,
        width + dx,
    )

    src_y_start = max(
        0,
        -dy,
    )

    src_y_end = min(
        height,
        height - dy,
    )

    dst_y_start = max(
        0,
        dy,
    )

    dst_y_end = min(
        height,
        height + dy,
    )

    if (
        src_x_start < src_x_end
        and src_y_start < src_y_end
    ):
        output[
            :,
            dst_y_start:dst_y_end,
            dst_x_start:dst_x_end,
        ] = mask[
            :,
            src_y_start:src_y_end,
            src_x_start:src_x_end,
        ]

    return output


def deterministic_shift(
    mask,
    magnitude,
    index,
):
    direction = SHIFT_DIRECTIONS[
        index
        % len(SHIFT_DIRECTIONS)
    ]

    dx = (
        direction[0]
        * magnitude
    )

    dy = (
        direction[1]
        * magnitude
    )

    return shift_mask(
        mask,
        dx=dx,
        dy=dy,
    )


def corrupt_object_mask(
    mask,
    condition,
    index,
):
    if condition == "clean":
        return mask.clone()

    if condition == "erode_5":
        return erode_mask(
            mask,
            kernel_size=5,
        )

    if condition == "erode_11":
        return erode_mask(
            mask,
            kernel_size=11,
        )

    if condition == "dilate_5":
        return dilate_mask(
            mask,
            kernel_size=5,
        )

    if condition == "dilate_11":
        return dilate_mask(
            mask,
            kernel_size=11,
        )

    if condition == "shift_5":
        return deterministic_shift(
            mask,
            magnitude=5,
            index=index,
        )

    if condition == "shift_15":
        return deterministic_shift(
            mask,
            magnitude=15,
            index=index,
        )

    raise ValueError(
        f"Unknown condition: {condition}"
    )


def binary_mask_iou(
    mask_a,
    mask_b,
    eps=1e-6,
):
    a = mask_a.bool()
    b = mask_b.bool()

    intersection = (
        a & b
    ).sum().float()

    union = (
        a | b
    ).sum().float()

    return (
        intersection + eps
    ) / (
        union + eps
    )