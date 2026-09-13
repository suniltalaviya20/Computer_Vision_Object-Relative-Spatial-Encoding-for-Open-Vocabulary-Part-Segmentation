import math

import torch
import torch.nn.functional as F

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(
    3,
    1,
    1,
)


IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(
    3,
    1,
    1,
)


CROP_CONTEXT_RATIO = 0.15


def get_square_object_crop(
    object_mask,
    context_ratio=0.15,
):
    if object_mask.ndim == 3:
        mask = object_mask.squeeze(0)

    else:
        mask = object_mask

    mask = mask.bool()

    height, width = mask.shape

    ys, xs = torch.where(
        mask
    )


    if len(xs) == 0:
        side = max(
            height,
            width,
        )

        return {
            "x1":
                (
                    width
                    - side
                )
                // 2,

            "y1":
                (
                    height
                    - side
                )
                // 2,

            "side":
                side,
        }


    x_min = xs.min().item()

    x_max = (
        xs.max().item()
        + 1
    )

    y_min = ys.min().item()

    y_max = (
        ys.max().item()
        + 1
    )


    box_width = (
        x_max
        - x_min
    )

    box_height = (
        y_max
        - y_min
    )


    base_side = max(
        box_width,
        box_height,
    )


    side = math.ceil(
        base_side
        * (
            1
            + 2
            * context_ratio
        )
    )


    side = max(
        side,
        1,
    )


    center_x = (
        x_min
        + x_max
    ) / 2


    center_y = (
        y_min
        + y_max
    ) / 2


    x1 = math.floor(
        center_x
        - side / 2
    )


    y1 = math.floor(
        center_y
        - side / 2
    )


    return {
        "x1":
            x1,

        "y1":
            y1,

        "side":
            side,
    }


def square_crop_with_padding(
    tensor,
    x1,
    y1,
    side,
    fill_value=0,
):
    original_ndim = (
        tensor.ndim
    )


    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(
            0
        )


    channels, height, width = (
        tensor.shape
    )


    output = torch.full(
        (
            channels,
            side,
            side,
        ),
        fill_value=fill_value,
        dtype=tensor.dtype,
    )


    x2 = (
        x1
        + side
    )

    y2 = (
        y1
        + side
    )


    source_x1 = max(
        0,
        x1,
    )

    source_y1 = max(
        0,
        y1,
    )

    source_x2 = min(
        width,
        x2,
    )

    source_y2 = min(
        height,
        y2,
    )


    destination_x1 = (
        source_x1
        - x1
    )

    destination_y1 = (
        source_y1
        - y1
    )


    destination_x2 = (
        destination_x1
        + source_x2
        - source_x1
    )

    destination_y2 = (
        destination_y1
        + source_y2
        - source_y1
    )


    if (
        source_x1
        < source_x2
        and
        source_y1
        < source_y2
    ):
        output[
            :,
            destination_y1:
            destination_y2,
            destination_x1:
            destination_x2,
        ] = tensor[
            :,
            source_y1:
            source_y2,
            source_x1:
            source_x2,
        ]


    if original_ndim == 2:
        output = output.squeeze(
            0
        )


    return output


def prepare_crop_image(
    crop_image,
    target_size=224,
):
    crop_image = (
        crop_image.float()
        / 255.0
    )


    display_image = TF.resize(
        crop_image,
        [
            target_size,
            target_size,
        ],
        interpolation=(
            InterpolationMode.BILINEAR
        ),
        antialias=True,
    )


    model_image = (
        display_image
        - IMAGENET_MEAN
    ) / IMAGENET_STD


    return (
        display_image,
        model_image,
    )


def prepare_crop_mask(
    crop_mask,
    target_size=224,
    preserve_positive=False,
):
    if crop_mask.ndim == 2:
        crop_mask = (
            crop_mask.unsqueeze(0)
        )


    crop_mask = (
        crop_mask.float()
    )


    resized = TF.resize(
        crop_mask,
        [
            target_size,
            target_size,
        ],
        interpolation=(
            InterpolationMode.NEAREST
        ),
    )


    if (
        preserve_positive
        and crop_mask.sum() > 0
        and resized.sum() == 0
    ):
        resized = F.interpolate(
            crop_mask.unsqueeze(0),
            size=(
                target_size,
                target_size,
            ),
            mode="area",
        ).squeeze(0)


        resized = (
            resized > 0
        ).float()


    return (
        resized > 0.5
    )


def count_positive_dino_patches(
    mask,
    grid_size=16,
):
    if mask.ndim == 3:
        mask = mask.unsqueeze(
            0
        )


    coarse = F.adaptive_max_pool2d(
        mask.float(),
        output_size=(
            grid_size,
            grid_size,
        ),
    )


    count = (
        coarse > 0
    ).sum().item()


    return (
        int(count),
        coarse.squeeze(),
    )