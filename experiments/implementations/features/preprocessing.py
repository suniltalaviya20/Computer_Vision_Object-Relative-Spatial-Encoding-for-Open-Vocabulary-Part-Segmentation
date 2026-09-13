import torch
import torch.nn.functional as F

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)


IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


def resize_and_pad_image(
    image,
    target_size=224,
):
    """
    Resize an image while preserving
    its aspect ratio, then pad it to
    a square image.

    Parameters
    ----------
    image:
        uint8 tensor [3, H, W]

    target_size:
        final image size

    Returns
    -------
    display_image:
        float32 tensor
        [3, target_size, target_size]

    normalized_image:
        ImageNet-normalized tensor
        for DINOv2

    resize_info:
        resize and padding information
    """

    image = image.float() / 255.0

    _, h, w = image.shape

    scale = min(
        target_size / h,
        target_size / w,
    )

    new_h = max(
        1,
        round(h * scale),
    )

    new_w = max(
        1,
        round(w * scale),
    )

    resized = TF.resize(
        image,
        [new_h, new_w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    top = pad_h // 2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    display_image = F.pad(
        resized,
        (
            left,
            right,
            top,
            bottom,
        ),
        value=0.0,
    )

    normalized_image = (
        display_image - IMAGENET_MEAN
    ) / IMAGENET_STD

    resize_info = {
        "original_height": h,
        "original_width": w,
        "resized_height": new_h,
        "resized_width": new_w,
        "scale": scale,
        "pad_top": top,
        "pad_bottom": bottom,
        "pad_left": left,
        "pad_right": right,
    }

    return (
        display_image,
        normalized_image,
        resize_info,
    )

def get_resize_info(
    height,
    width,
    target_size=224,
):
    scale = min(
        target_size / height,
        target_size / width,
    )

    new_h = max(
        1,
        round(height * scale),
    )

    new_w = max(
        1,
        round(width * scale),
    )

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    top = pad_h // 2
    bottom = pad_h - top

    left = pad_w // 2
    right = pad_w - left

    return {
        "new_h": new_h,
        "new_w": new_w,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "scale": scale,
    }


def preprocess_segmentation_image(
    image,
    target_size=224,
):
    image = image.float() / 255.0

    _, h, w = image.shape

    info = get_resize_info(
        h,
        w,
        target_size,
    )

    resized = TF.resize(
        image,
        [
            info["new_h"],
            info["new_w"],
        ],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    display_image = F.pad(
        resized,
        (
            info["left"],
            info["right"],
            info["top"],
            info["bottom"],
        ),
        value=0.0,
    )

    normalized = (
        resized - IMAGENET_MEAN
    ) / IMAGENET_STD

    model_image = F.pad(
        normalized,
        (
            info["left"],
            info["right"],
            info["top"],
            info["bottom"],
        ),
        value=0.0,
    )

    return (
        display_image,
        model_image,
        info,
    )


def preprocess_mask(
    mask,
    resize_info,
    preserve_positive=False,
):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    mask = mask.float()

    resized = TF.resize(
        mask,
        [
            resize_info["new_h"],
            resize_info["new_w"],
        ],
        interpolation=InterpolationMode.NEAREST,
    )

    if (
        preserve_positive
        and mask.sum() > 0
        and resized.sum() == 0
    ):
        resized = F.interpolate(
            mask.unsqueeze(0),
            size=(
                resize_info["new_h"],
                resize_info["new_w"],
            ),
            mode="area",
        ).squeeze(0)

        resized = (
            resized > 0
        ).float()

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

    return padded > 0.5