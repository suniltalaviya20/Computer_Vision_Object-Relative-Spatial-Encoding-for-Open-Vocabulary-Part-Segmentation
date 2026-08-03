from __future__ import annotations

import torch
from torch import Tensor


def create_relative_coordinate_maps(object_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Create object-relative horizontal and vertical coordinate maps.

    Args:
        object_mask:
            Binary tensor with shape [H, W]. Non-zero pixels belong to the
            selected parent object.

    Returns:
        u_map:
            Horizontal coordinates in [0, 1] relative to the object's
            bounding box.

        v_map:
            Vertical coordinates in [0, 1] relative to the object's
            bounding box.

        Values outside the object mask are set to zero.
    """
    if object_mask.ndim != 2:
        raise ValueError(
            f"Expected object_mask with shape [H, W], "
            f"but received {tuple(object_mask.shape)}."
        )

    mask = object_mask.bool()
    height, width = mask.shape

    object_pixels = torch.nonzero(mask, as_tuple=False)

    # Return empty maps if no object pixels exist.
    if object_pixels.numel() == 0:
        empty = torch.zeros(
            (height, width),
            dtype=torch.float32,
            device=object_mask.device,
        )
        return empty.clone(), empty.clone()

    y_coordinates = object_pixels[:, 0]
    x_coordinates = object_pixels[:, 1]

    x_min = x_coordinates.min()
    x_max = x_coordinates.max()
    y_min = y_coordinates.min()
    y_max = y_coordinates.max()

    yy, xx = torch.meshgrid(
        torch.arange(height, device=object_mask.device),
        torch.arange(width, device=object_mask.device),
        indexing="ij",
    )

    object_width = torch.clamp(x_max - x_min, min=1).float()
    object_height = torch.clamp(y_max - y_min, min=1).float()

    u_map = (xx.float() - x_min.float()) / object_width
    v_map = (yy.float() - y_min.float()) / object_height

    # Restrict the values to the object and remove numerical overflow.
    u_map = torch.clamp(u_map, 0.0, 1.0) * mask.float()
    v_map = torch.clamp(v_map, 0.0, 1.0) * mask.float()

    return u_map, v_map