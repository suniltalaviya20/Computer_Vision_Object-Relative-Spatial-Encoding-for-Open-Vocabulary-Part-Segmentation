import torch


def create_absolute_xy(
    object_mask,
):
    if object_mask.ndim == 3:
        mask = object_mask.squeeze(0)
    else:
        mask = object_mask

    mask = mask.bool()

    height, width = mask.shape

    y = torch.linspace(
        0.0,
        1.0,
        height,
        dtype=torch.float32,
    )

    x = torch.linspace(
        0.0,
        1.0,
        width,
        dtype=torch.float32,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    mask_float = mask.float()

    absolute_x = (
        xx * mask_float
    )

    absolute_y = (
        yy * mask_float
    )

    return (
        absolute_x.unsqueeze(0),
        absolute_y.unsqueeze(0),
    )


def create_relative_uv(
    object_mask,
):
    if object_mask.ndim == 3:
        mask = object_mask.squeeze(0)
    else:
        mask = object_mask

    mask = mask.bool()

    height, width = mask.shape

    u_map = torch.zeros(
        (height, width),
        dtype=torch.float32,
    )

    v_map = torch.zeros(
        (height, width),
        dtype=torch.float32,
    )

    ys, xs = torch.where(
        mask
    )

    if len(xs) == 0:
        return (
            u_map.unsqueeze(0),
            v_map.unsqueeze(0),
        )

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    x_denom = max(
        int(x_max - x_min),
        1,
    )

    y_denom = max(
        int(y_max - y_min),
        1,
    )

    u_values = (
        xs.float()
        - x_min.float()
    ) / x_denom

    v_values = (
        ys.float()
        - y_min.float()
    ) / y_denom

    u_map[
        ys,
        xs,
    ] = u_values

    v_map[
        ys,
        xs,
    ] = v_values

    return (
        u_map.unsqueeze(0),
        v_map.unsqueeze(0),
    )

def create_boundary_distance(
    object_mask,
):
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    if object_mask.ndim == 3:
        mask = object_mask.squeeze(0)
    else:
        mask = object_mask

    mask = mask.bool()

    mask_numpy = (
        mask
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    if mask_numpy.sum() == 0:
        return torch.zeros(
            (1, *mask_numpy.shape),
            dtype=torch.float32,
        )

    distance = distance_transform_edt(
        mask_numpy
    ).astype(np.float32)

    max_distance = distance.max()

    if max_distance > 0:
        distance = (
            distance
            / max_distance
        )

    distance = (
        distance
        * mask_numpy
    )

    return torch.from_numpy(
        distance
    ).unsqueeze(0)
