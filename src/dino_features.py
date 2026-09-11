import torch


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_dino_model(
    device=None,
):
    if device is None:
        device = get_device()

    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)

    return model


def extract_dino_features(
    model,
    image,
    device=None,
):
    if device is None:
        device = get_device()

    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        features = model.forward_features(
            image
        )

    patch_tokens = features[
        "x_norm_patchtokens"
    ]

    batch_size = patch_tokens.shape[0]
    num_patches = patch_tokens.shape[1]
    feature_dim = patch_tokens.shape[2]

    grid_size = int(
        num_patches ** 0.5
    )

    dense_features = (
        patch_tokens
        .reshape(
            batch_size,
            grid_size,
            grid_size,
            feature_dim,
        )
        .permute(
            0,
            3,
            1,
            2,
        )
    )

    return dense_features