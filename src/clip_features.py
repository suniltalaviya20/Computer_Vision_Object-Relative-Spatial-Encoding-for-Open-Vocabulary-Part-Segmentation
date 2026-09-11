import torch
import open_clip

from src.dino_features import get_device


def load_clip_model(
    device=None,
):
    if device is None:
        device = get_device()

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )

    tokenizer = open_clip.get_tokenizer(
        "ViT-B-32"
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.to(device)

    return model, tokenizer


def build_text_prompt(
    query,
):
    return f"a photo of the {query} of an object"


def extract_clip_features(
    model,
    tokenizer,
    query,
    device=None,
):
    if device is None:
        device = get_device()

    prompt = build_text_prompt(
        query
    )

    tokens = tokenizer(
        [prompt]
    ).to(device)

    with torch.no_grad():
        text_features = model.encode_text(
            tokens
        )

    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

    return text_features