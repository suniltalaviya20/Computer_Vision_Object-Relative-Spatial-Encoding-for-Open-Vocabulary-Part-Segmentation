from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.dino_clip_part_model import DINOCLIPPartModel
from utils.losses import combined_segmentation_loss


def count_parameters(
    module: torch.nn.Module,
    trainable_only: bool,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if (
            not trainable_only
            or parameter.requires_grad
        )
    )


def has_any_gradient(
    module: torch.nn.Module,
) -> bool:
    return any(
        parameter.grad is not None
        for parameter in module.parameters()
    )


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    dataset = PascalPartQueryDataset(
        manifest_path="data/manifests/train.json",
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"].to(device)
    object_masks = batch["object_mask"].to(device)
    target_masks = batch["part_mask"].to(device)
    u_maps = batch["u_map"].to(device)
    v_maps = batch["v_map"].to(device)

    part_names = list(
        batch["part_name"]
    )

    print("\nBatch")
    print("-" * 50)
    print("Images:", tuple(images.shape))
    print("Object masks:", tuple(object_masks.shape))
    print("Target masks:", tuple(target_masks.shape))
    print("U maps:", tuple(u_maps.shape))
    print("V maps:", tuple(v_maps.shape))
    print("Part names:", part_names)

    model = DINOCLIPPartModel(
        dino_model_name="dinov2_vits14",
        clip_model_name="ViT-B/32",
        visual_channels=128,
        text_channels=32,
    ).to(device)

    print("\nParameter counts")
    print("-" * 50)

    print(
        "DINOv2 total:",
        count_parameters(
            model.visual_encoder,
            trainable_only=False,
        ),
    )

    print(
        "DINOv2 trainable:",
        count_parameters(
            model.visual_encoder,
            trainable_only=True,
        ),
    )

    print(
        "CLIP total:",
        count_parameters(
            model.text_encoder,
            trainable_only=False,
        ),
    )

    print(
        "CLIP trainable:",
        count_parameters(
            model.text_encoder,
            trainable_only=True,
        ),
    )

    print(
        "Entire model trainable:",
        count_parameters(
            model,
            trainable_only=True,
        ),
    )

    model.train()

    logits = model(
        image=images,
        object_mask=object_masks,
        u_map=u_maps,
        v_map=v_maps,
        part_names=part_names,
    )

    print("\nForward pass")
    print("-" * 50)
    print("Logits:", tuple(logits.shape))
    print("Logit minimum:", float(logits.min()))
    print("Logit maximum:", float(logits.max()))

    expected_shape = tuple(target_masks.shape)

    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected logits shape {expected_shape}, "
            f"received {tuple(logits.shape)}."
        )

    loss, components = combined_segmentation_loss(
        logits=logits,
        targets=target_masks,
        object_mask=object_masks,
        containment_weight=0.1,
    )

    model.zero_grad(
        set_to_none=True,
    )

    loss.backward()

    print("\nLoss")
    print("-" * 50)

    for name, value in components.items():
        print(f"{name}: {value:.6f}")

    print("\nGradient checks")
    print("-" * 50)

    dino_has_gradients = has_any_gradient(
        model.visual_encoder
    )

    clip_has_gradients = has_any_gradient(
        model.text_encoder
    )

    visual_projection_has_gradients = (
        has_any_gradient(
            model.visual_projection
        )
    )

    text_projection_has_gradients = (
        has_any_gradient(
            model.text_projection
        )
    )

    decoder_has_gradients = has_any_gradient(
        model.decoder
    )

    print(
        "DINOv2 gradients:",
        dino_has_gradients,
    )

    print(
        "CLIP gradients:",
        clip_has_gradients,
    )

    print(
        "Visual projection gradients:",
        visual_projection_has_gradients,
    )

    print(
        "Text projection gradients:",
        text_projection_has_gradients,
    )

    print(
        "Decoder gradients:",
        decoder_has_gradients,
    )

    if dino_has_gradients:
        raise RuntimeError(
            "DINOv2 must remain frozen."
        )

    if clip_has_gradients:
        raise RuntimeError(
            "CLIP must remain frozen."
        )

    if not visual_projection_has_gradients:
        raise RuntimeError(
            "Visual projection received no gradients."
        )

    if not text_projection_has_gradients:
        raise RuntimeError(
            "Text projection received no gradients."
        )

    if not decoder_has_gradients:
        raise RuntimeError(
            "Decoder received no gradients."
        )

    print(
        "\nAll combined-model checks passed."
    )


if __name__ == "__main__":
    main()