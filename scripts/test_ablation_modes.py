from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.dino_clip_part_model import (
    ABLATION_SETTINGS,
    DINOCLIPPartModel,
)
from utils.losses import combined_segmentation_loss


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

    model = DINOCLIPPartModel(
        dino_model_name="dinov2_vits14",
        clip_model_name="ViT-B/32",
        visual_channels=128,
        text_channels=32,
        ablation_mode="full",
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", trainable_parameters)

    print("\nTesting output shapes")
    print("-" * 60)

    model.eval()

    outputs: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for mode in ABLATION_SETTINGS:
            model.set_ablation(mode)

            logits = model(
                image=images,
                object_mask=object_masks,
                u_map=u_maps,
                v_map=v_maps,
                part_names=part_names,
            )

            outputs[mode] = logits.detach().cpu()

            print(
                f"{mode:15s} | "
                f"logits {tuple(logits.shape)} | "
                f"min {float(logits.min()):.4f} | "
                f"max {float(logits.max()):.4f}"
            )

            if logits.shape != target_masks.shape:
                raise RuntimeError(
                    f"Wrong output shape for mode {mode}."
                )

    print("\nTesting whether disabled inputs are ignored")
    print("-" * 60)

    random_masks = torch.rand_like(
        object_masks
    )

    random_u_maps = torch.rand_like(
        u_maps
    )

    random_v_maps = torch.rand_like(
        v_maps
    )

    # -------------------------------------------------------------
    # Part-only mode should ignore mask, U and V completely.
    # -------------------------------------------------------------

    model.set_ablation("part_only")

    with torch.no_grad():
        normal_part_only = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_names=part_names,
        )

        changed_part_only = model(
            image=images,
            object_mask=random_masks,
            u_map=random_u_maps,
            v_map=random_v_maps,
            part_names=part_names,
        )

    part_only_difference = (
        normal_part_only
        - changed_part_only
    ).abs().max()

    print(
        "Part-only maximum difference after changing geometry:",
        float(part_only_difference),
    )

    if not torch.allclose(
        normal_part_only,
        changed_part_only,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "part_only is still using mask or coordinate inputs."
        )

    # -------------------------------------------------------------
    # Object-mask mode should ignore U and V.
    # -------------------------------------------------------------

    model.set_ablation("object_mask")

    with torch.no_grad():
        normal_object_mask = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_names=part_names,
        )

        changed_coordinates = model(
            image=images,
            object_mask=object_masks,
            u_map=random_u_maps,
            v_map=random_v_maps,
            part_names=part_names,
        )

    object_mask_difference = (
        normal_object_mask
        - changed_coordinates
    ).abs().max()

    print(
        "Object-mask maximum difference after changing U/V:",
        float(object_mask_difference),
    )

    if not torch.allclose(
        normal_object_mask,
        changed_coordinates,
        atol=1e-6,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "object_mask mode is still using U or V."
        )

    # -------------------------------------------------------------
    # Check gradients in the full model.
    # -------------------------------------------------------------

    print("\nTesting gradients")
    print("-" * 60)

    model.set_ablation("full")
    model.train()

    model.zero_grad(
        set_to_none=True,
    )

    logits = model(
        image=images,
        object_mask=object_masks,
        u_map=u_maps,
        v_map=v_maps,
        part_names=part_names,
    )

    loss, components = combined_segmentation_loss(
        logits=logits,
        targets=target_masks,
        object_mask=object_masks,
        containment_weight=0.1,
    )

    loss.backward()

    print("Loss:", components["total"])

    print(
        "DINOv2 gradients:",
        has_any_gradient(
            model.visual_encoder
        ),
    )

    print(
        "CLIP gradients:",
        has_any_gradient(
            model.text_encoder
        ),
    )

    print(
        "Visual projection gradients:",
        has_any_gradient(
            model.visual_projection
        ),
    )

    print(
        "Text projection gradients:",
        has_any_gradient(
            model.text_projection
        ),
    )

    print(
        "Decoder gradients:",
        has_any_gradient(
            model.decoder
        ),
    )

    if has_any_gradient(model.visual_encoder):
        raise RuntimeError(
            "DINOv2 must remain frozen."
        )

    if has_any_gradient(model.text_encoder):
        raise RuntimeError(
            "CLIP must remain frozen."
        )

    if not has_any_gradient(model.visual_projection):
        raise RuntimeError(
            "Visual projection received no gradients."
        )

    if not has_any_gradient(model.text_projection):
        raise RuntimeError(
            "Text projection received no gradients."
        )

    if not has_any_gradient(model.decoder):
        raise RuntimeError(
            "Decoder received no gradients."
        )

    print("\nAblation configuration:")
    print(model.get_config())

    print("\nAll ablation checks passed.")


if __name__ == "__main__":
    main()