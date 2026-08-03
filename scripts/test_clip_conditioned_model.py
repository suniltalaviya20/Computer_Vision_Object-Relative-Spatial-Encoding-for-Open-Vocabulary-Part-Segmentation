from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.part_dataset import PascalPartQueryDataset
from models.clip_conditioned_model import CLIPConditionedPartModel


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    model = CLIPConditionedPartModel(
        clip_model_name="ViT-B/32",
        text_channels=32,
    ).to(device)

    images = batch["image"].to(device)
    object_masks = batch["object_mask"].to(device)
    u_maps = batch["u_map"].to(device)
    v_maps = batch["v_map"].to(device)

    part_names = list(
        batch["part_name"]
    )

    with torch.no_grad():
        logits = model(
            image=images,
            object_mask=object_masks,
            u_map=u_maps,
            v_map=v_maps,
            part_names=part_names,
        )

    print("Part names:", part_names)
    print("Images:", tuple(images.shape))
    print("Logits:", tuple(logits.shape))

    print(
        "Frozen CLIP parameters:",
        sum(
            parameter.numel()
            for parameter in model.text_encoder.parameters()
            if not parameter.requires_grad
        ),
    )

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )


if __name__ == "__main__":
    main()