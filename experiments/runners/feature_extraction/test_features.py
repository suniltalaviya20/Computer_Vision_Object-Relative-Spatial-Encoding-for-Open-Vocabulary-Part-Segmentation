from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import PascalPart116Dataset

from experiments.implementations.features.preprocessing import (
    resize_and_pad_image,
)

from experiments.implementations.features.dino_features import (
    get_device,
    load_dino_model,
    extract_dino_features,
)

from experiments.implementations.features.clip_features import (
    load_clip_model,
    build_text_prompt,
    extract_clip_features,
)

def main():
    device = get_device()

    print("Device:", device)

    dataset = PascalPart116Dataset(
        split="train_seen"
    )

    print(
        "Dataset length:",
        len(dataset),
    )

    sample = dataset[0]

    print()
    print(
        "Object:",
        sample["object_name"],
    )

    print(
        "Part:",
        sample["part_name"],
    )

    print(
        "Query:",
        sample["query"],
    )

    print(
        "Prompt:",
        build_text_prompt(
            sample["query"]
        ),
    )

    _, dino_input, _ = (
        resize_and_pad_image(
            sample["image"]
        )
    )

    dino_model = (
        load_dino_model(
            device=device
        )
    )

    dino_features = (
        extract_dino_features(
            dino_model,
            dino_input,
            device=device,
        )
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=device
        )
    )

    clip_features = (
        extract_clip_features(
            clip_model,
            tokenizer,
            sample["query"],
            device=device,
        )
    )

    print()
    print(
        "DINO features:",
        dino_features.shape,
    )

    print(
        "CLIP features:",
        clip_features.shape,
    )

    print()
    print(
        "DINO finite:",
        torch.isfinite(
            dino_features
        ).all(),
    )

    print(
        "CLIP finite:",
        torch.isfinite(
            clip_features
        ).all(),
    )

    print(
        "CLIP norm:",
        clip_features.norm(
            dim=-1
        ),
    )

    dino_trainable = sum(
        parameter.numel()
        for parameter
        in dino_model.parameters()
        if parameter.requires_grad
    )

    clip_trainable = sum(
        parameter.numel()
        for parameter
        in clip_model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Trainable DINO parameters:",
        dino_trainable,
    )

    print(
        "Trainable CLIP parameters:",
        clip_trainable,
    )

    assert dino_features.shape == (
        1,
        384,
        16,
        16,
    )

    assert clip_features.shape == (
        1,
        512,
    )

    assert torch.isfinite(
        dino_features
    ).all()

    assert torch.isfinite(
        clip_features
    ).all()

    assert dino_trainable == 0
    assert clip_trainable == 0

    print()
    print(
        "Feature pipeline checks passed."
    )


if __name__ == "__main__":
    main()