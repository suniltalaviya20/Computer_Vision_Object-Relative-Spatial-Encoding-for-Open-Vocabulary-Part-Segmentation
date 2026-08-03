from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.clip_text_encoder import FrozenCLIPTextEncoder


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    encoder = FrozenCLIPTextEncoder(
        model_name="ViT-B/32",
    ).to(device)

    queries = [
        "wheel",
        "seat",
        "head",
        "leg",
    ]

    embeddings = encoder(queries)

    print("Queries:", queries)
    print("Embedding shape:", tuple(embeddings.shape))
    print(
        "Trainable CLIP parameters:",
        sum(
            parameter.numel()
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ),
    )

    norms = embeddings.norm(
        dim=-1
    )

    print("Embedding norms:", norms.cpu())

    similarity = embeddings @ embeddings.T

    print("\nCosine-similarity matrix:")
    print(similarity.cpu())


if __name__ == "__main__":
    main()