from __future__ import annotations

from collections.abc import Sequence

import clip
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen OpenAI CLIP text encoder."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        prompt_template: str = "a photo of the {} of an object",
    ) -> None:
        super().__init__()

        clip_model, _ = clip.load(
            model_name,
            device="cpu",
            jit=False,
        )

        self.clip_model = clip_model
        self.prompt_template = prompt_template

        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        # encode_text() returns vectors with this dimension.
        self.output_dim = int(
            self.clip_model.text_projection.shape[1]
        )

    def train(self, mode: bool = True):
        """Keep CLIP in evaluation mode when the full model trains."""

        super().train(mode)
        self.clip_model.eval()

        return self

    @torch.no_grad()
    def forward(
        self,
        part_names: Sequence[str],
    ) -> Tensor:
        if not part_names:
            raise ValueError(
                "part_names must contain at least one query."
            )

        prompts = [
            self.prompt_template.format(part_name)
            for part_name in part_names
        ]

        device = next(
            self.clip_model.parameters()
        ).device

        tokens = clip.tokenize(
            prompts,
            truncate=True,
        ).to(device)

        text_features = self.clip_model.encode_text(
            tokens
        )

        # CLIP may return float16 features on GPU.
        text_features = text_features.float()

        # Normalized vectors are easier for the projection layer to use.
        text_features = F.normalize(
            text_features,
            dim=-1,
        )

        return text_features