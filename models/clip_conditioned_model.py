from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.clip_text_encoder import FrozenCLIPTextEncoder
from models.decoder import TinyPartDecoder


class CLIPConditionedPartModel(nn.Module):
    """Part segmenter conditioned on frozen CLIP text features."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        text_channels: int = 32,
    ) -> None:
        super().__init__()

        self.text_channels = text_channels

        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=clip_model_name,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                self.text_encoder.output_dim,
                128,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                128,
                text_channels,
            ),
        )

        # RGB: 3
        # Object mask: 1
        # Relative U: 1
        # Relative V: 1
        # Projected CLIP text: text_channels
        input_channels = 6 + text_channels

        self.decoder = TinyPartDecoder(
            input_channels=input_channels,
        )

    def forward(
        self,
        image: Tensor,
        object_mask: Tensor,
        u_map: Tensor,
        v_map: Tensor,
        part_names: Sequence[str],
    ) -> Tensor:
        batch_size, _, height, width = image.shape

        if len(part_names) != batch_size:
            raise ValueError(
                f"Received {len(part_names)} part names "
                f"for a batch of {batch_size} images."
            )

        clip_features = self.text_encoder(
            part_names
        )

        projected_text = self.text_projection(
            clip_features
        )

        text_feature_map = (
            projected_text
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                batch_size,
                self.text_channels,
                height,
                width,
            )
        )

        model_input = torch.cat(
            [
                image,
                object_mask,
                u_map,
                v_map,
                text_feature_map,
            ],
            dim=1,
        )

        return self.decoder(model_input)