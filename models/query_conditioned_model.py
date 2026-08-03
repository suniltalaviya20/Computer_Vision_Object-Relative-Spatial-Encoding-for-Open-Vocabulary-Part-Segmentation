from __future__ import annotations

import torch
from torch import Tensor, nn

from models.decoder import TinyPartDecoder


class QueryConditionedPartModel(nn.Module):
    """Temporary query-conditioned segmentation model.

    This debugging model uses a learned embedding for each generic
    part name. It will later be replaced with a CLIP text embedding.
    """

    def __init__(
        self,
        number_of_parts: int,
        text_channels: int = 16,
    ) -> None:
        super().__init__()

        if number_of_parts <= 0:
            raise ValueError(
                "number_of_parts must be positive."
            )

        self.text_channels = text_channels

        self.part_embedding = nn.Embedding(
            num_embeddings=number_of_parts,
            embedding_dim=text_channels,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_channels, text_channels),
            nn.ReLU(inplace=True),
        )

        # Image: 3
        # Object mask: 1
        # Relative U: 1
        # Relative V: 1
        # Text feature map: text_channels
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
        part_indices: Tensor,
    ) -> Tensor:
        """Predict the requested part mask.

        Args:
            image:
                Tensor with shape [B, 3, H, W].

            object_mask:
                Tensor with shape [B, 1, H, W].

            u_map:
                Tensor with shape [B, 1, H, W].

            v_map:
                Tensor with shape [B, 1, H, W].

            part_indices:
                Integer tensor with shape [B].

        Returns:
            Part-mask logits with shape [B, 1, H, W].
        """
        if part_indices.ndim != 1:
            raise ValueError(
                "part_indices must have shape [B]."
            )

        batch_size, _, height, width = image.shape

        text_embedding = self.part_embedding(
            part_indices
        )

        text_embedding = self.text_projection(
            text_embedding
        )

        # Convert [B, C] into [B, C, H, W].
        text_feature_map = (
            text_embedding
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