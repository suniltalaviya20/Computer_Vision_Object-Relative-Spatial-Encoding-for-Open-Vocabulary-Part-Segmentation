import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselinePartSegmenter(nn.Module):

    def __init__(
        self,
        dino_encoder,
        mode="part_only",
        visual_dim=128,
        text_dim=32,
    ):
        super().__init__()

        assert mode in {
            "part_only",
            "object_mask",
        }

        self.mode = mode
        self.dino = dino_encoder

        for parameter in self.dino.parameters():
            parameter.requires_grad = False

        self.visual_projection = nn.Conv2d(
            384,
            visual_dim,
            kernel_size=1,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                512,
                64,
            ),
            nn.GELU(),
            nn.Linear(
                64,
                text_dim,
            ),
        )

        fusion_dim = (
            visual_dim
            + text_dim
            + 1
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                fusion_dim,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                128,
            ),
            nn.GELU(),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                64,
            ),
            nn.GELU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        self.dino.eval()

        return self

    def extract_dino_features(
        self,
        images,
    ):
        with torch.no_grad():
            output = self.dino.forward_features(
                images
            )

            patch_tokens = output[
                "x_norm_patchtokens"
            ]

        batch_size, num_patches, channels = (
            patch_tokens.shape
        )

        grid_size = int(
            num_patches ** 0.5
        )

        features = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                grid_size,
                grid_size,
            )
        )

        return features

    def forward(
        self,
        images,
        text_embeddings,
        object_mask,
    ):
        dino_features = (
            self.extract_dino_features(
                images
            )
        )

        visual = self.visual_projection(
            dino_features
        )

        text = self.text_projection(
            text_embeddings
        )

        text_map = (
            text[:, :, None, None]
            .expand(
                -1,
                -1,
                visual.shape[-2],
                visual.shape[-1],
            )
        )

        mask_low = F.interpolate(
            object_mask,
            size=visual.shape[-2:],
            mode="nearest",
        )

        if self.mode == "part_only":
            mask_low = torch.zeros_like(
                mask_low
            )

        fused = torch.cat(
            [
                visual,
                text_map,
                mask_low,
            ],
            dim=1,
        )

        logits_low = self.decoder(
            fused
        )

        logits = F.interpolate(
            logits_low,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits