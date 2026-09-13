import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryPartSegmenter(nn.Module):

    def __init__(
        self,
        dino_encoder,
        mode="object_mask",
        visual_dim=128,
        text_dim=32,
    ):
        super().__init__()

        valid_modes = {
            "object_mask",
            "absolute_xy",
            "relative_uv",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

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
            + 2
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


        batch_size = (
            patch_tokens.shape[0]
        )

        num_patches = (
            patch_tokens.shape[1]
        )

        channels = (
            patch_tokens.shape[2]
        )


        grid_size = int(
            num_patches ** 0.5
        )


        features = (
            patch_tokens
            .transpose(
                1,
                2,
            )
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
        absolute_x,
        absolute_y,
        relative_u,
        relative_v,
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
            text[
                :,
                :,
                None,
                None,
            ]
            .expand(
                -1,
                -1,
                visual.shape[-2],
                visual.shape[-1],
            )
        )


        spatial_size = (
            visual.shape[-2:]
        )


        mask_low = F.interpolate(
            object_mask,
            size=spatial_size,
            mode="nearest",
        )


        absolute_x_low = F.interpolate(
            absolute_x,
            size=spatial_size,
            mode="nearest",
        )


        absolute_y_low = F.interpolate(
            absolute_y,
            size=spatial_size,
            mode="nearest",
        )


        relative_u_low = F.interpolate(
            relative_u,
            size=spatial_size,
            mode="nearest",
        )


        relative_v_low = F.interpolate(
            relative_v,
            size=spatial_size,
            mode="nearest",
        )


        if self.mode == "object_mask":

            geometry_1 = torch.zeros_like(
                mask_low
            )

            geometry_2 = torch.zeros_like(
                mask_low
            )


        elif self.mode == "absolute_xy":

            geometry_1 = absolute_x_low
            geometry_2 = absolute_y_low


        else:

            geometry_1 = relative_u_low
            geometry_2 = relative_v_low


        fused = torch.cat(
            [
                visual,
                text_map,
                mask_low,
                geometry_1,
                geometry_2,
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