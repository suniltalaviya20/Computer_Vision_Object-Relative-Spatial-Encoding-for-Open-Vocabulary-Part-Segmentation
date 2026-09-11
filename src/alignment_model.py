import torch
import torch.nn as nn
import torch.nn.functional as F

from src.uvd_model import (
    QueryGeometryGate,
)


class PartQueryAlignmentSegmenter(nn.Module):

    def __init__(
        self,
        dino_encoder,
        mode,
        visual_dim=128,
        text_common_dim=128,
        text_decoder_dim=32,
        temperature=0.10,
        gate_hidden_dim=64,
    ):
        super().__init__()

        valid_modes = {
            "mask_baseline",
            "alignment_mask",
            "alignment_relative_uv",
            "alignment_fixed_uvd",
            "alignment_query_gated_uvd",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode
        self.temperature = temperature

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
                text_common_dim,
            ),
            nn.GELU(),
        )

        self.text_decoder_projection = (
            nn.Linear(
                text_common_dim,
                text_decoder_dim,
            )
        )

        if (
            mode
            == "alignment_query_gated_uvd"
        ):
            self.geometry_gate = (
                QueryGeometryGate(
                    input_dim=512,
                    hidden_dim=gate_hidden_dim,
                )
            )
        else:
            self.geometry_gate = None

        self.uses_uvd = mode in {
            "alignment_fixed_uvd",
            "alignment_query_gated_uvd",
        }

        if self.uses_uvd:
            # visual     128
            # text        32
            # object       1
            # U,V,D        3
            # alignment    1
            # ----------------
            # total       165
            geometry_channels = 3
        else:
            # Keep legacy architecture
            # exactly unchanged.
            #
            # visual     128
            # text        32
            # object       1
            # U,V          2
            # alignment    1
            # ----------------
            # total       164
            geometry_channels = 2

        fusion_dim = (
            visual_dim
            + text_decoder_dim
            + 1
            + geometry_channels
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


    @torch.no_grad()
    def extract_dino_features(
        self,
        images,
    ):
        output = self.dino.forward_features(
            images
        )

        tokens = output[
            "x_norm_patchtokens"
        ]

        batch_size = tokens.shape[0]
        num_patches = tokens.shape[1]
        channels = tokens.shape[2]

        grid_size = int(
            num_patches ** 0.5
        )

        features = (
            tokens
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


    def compute_alignment(
        self,
        visual,
        text_common,
    ):
        visual_normalized = F.normalize(
            visual,
            dim=1,
        )

        text_normalized = F.normalize(
            text_common,
            dim=1,
        )

        text_spatial = (
            text_normalized[
                :,
                :,
                None,
                None,
            ]
        )

        cosine_similarity = (
            visual_normalized
            * text_spatial
        ).sum(
            dim=1,
            keepdim=True,
        )

        alignment_logits = (
            cosine_similarity
            / self.temperature
        )

        alignment_probability = (
            torch.sigmoid(
                alignment_logits
            )
        )

        return (
            alignment_logits,
            alignment_probability,
        )


    def forward(
        self,
        images,
        text_embeddings,
        object_mask,
        relative_u,
        relative_v,
        boundary_d=None,
        return_gate_weights=False,
    ):
        dino_features = (
            self.extract_dino_features(
                images
            )
        )

        visual = self.visual_projection(
            dino_features
        )

        text_common = self.text_projection(
            text_embeddings
        )

        text_decoder = (
            self.text_decoder_projection(
                text_common
            )
        )

        text_map = (
            text_decoder[
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

        (
            alignment_logits,
            alignment_probability,
        ) = self.compute_alignment(
            visual,
            text_common,
        )

        spatial_size = (
            visual.shape[-2:]
        )

        mask_low = F.interpolate(
            object_mask,
            size=spatial_size,
            mode="nearest",
        )

        alignment_map = (
            alignment_probability
            * mask_low
        )

        u_low = F.interpolate(
            relative_u,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        v_low = F.interpolate(
            relative_v,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        u_low = (
            u_low
            * mask_low
        )

        v_low = (
            v_low
            * mask_low
        )


        # =====================================
        # Existing modes
        # =====================================

        if self.mode == "mask_baseline":

            alignment_input = (
                torch.zeros_like(
                    alignment_map
                )
            )

            u_input = (
                torch.zeros_like(
                    u_low
                )
            )

            v_input = (
                torch.zeros_like(
                    v_low
                )
            )

            fused = torch.cat(
                [
                    visual,
                    text_map,
                    mask_low,
                    u_input,
                    v_input,
                    alignment_input,
                ],
                dim=1,
            )

            gate_weights = None


        elif self.mode == "alignment_mask":

            alignment_input = (
                alignment_map
            )

            u_input = (
                torch.zeros_like(
                    u_low
                )
            )

            v_input = (
                torch.zeros_like(
                    v_low
                )
            )

            fused = torch.cat(
                [
                    visual,
                    text_map,
                    mask_low,
                    u_input,
                    v_input,
                    alignment_input,
                ],
                dim=1,
            )

            gate_weights = None


        elif self.mode == "alignment_relative_uv":

            fused = torch.cat(
                [
                    visual,
                    text_map,
                    mask_low,
                    u_low,
                    v_low,
                    alignment_map,
                ],
                dim=1,
            )

            gate_weights = None


        # =====================================
        # Notebook 10 UVD modes
        # =====================================

        else:

            if boundary_d is None:
                raise ValueError(
                    "boundary_d is required "
                    f"for mode {self.mode}"
                )

            d_low = F.interpolate(
                boundary_d,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )

            d_low = (
                d_low
                * mask_low
            )


            if (
                self.mode
                == "alignment_fixed_uvd"
            ):
                gate_weights = torch.ones(
                    (
                        images.shape[0],
                        3,
                    ),
                    device=images.device,
                    dtype=visual.dtype,
                )

                weighted_u = u_low
                weighted_v = v_low
                weighted_d = d_low


            else:

                gate_weights = (
                    self.geometry_gate(
                        text_embeddings.float()
                    )
                )

                alpha_u = (
                    gate_weights[
                        :,
                        0,
                    ]
                    .view(
                        -1,
                        1,
                        1,
                        1,
                    )
                )

                alpha_v = (
                    gate_weights[
                        :,
                        1,
                    ]
                    .view(
                        -1,
                        1,
                        1,
                        1,
                    )
                )

                alpha_d = (
                    gate_weights[
                        :,
                        2,
                    ]
                    .view(
                        -1,
                        1,
                        1,
                        1,
                    )
                )

                weighted_u = (
                    alpha_u
                    * u_low
                )

                weighted_v = (
                    alpha_v
                    * v_low
                )

                weighted_d = (
                    alpha_d
                    * d_low
                )


            fused = torch.cat(
                [
                    visual,
                    text_map,
                    mask_low,
                    weighted_u,
                    weighted_v,
                    weighted_d,
                    alignment_map,
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

        aux = {
            "alignment_logits":
                alignment_logits,

            "alignment_probability":
                alignment_probability,

            "alignment_map":
                alignment_map,

            "object_mask_low":
                mask_low,

            "gate_weights":
                gate_weights,
        }

        if (
            return_gate_weights
            and gate_weights is not None
        ):
            return (
                logits,
                aux,
                gate_weights,
            )

        return (
            logits,
            aux,
        )
    