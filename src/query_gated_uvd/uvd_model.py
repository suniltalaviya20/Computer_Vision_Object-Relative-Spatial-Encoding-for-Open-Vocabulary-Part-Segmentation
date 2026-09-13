import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryGeometryGate(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=64,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                3,
            ),
        )

    def forward(
        self,
        text_embeddings,
    ):
        gate_logits = self.network(
            text_embeddings
        )

        return F.softmax(
            gate_logits,
            dim=-1,
        )


class UVDPartSegmenter(nn.Module):

    def __init__(
        self,
        dino_encoder,
        mode,
        visual_dim=128,
        text_dim=32,
        gate_hidden_dim=64,
    ):
        super().__init__()

        valid_modes = {
            "fixed_uvd",
            "query_gated_uvd",
        }

        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode

        # Frozen DINOv2
        self.dino = dino_encoder

        for parameter in self.dino.parameters():
            parameter.requires_grad = False

        # DINO 384 -> 128
        self.visual_projection = nn.Conv2d(
            384,
            visual_dim,
            kernel_size=1,
        )

        # CLIP 512 -> 32
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

        if mode == "query_gated_uvd":
            self.geometry_gate = (
                QueryGeometryGate(
                    input_dim=512,
                    hidden_dim=gate_hidden_dim,
                )
            )
        else:
            self.geometry_gate = None

        # visual 128
        # text    32
        # mask     1
        # UVD      3
        # ----------
        # total   164
        fusion_dim = (
            visual_dim
            + text_dim
            + 1
            + 3
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

        # Frozen backbone always stays in eval mode.
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
        num_tokens = tokens.shape[1]
        channels = tokens.shape[2]

        grid_size = int(
            num_tokens ** 0.5
        )

        return (
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


    def forward(
        self,
        images,
        text_embeddings,
        object_mask,
        relative_u,
        relative_v,
        boundary_d,
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

        d_low = F.interpolate(
            boundary_d,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

        u_low = u_low * mask_low
        v_low = v_low * mask_low
        d_low = d_low * mask_low

        if self.mode == "fixed_uvd":

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
                gate_weights[:, 0]
                .view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            alpha_v = (
                gate_weights[:, 1]
                .view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            alpha_d = (
                gate_weights[:, 2]
                .view(
                    -1,
                    1,
                    1,
                    1,
                )
            )

            weighted_u = (
                alpha_u * u_low
            )

            weighted_v = (
                alpha_v * v_low
            )

            weighted_d = (
                alpha_d * d_low
            )

        fused = torch.cat(
            [
                visual,
                text_map,
                mask_low,
                weighted_u,
                weighted_v,
                weighted_d,
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

        if return_gate_weights:
            return (
                logits,
                gate_weights,
            )

        return logits
