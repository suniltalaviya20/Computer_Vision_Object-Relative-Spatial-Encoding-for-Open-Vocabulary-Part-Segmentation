from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.clip_text_encoder import FrozenCLIPTextEncoder
from models.dinov2_encoder import FrozenDINOv2Encoder


ABLATION_SETTINGS: dict[str, tuple[bool, bool]] = {
    # mode: (use_object_mask, use_relative_coordinates)
    "part_only": (False, False),
    "object_mask": (True, False),
    "relative_uv": (True, True),
    "full": (True, True),
}


class ConvNormActivation(nn.Module):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class DINOCLIPPartModel(nn.Module):
    """Object-relative, text-conditioned part segmentation model.

    Frozen:
        - DINOv2 visual encoder
        - CLIP text encoder

    Trainable:
        - visual projection
        - text projection
        - segmentation decoder
    """

    def __init__(
        self,
        dino_model_name: str = "dinov2_vits14",
        clip_model_name: str = "ViT-B/32",
        visual_channels: int = 128,
        text_channels: int = 32,
        ablation_mode: str = "full",
    ) -> None:
        super().__init__()

        if visual_channels % 8 != 0:
            raise ValueError(
                "visual_channels must be divisible by 8 "
                "because GroupNorm uses 8 groups."
            )

        self.dino_model_name = dino_model_name
        self.clip_model_name = clip_model_name
        self.visual_channels = visual_channels
        self.text_channels = text_channels

        self.visual_encoder = FrozenDINOv2Encoder(
            model_name=dino_model_name,
        )

        self.text_encoder = FrozenCLIPTextEncoder(
            model_name=clip_model_name,
        )

        self.visual_projection = nn.Sequential(
            nn.Conv2d(
                self.visual_encoder.output_dim,
                visual_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=visual_channels,
            ),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(
                self.text_encoder.output_dim,
                128,
            ),
            nn.GELU(),
            nn.Linear(
                128,
                text_channels,
            ),
        )

        # We always reserve three geometry channels:
        #
        # 1. object mask
        # 2. relative U
        # 3. relative V
        #
        # In an ablation where a channel is disabled, it is replaced
        # with zeros rather than removed. This keeps the decoder
        # architecture and parameter count identical.
        decoder_input_channels = (
            visual_channels
            + text_channels
            + 3
        )

        self.decoder = nn.Sequential(
            ConvNormActivation(
                decoder_input_channels,
                128,
            ),
            ConvNormActivation(
                128,
                128,
            ),
            ConvNormActivation(
                128,
                64,
            ),
            ConvNormActivation(
                64,
                32,
            ),
            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),
        )

        self.set_ablation(ablation_mode)

    def set_ablation(
        self,
        ablation_mode: str,
    ) -> None:
        """Change the active ablation without rebuilding the model."""

        if ablation_mode not in ABLATION_SETTINGS:
            available_modes = ", ".join(
                sorted(ABLATION_SETTINGS)
            )

            raise ValueError(
                f"Unknown ablation mode: {ablation_mode}. "
                f"Available modes: {available_modes}"
            )

        use_object_mask, use_relative_coordinates = (
            ABLATION_SETTINGS[ablation_mode]
        )

        if (
            use_relative_coordinates
            and not use_object_mask
        ):
            raise ValueError(
                "Relative coordinates require the parent-object mask."
            )

        self.ablation_mode = ablation_mode
        self.use_object_mask = use_object_mask
        self.use_relative_coordinates = (
            use_relative_coordinates
        )

    def get_config(self) -> dict[str, object]:
        """Return the model configuration for checkpointing."""

        return {
            "dino_model_name": self.dino_model_name,
            "clip_model_name": self.clip_model_name,
            "visual_channels": self.visual_channels,
            "text_channels": self.text_channels,
            "ablation_mode": self.ablation_mode,
            "use_object_mask": self.use_object_mask,
            "use_relative_coordinates": (
                self.use_relative_coordinates
            ),
        }

    @staticmethod
    def _validate_spatial_tensor(
        tensor: Tensor,
        name: str,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        expected_shape = (
            batch_size,
            1,
            height,
            width,
        )

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"received {tuple(tensor.shape)}."
            )

    def forward(
        self,
        image: Tensor,
        object_mask: Tensor,
        u_map: Tensor,
        v_map: Tensor,
        part_names: Sequence[str],
    ) -> Tensor:
        """Predict requested-part segmentation logits."""

        if image.ndim != 4:
            raise ValueError(
                "image must have shape [B, 3, H, W]."
            )

        batch_size, channels, height, width = (
            image.shape
        )

        if channels != 3:
            raise ValueError(
                "image must contain three RGB channels."
            )

        self._validate_spatial_tensor(
            object_mask,
            "object_mask",
            batch_size,
            height,
            width,
        )

        self._validate_spatial_tensor(
            u_map,
            "u_map",
            batch_size,
            height,
            width,
        )

        self._validate_spatial_tensor(
            v_map,
            "v_map",
            batch_size,
            height,
            width,
        )

        if len(part_names) != batch_size:
            raise ValueError(
                f"Received {len(part_names)} part names "
                f"for batch size {batch_size}."
            )

        # ---------------------------------------------------------
        # 1. Frozen DINOv2 visual features
        # ---------------------------------------------------------

        visual_features = self.visual_encoder(
            image
        )

        patch_height = visual_features.shape[-2]
        patch_width = visual_features.shape[-1]

        patch_resolution = (
            patch_height,
            patch_width,
        )

        # ---------------------------------------------------------
        # 2. Resize geometry to the DINOv2 patch grid
        # ---------------------------------------------------------

        object_mask_low = F.interpolate(
            object_mask.float(),
            size=patch_resolution,
            mode="nearest",
        )

        object_mask_low = object_mask_low.clamp(
            min=0.0,
            max=1.0,
        )

        u_map_low = F.interpolate(
            u_map.float(),
            size=patch_resolution,
            mode="bilinear",
            align_corners=False,
        )

        v_map_low = F.interpolate(
            v_map.float(),
            size=patch_resolution,
            mode="bilinear",
            align_corners=False,
        )

        # Prevent interpolation from producing coordinate values
        # outside the parent object.
        u_map_low = u_map_low * object_mask_low
        v_map_low = v_map_low * object_mask_low

        # ---------------------------------------------------------
        # 3. Project DINOv2 features
        # ---------------------------------------------------------

        projected_visual_features = (
            self.visual_projection(
                visual_features
            )
        )

        if self.use_object_mask:
            decoder_visual_features = (
                projected_visual_features
                * object_mask_low
            )

            decoder_object_mask = object_mask_low
        else:
            # The part-only baseline must not receive the mask,
            # including through visual-feature masking.
            decoder_visual_features = (
                projected_visual_features
            )

            decoder_object_mask = torch.zeros_like(
                object_mask_low
            )

        # ---------------------------------------------------------
        # 4. Enable or disable relative coordinates
        # ---------------------------------------------------------

        if self.use_relative_coordinates:
            decoder_u_map = u_map_low
            decoder_v_map = v_map_low
        else:
            decoder_u_map = torch.zeros_like(
                u_map_low
            )

            decoder_v_map = torch.zeros_like(
                v_map_low
            )

        # ---------------------------------------------------------
        # 5. Frozen CLIP text features
        # ---------------------------------------------------------

        text_features = self.text_encoder(
            part_names
        )

        projected_text_features = (
            self.text_projection(
                text_features
            )
        )

        text_feature_map = (
            projected_text_features
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                batch_size,
                self.text_channels,
                patch_height,
                patch_width,
            )
        )

        # ---------------------------------------------------------
        # 6. Multimodal feature fusion
        # ---------------------------------------------------------

        fused_features = torch.cat(
            [
                decoder_visual_features,
                text_feature_map,
                decoder_object_mask,
                decoder_u_map,
                decoder_v_map,
            ],
            dim=1,
        )

        low_resolution_logits = self.decoder(
            fused_features
        )

        # ---------------------------------------------------------
        # 7. Return full-resolution logits
        # ---------------------------------------------------------

        full_resolution_logits = F.interpolate(
            low_resolution_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return full_resolution_logits