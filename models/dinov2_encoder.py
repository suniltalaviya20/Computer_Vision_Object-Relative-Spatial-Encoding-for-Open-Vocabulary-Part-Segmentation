from __future__ import annotations

import torch
from torch import Tensor, nn


class FrozenDINOv2Encoder(nn.Module):
    """Frozen DINOv2 encoder that returns dense patch features."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()

        print(f"Loading DINOv2 model: {model_name}")

        self.backbone = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=model_name,
            source="github",
            trust_repo=True,
        )

        # The visual encoder remains frozen throughout training.
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.output_dim = int(self.backbone.embed_dim)

        patch_size = self.backbone.patch_size

        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise ValueError(
                    "Only square DINOv2 patches are supported."
                )

            patch_size = patch_size[0]

        self.patch_size = int(patch_size)

        # ImageNet normalization expected by DINOv2.
        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(
        self,
        mode: bool = True,
    ) -> "FrozenDINOv2Encoder":
        """Keep the DINOv2 backbone in evaluation mode."""

        super().train(mode)
        self.backbone.eval()

        return self

    @torch.no_grad()
    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """Extract spatial DINOv2 patch features.

        Args:
            images:
                Tensor with shape [B, 3, H, W], values in [0, 1].

        Returns:
            Feature map with shape:
                [B, output_dim, H / patch_size, W / patch_size]
        """

        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, 3, H, W], "
                f"received {tuple(images.shape)}."
            )

        if images.shape[1] != 3:
            raise ValueError(
                "DINOv2 expects three-channel RGB images."
            )

        _, _, height, width = images.shape

        if (
            height % self.patch_size != 0
            or width % self.patch_size != 0
        ):
            raise ValueError(
                f"Image dimensions must be divisible by patch size "
                f"{self.patch_size}. Received {height} x {width}."
            )

        images = images.float()

        normalized_images = (
            images - self.mean
        ) / self.std

        outputs = self.backbone.forward_features(
            normalized_images
        )

        if "x_norm_patchtokens" not in outputs:
            raise KeyError(
                "DINOv2 output does not contain "
                "'x_norm_patchtokens'. Available keys: "
                f"{list(outputs.keys())}"
            )

        patch_tokens = outputs[
            "x_norm_patchtokens"
        ]

        if patch_tokens.ndim != 3:
            raise RuntimeError(
                "Expected DINOv2 patch tokens with shape "
                f"[B, N, C], received {tuple(patch_tokens.shape)}."
            )

        batch_size, number_of_patches, channels = (
            patch_tokens.shape
        )

        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        expected_patches = (
            patch_height * patch_width
        )

        if number_of_patches != expected_patches:
            raise RuntimeError(
                "Unexpected number of patch tokens. "
                f"Expected {expected_patches}, "
                f"received {number_of_patches}."
            )

        feature_map = (
            patch_tokens
            .transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                channels,
                patch_height,
                patch_width,
            )
        )

        return feature_map