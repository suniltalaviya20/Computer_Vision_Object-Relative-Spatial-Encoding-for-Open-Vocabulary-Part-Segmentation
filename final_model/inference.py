from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch.torch_version import TorchVersion

from .training_core import (
    PROJECT_ROOT,
    PartSegmenter,
    TextCache,
    TrainingConfig,
    preprocess_image,
    preprocess_mask,
    relative_uvd,
    resize_info,
)


class UIPredictor:
    """Load a trained UI checkpoint and predict a part from an unseen parent mask."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or "model_state" not in checkpoint:
            raise RuntimeError("Unsupported UI checkpoint format")
        names = {field.name for field in fields(TrainingConfig)}
        config_values = {key: value for key, value in checkpoint["config"].items() if key in names}
        self.config = TrainingConfig(**config_values)

        dino_repository = PROJECT_ROOT / "dinov2"
        dino_weights = PROJECT_ROOT / checkpoint["config"]["dino_checkpoint"]
        if dino_repository.is_dir() and dino_weights.is_file():
            stale = [name for name in sys.modules if name == "dinov2" or name.startswith("dinov2.")]
            for name in sorted(stale, reverse=True):
                del sys.modules[name]
            dino = torch.hub.load(
                str(dino_repository), "dinov2_vits14", source="local", pretrained=False
            )
            state = torch.load(dino_weights, map_location="cpu", weights_only=True)
            state = state.get("model", state) if isinstance(state, dict) else state
            dino.load_state_dict(
                {key.removeprefix("module."): value for key, value in state.items()}, strict=True
            )
        else:
            dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dino = dino.to(self.device).eval()
        for parameter in dino.parameters():
            parameter.requires_grad = False

        self.model = PartSegmenter(dino, self.config).to(self.device)
        result = self.model.load_state_dict(checkpoint["model_state"], strict=False)
        invalid_missing = [key for key in result.missing_keys if not key.startswith("dino.")]
        if invalid_missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {invalid_missing}, {result.unexpected_keys}")
        self.model.eval()

        clip_path = PROJECT_ROOT / checkpoint["config"]["clip_checkpoint"]
        if clip_path.is_file():
            clip, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32-quickgelu", pretrained=None
            )
            open_clip.load_checkpoint(clip, str(clip_path), strict=True, weights_only=True)
        else:
            clip, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32-quickgelu", pretrained="openai"
            )
        clip = clip.to(self.device).eval()
        for parameter in clip.parameters():
            parameter.requires_grad = False
        self.text = TextCache(
            clip,
            open_clip.get_tokenizer("ViT-B-32"),
            self.device,
            self.device.type == "cuda",
        )

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, parent_mask: torch.Tensor, query: str) -> torch.Tensor:
        """Return an H×W CPU probability map for uint8 CHW RGB and an H×W mask."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape [3, H, W]")
        if parent_mask.ndim not in (2, 3):
            raise ValueError("parent_mask must have shape [H, W] or [1, H, W]")
        height, width = image.shape[-2:]
        if parent_mask.shape[-2:] != (height, width):
            raise ValueError("image and parent_mask spatial sizes must match")
        _, model_image = preprocess_image(image.cpu(), self.config.image_size)
        model_mask = preprocess_mask(parent_mask.cpu(), height, width, self.config.image_size).float()
        u, v, d = relative_uvd(model_mask.bool())
        values = [tensor.unsqueeze(0).to(self.device) for tensor in (model_image, model_mask, u, v, d)]
        context = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
        with context:
            logits, _ = self.model(values[0], self.text([query]), values[1], values[2], values[3], values[4])
        probability = torch.sigmoid(logits).float()
        info = resize_info(height, width, self.config.image_size)
        top, left = int(info["top"]), int(info["left"])
        new_h, new_w = int(info["new_h"]), int(info["new_w"])
        cropped = probability[:, :, top : top + new_h, left : left + new_w]
        restored = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
        return restored[0, 0].cpu()


def load_predictor(checkpoint_path: str | Path, device: str | None = None) -> UIPredictor:
    return UIPredictor(checkpoint_path, device=device)
