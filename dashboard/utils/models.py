from dataclasses import fields
from pathlib import Path

import open_clip
import streamlit as st
import torch
import torch.nn.functional as F
from torch.torch_version import TorchVersion

from dashboard.utils.demo_registry import DEMO_MODELS
from datasets import ObjectCentricDataset
from final_training.training_core import PartSegmenter, TrainingConfig, relative_uvd
from src.features.dino_features import get_device, load_dino_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = get_device()

QUALITATIVE_MODELS = {
    model["label"]: model_id
    for model_id, model in DEMO_MODELS.items()
}


@st.cache_resource
def load_unseen_dataset():
    return ObjectCentricDataset(
        split="test_unseen", image_size=224, context_ratio=0.15
    )


@st.cache_resource
def load_seen_dataset():
    return ObjectCentricDataset(
        split="test_seen", image_size=224, context_ratio=0.15
    )


@st.cache_resource
def load_encoders():
    """Load the frozen encoders once and share them across all five heads."""
    dino = load_dino_model(device=DEVICE)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    clip_model = clip_model.to(DEVICE).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return dino, clip_model, tokenizer


def resolve_model_spec(model_ref):
    model_ref = str(model_ref)
    if model_ref in DEMO_MODELS:
        model = DEMO_MODELS[model_ref]
    else:
        matches = [m for m in DEMO_MODELS.values() if m["mode"] == model_ref]
        if len(matches) != 1:
            raise KeyError(f"Unknown demo model: {model_ref}")
        model = matches[0]
    return {
        **model,
        "checkpoint": Path(model["checkpoint"]),
        "geometry": list(model.get("geometry", [])),
        "gated": bool(model.get("gated", False)),
    }


def checkpoint_path(model_ref):
    return resolve_model_spec(model_ref)["checkpoint"]


def checkpoint_exists(model_ref):
    return checkpoint_path(model_ref).is_file()


def _load_ui_checkpoint(path):
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or "model_state" not in checkpoint:
        raise RuntimeError(f"Unsupported UI checkpoint format: {path}")
    return checkpoint


@st.cache_resource
def load_segmentation_model(model_ref):
    spec = resolve_model_spec(model_ref)
    path = spec["checkpoint"]
    if not path.is_file():
        print("Checkpoint not found:", path)
        return None

    checkpoint = _load_ui_checkpoint(path)
    if checkpoint.get("experiment") != spec["mode"]:
        raise RuntimeError(
            f"Checkpoint experiment mismatch for {path}: "
            f"expected {spec['mode']!r}, found {checkpoint.get('experiment')!r}"
        )

    config_fields = {field.name for field in fields(TrainingConfig)}
    config_values = {
        key: value
        for key, value in checkpoint["config"].items()
        if key in config_fields
    }
    config = TrainingConfig(**config_values)
    dino, _, _ = load_encoders()
    model = PartSegmenter(dino, config).to(DEVICE)
    loaded = model.load_state_dict(checkpoint["model_state"], strict=False)
    missing = [key for key in loaded.missing_keys if not key.startswith("dino.")]
    if missing or loaded.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch for {path}: "
            f"missing={missing}, unexpected={loaded.unexpected_keys}"
        )
    return model.eval()


@torch.inference_mode()
def encode_query(query):
    _, clip_model, tokenizer = load_encoders()
    prompt = f"a photo of the {query} of an object"
    tokens = tokenizer([prompt]).to(DEVICE)
    return F.normalize(clip_model.encode_text(tokens).float(), dim=-1)


def predict_sample(sample, model_ref):
    spec = resolve_model_spec(model_ref)
    model = load_segmentation_model(model_ref)
    if model is None:
        return None

    image = sample["full_image"].unsqueeze(0).to(DEVICE)
    object_mask_cpu = sample["full_object_mask"].bool().cpu()
    u_cpu, v_cpu, d_cpu = relative_uvd(object_mask_cpu)
    object_mask = object_mask_cpu.float().unsqueeze(0).to(DEVICE)
    u, v, d = [value.unsqueeze(0).to(DEVICE) for value in (u_cpu, v_cpu, d_cpu)]

    with torch.inference_mode():
        logits, model_aux = model(
            image,
            encode_query(sample["query"]),
            object_mask,
            u,
            v,
            d,
        )

    probability = torch.sigmoid(logits)[0]
    prediction = probability > 0.5
    target = sample["full_part_mask"].to(prediction.device) > 0.5
    parent = object_mask[0] > 0.5

    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float()
    prediction_sum = prediction.sum().float()
    target_sum = target.sum().float()
    iou = (intersection / union.clamp_min(1)).item()
    dice = (2 * intersection / (prediction_sum + target_sum).clamp_min(1)).item()
    leakage = ((prediction & ~parent).sum().float() / prediction_sum.clamp_min(1)).item()

    aux = dict(model_aux)
    aux["gate_weights"] = model_aux["effective_gates"]

    return {
        "model_id": spec["id"],
        "mode": spec["mode"],
        "checkpoint": str(spec["checkpoint"]),
        "crop_probability": probability,
        "model_probability": probability,
        "projected_probability": probability,
        "prediction": prediction.float(),
        "target": target.float(),
        "iou": iou,
        "dice": dice,
        "leakage": leakage,
        "u": u_cpu,
        "v": v_cpu,
        "d": d_cpu,
        "aux": aux,
    }
