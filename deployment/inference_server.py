"""HTTP API and local web server for inference on user-uploaded images."""

from __future__ import annotations

import base64
import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from inference_options import (
    OPTIONS as INFERENCE_OPTIONS,
    PARENT_CATEGORIES,
    PART_QUERIES,
    suggested_parts_for_parent,
)
from datasets.metadata import PARTS_BY_OBJECT
from final_model.inference import UIPredictor, load_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
REGISTRY_PATH = PROJECT_ROOT / "models" / "final_study" / "model_registry.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
PREDICTION_THRESHOLD = 0.5
PARENT_SCORE_THRESHOLD = 0.55
MAX_PARENT_DETECTIONS = 12


COCO_TO_PASCAL = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "person",
    "potted plant": "pottedplant",
    "sheep": "sheep",
    "couch": "sofa",
    "train": "train",
    "tv": "tvmonitor",
}


def _load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


MODEL_REGISTRY = _load_registry()
MODEL_SPECS = MODEL_REGISTRY["models"]
PARTS_BY_NAME = {
    object_name: set(part_names)
    for object_name, part_names in PARTS_BY_OBJECT.values()
}
SUPPORTED_PART_QUERIES = set(PART_QUERIES)


app = FastAPI(
    title="Object-Relative Part Segmentation API",
    version="1.0.0",
)


_inference_lock = threading.Lock()


def _normalise_model_id(model_id: str) -> str:
    return model_id.removeprefix("final_")


def _checkpoint_path(model_id: str) -> Path:
    normalised = _normalise_model_id(model_id)
    spec = MODEL_SPECS.get(normalised)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_id}'.",
        )

    path = PROJECT_ROOT / spec["checkpoint"]
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Checkpoint is unavailable for model '{normalised}'.",
        )
    return path


@lru_cache(maxsize=1)
def _predictor(model_id: str) -> UIPredictor:
    """Keep the most recently selected model in memory."""
    return load_predictor(_checkpoint_path(model_id))


@lru_cache(maxsize=1)
def _parent_predictor() -> tuple[torch.nn.Module, object, torch.device]:
    """Load a COCO instance segmenter for Pascal parent categories."""
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, weights, device


async def _read_upload(upload: UploadFile, label: str) -> bytes:
    content_type = upload.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{label} must be an image file.")

    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail=f"{label} is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} exceeds the 15 MB limit.")
    return content


def _open_image(content: bytes, label: str, mode: str) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert(mode)
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail=f"{label} is not a valid image.") from error

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"{label} is too large; use at most 25 megapixels.",
        )
    return image


def _validate_request(category: str, part: str) -> tuple[str, str, str]:
    clean_category = category.strip().lower()
    clean_part = part.strip().lower()

    if not clean_category:
        raise HTTPException(
            status_code=422,
            detail="Object category cannot be empty.",
        )
    if not clean_part:
        raise HTTPException(
            status_code=422,
            detail="Part query cannot be empty.",
        )
    if len(clean_category) > 80 or len(clean_part) > 80:
        raise HTTPException(
            status_code=422,
            detail="Object category and part query must be at most 80 characters.",
        )
    if clean_category not in PARENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Category '{category}' is not supported by the COCO parent detector.",
        )
    if clean_part not in SUPPORTED_PART_QUERIES:
        raise HTTPException(
            status_code=422,
            detail=f"Part '{part}' is outside the trained part-query vocabulary.",
        )

    suggested_parts = suggested_parts_for_parent(clean_category, PARTS_BY_NAME)
    if clean_part not in suggested_parts:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Part '{part}' is not a relevant trained query for "
                f"category '{category}'."
            ),
        )

    evaluated = (
        clean_category in PARTS_BY_NAME
        and clean_part in PARTS_BY_NAME[clean_category]
    )
    evaluation_mode = (
        "pascal_part_116"
        if evaluated
        else "open_vocabulary_experimental"
    )
    return clean_category, clean_part, evaluation_mode


def _to_tensors(
    rgb_image: Image.Image,
    mask_image: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb_image.size != mask_image.size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Image size {rgb_image.width}x{rgb_image.height} and mask size "
                f"{mask_image.width}x{mask_image.height} must match."
            ),
        )

    rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
    mask_array = np.asarray(mask_image, dtype=np.uint8)
    binary_mask = mask_array >= 128

    selected_pixels = int(binary_mask.sum())
    if selected_pixels == 0:
        raise HTTPException(
            status_code=422,
            detail="Parent mask is empty; the object must be white on black.",
        )
    if selected_pixels == binary_mask.size:
        raise HTTPException(
            status_code=422,
            detail="Parent mask covers the complete image; check that its background is black.",
        )

    image_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).contiguous()
    mask_tensor = torch.from_numpy(binary_mask.copy())
    return image_tensor, mask_tensor


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_outputs(
    rgb_image: Image.Image,
    probability: torch.Tensor,
) -> tuple[str, str]:
    probability_array = probability.numpy()
    prediction = probability_array >= PREDICTION_THRESHOLD

    rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
    overlay = rgb_array.astype(np.float32)
    red = np.array([239, 68, 68], dtype=np.float32)
    overlay[prediction] = 0.42 * overlay[prediction] + 0.58 * red
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB")
    mask_image = Image.fromarray((prediction * 255).astype(np.uint8), "L")
    return _png_data_url(overlay_image), _png_data_url(mask_image)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "models": sorted(MODEL_SPECS),
        "automatic_parent_prediction": True,
        "parent_categories": list(PARENT_CATEGORIES),
        "part_queries": list(PART_QUERIES),
        "part_suggestion_groups": INFERENCE_OPTIONS["part_suggestion_groups"],
        "parent_part_groups": INFERENCE_OPTIONS["parent_part_groups"],
    }


@app.post("/api/parent/predict")
async def predict_parent(image: UploadFile = File(...)) -> dict:
    image_bytes = await _read_upload(image, "RGB image")
    rgb_image = _open_image(image_bytes, "RGB image", "RGB")
    rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
    image_tensor = (
        torch.from_numpy(rgb_array)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div(255.0)
    )

    try:
        with _inference_lock, torch.inference_mode():
            model, weights, device = _parent_predictor()
            output = model([image_tensor.to(device)])[0]
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Parent-mask prediction failed: {error}",
        ) from error

    categories = weights.meta["categories"]
    detections = []

    for index, score_tensor in enumerate(output["scores"]):
        score = float(score_tensor.detach().cpu())
        if score < PARENT_SCORE_THRESHOLD:
            break

        coco_label = categories[int(output["labels"][index].detach().cpu())]
        category = COCO_TO_PASCAL.get(coco_label, coco_label.strip().lower())
        if not category or category in {"__background__", "n/a"}:
            continue

        mask = output["masks"][index, 0].detach().float().cpu() >= 0.5
        mask_image = Image.fromarray((mask.numpy() * 255).astype(np.uint8), "L")
        box = output["boxes"][index].detach().float().cpu().tolist()
        detections.append(
            {
                "id": len(detections),
                "category": category,
                "score": round(score, 4),
                "box": [round(value, 2) for value in box],
                "pascal_part_category": category in PARTS_BY_NAME,
                "mask_data_url": _png_data_url(mask_image),
            }
        )

        if len(detections) >= MAX_PARENT_DETECTIONS:
            break

    return {
        "width": rgb_image.width,
        "height": rgb_image.height,
        "detections": detections,
        "score_threshold": PARENT_SCORE_THRESHOLD,
    }


@app.post("/api/predict")
async def predict_part(
    image: UploadFile = File(...),
    parent_mask: UploadFile = File(...),
    category: str = Form(...),
    part: str = Form(...),
    model: str = Form("rotation_consistent"),
) -> dict:
    clean_category, clean_part, evaluation_mode = _validate_request(category, part)
    model_id = _normalise_model_id(model)
    _checkpoint_path(model_id)

    image_bytes = await _read_upload(image, "RGB image")
    mask_bytes = await _read_upload(parent_mask, "Parent mask")
    rgb_image = _open_image(image_bytes, "RGB image", "RGB")
    mask_image = _open_image(mask_bytes, "Parent mask", "L")
    image_tensor, mask_tensor = _to_tensors(rgb_image, mask_image)

    try:
        with _inference_lock:
            probability = _predictor(model_id).predict(
                image_tensor,
                mask_tensor,
                clean_part,
            )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {error}",
        ) from error

    overlay_data_url, mask_data_url = _render_outputs(rgb_image, probability)
    return {
        "category": clean_category,
        "part": clean_part,
        "model": model_id,
        "threshold": PREDICTION_THRESHOLD,
        "evaluation_mode": evaluation_mode,
        "width": rgb_image.width,
        "height": rgb_image.height,
        "overlay_data_url": overlay_data_url,
        "mask_data_url": mask_data_url,
    }


# Mount this last so /api routes take precedence over the static website.
app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
