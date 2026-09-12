from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_ROOT = PROJECT_ROOT / "models" / "final_study"


DEMO_MODELS = {
    "final_baseline_object_mask": {
        "id": "final_baseline_object_mask",
        "label": "Object-Mask Baseline",
        "family": "final_training",
        "mode": "baseline_object_mask",
        "checkpoint": FINAL_MODEL_ROOT / "baseline_object_mask.pt",
        "description": "Text-conditioned baseline using the parent mask without explicit geometry.",
        "geometry": [],
        "gated": False,
        "comparison_group": "final_training_4217799",
        "supports_live": True,
    },
    "final_fixed_uvd": {
        "id": "final_fixed_uvd",
        "label": "Fixed UVD",
        "family": "final_training",
        "mode": "fixed_uvd",
        "checkpoint": FINAL_MODEL_ROOT / "fixed_uvd.pt",
        "description": "Uses fixed object-relative U, V, and boundary-distance D geometry.",
        "geometry": ["u", "v", "d"],
        "gated": False,
        "comparison_group": "final_training_4217799",
        "supports_live": True,
    },
    "final_query_gated_uvd": {
        "id": "final_query_gated_uvd",
        "label": "Query-Gated UVD",
        "family": "final_training",
        "mode": "query_gated_uvd",
        "checkpoint": FINAL_MODEL_ROOT / "query_gated_uvd.pt",
        "description": "Uses text-query-conditioned weights for object-relative U, V, and D.",
        "geometry": ["u", "v", "d"],
        "gated": True,
        "comparison_group": "final_training_4217799",
        "supports_live": True,
    },
    "final_rotation_consistent": {
        "id": "final_rotation_consistent",
        "label": "Rotation-Consistent UVD",
        "family": "final_training",
        "mode": "rotation_consistent",
        "checkpoint": FINAL_MODEL_ROOT / "rotation_consistent.pt",
        "description": "Query-gated UVD trained with a 90-degree rotation-consistency objective.",
        "geometry": ["u", "v", "d"],
        "gated": True,
        "comparison_group": "final_training_4217799",
        "supports_live": True,
    },
    "final_geometry_dropout": {
        "id": "final_geometry_dropout",
        "label": "Geometry-Dropout UVD",
        "family": "final_training",
        "mode": "geometry_dropout",
        "checkpoint": FINAL_MODEL_ROOT / "geometry_dropout.pt",
        "description": "Query-gated UVD trained with joint geometry-branch dropout.",
        "geometry": ["u", "v", "d"],
        "gated": True,
        "comparison_group": "final_training_4217799",
        "supports_live": True,
    },
}


def get_demo_model(model_id):
    if model_id not in DEMO_MODELS:
        raise KeyError(f"Unknown demo model: {model_id}")
    return DEMO_MODELS[model_id]


def get_demo_models():
    return list(DEMO_MODELS.values())


def get_model_ids():
    return list(DEMO_MODELS)


def get_frontend_models():
    return [
        {
            "id": model["id"],
            "label": model["label"],
            "description": model["description"],
            "geometry": list(model["geometry"]),
            "gated": bool(model["gated"]),
            "comparison_group": model["comparison_group"],
        }
        for model in get_demo_models()
    ]


def validate_demo_models():
    errors = []
    labels = set()

    for model_id, model in DEMO_MODELS.items():
        if model["id"] != model_id:
            errors.append(f"{model_id}: id mismatch")
        if model["label"] in labels:
            errors.append(f"{model_id}: duplicate label: {model['label']}")
        labels.add(model["label"])

        checkpoint = Path(model["checkpoint"])
        if not checkpoint.is_file():
            errors.append(f"{model_id}: checkpoint missing: {checkpoint}")

        geometry = model.get("geometry", [])
        if not isinstance(geometry, list):
            errors.append(f"{model_id}: geometry must be a list")
            continue
        unknown = set(geometry) - {"x", "y", "u", "v", "d"}
        if unknown:
            errors.append(f"{model_id}: unsupported geometry: {sorted(unknown)}")
        if model.get("gated", False) and not {"u", "v", "d"}.issubset(geometry):
            errors.append(f"{model_id}: gated UVD model must declare u, v and d")

    if errors:
        raise RuntimeError("Demo model registry is invalid:\n" + "\n".join(f"- {e}" for e in errors))
    return True
