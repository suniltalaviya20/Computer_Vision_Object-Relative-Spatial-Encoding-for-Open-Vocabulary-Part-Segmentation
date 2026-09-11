from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# DEMO MODEL REGISTRY
#
# This is the single source of truth for models used by:
#
# 1. saved-example exporter
# 2. future robustness exporter
# 3. future FastAPI live inference backend
# 4. frontend catalogue
#
# When models change later, update this file rather than
# hard-coding model names throughout the dashboard.
# ============================================================

DEMO_MODELS = {

    "crop_alignment": {

        "id":
            "crop_alignment",

        "label":
            "Crop + Alignment",

        # Current internal mode understood by
        # dashboard.utils.models
        "mode":
            "alignment_mask",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "object_zoom"
            / "alignment_mask"
            / "best.pt",

        "description":
            "Object-centric crop and text alignment "
            "without explicit U/V/D geometry.",

        "geometry":
            [],

        "gated":
            False,

        "comparison_group":
            "final_object_centric",

        "supports_live":
            True,
    },


    "crop_alignment_fixed_uvd": {

        "id":
            "crop_alignment_fixed_uvd",

        "label":
            "Crop + Alignment + Fixed UVD",

        "mode":
            "alignment_fixed_uvd",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "object_zoom"
            / "alignment_fixed_uvd"
            / "best.pt",

        "description":
            "Uses object-relative U, V and D "
            "without a learned geometry gate.",

        "geometry": [
            "u",
            "v",
            "d",
        ],

        "gated":
            False,

        "comparison_group":
            "final_object_centric",

        "supports_live":
            True,
    },


    "crop_alignment_query_gated_uvd": {

        "id":
            "crop_alignment_query_gated_uvd",

        "label":
            "Crop + Alignment + Query-Gated UVD",

        "mode":
            "alignment_query_gated_uvd",

        "checkpoint":
            PROJECT_ROOT
            / "outputs"
            / "object_zoom"
            / "alignment_query_gated_uvd"
            / "best.pt",

        "description":
            "Uses query-conditioned weighting of "
            "object-relative U, V and D.",

        "geometry": [
            "u",
            "v",
            "d",
        ],

        "gated":
            True,

        "comparison_group":
            "final_object_centric",

        "supports_live":
            True,
    },

}


# ============================================================
# ACCESS HELPERS
# ============================================================

def get_demo_model(
    model_id,
):
    if model_id not in DEMO_MODELS:
        raise KeyError(
            f"Unknown demo model: {model_id}"
        )

    return DEMO_MODELS[
        model_id
    ]


def get_demo_models():
    return list(
        DEMO_MODELS.values()
    )


def get_model_ids():
    return list(
        DEMO_MODELS.keys()
    )


# ============================================================
# FRONTEND VERSION
#
# Path objects cannot be written directly into JSON.
# This returns only browser-safe metadata.
# ============================================================

def get_frontend_models():

    frontend_models = []

    for model in get_demo_models():

        frontend_models.append(
            {
                "id":
                    model["id"],

                "label":
                    model["label"],

                "description":
                    model["description"],

                "geometry":
                    list(
                        model["geometry"]
                    ),

                "gated":
                    bool(
                        model["gated"]
                    ),

                "comparison_group":
                    model[
                        "comparison_group"
                    ],
            }
        )

    return frontend_models


# ============================================================
# VALIDATION
#
# Run before exporting or starting the future API.
# ============================================================

def validate_demo_models():

    errors = []

    for model_id, model in DEMO_MODELS.items():

        if model[
            "id"
        ] != model_id:

            errors.append(
                f"{model_id}: id mismatch"
            )


        checkpoint = Path(
            model[
                "checkpoint"
            ]
        )

        if not checkpoint.is_file():

            errors.append(
                f"{model_id}: checkpoint missing: "
                f"{checkpoint}"
            )


        geometry = model.get(
            "geometry",
            [],
        )

        if not isinstance(
            geometry,
            list,
        ):

            errors.append(
                f"{model_id}: geometry must be a list"
            )


        unknown_geometry = (
            set(
                geometry
            )
            - {
                "x",
                "y",
                "u",
                "v",
                "d",
            }
        )

        if unknown_geometry:

            errors.append(
                f"{model_id}: unsupported geometry: "
                f"{sorted(unknown_geometry)}"
            )


        if (
            model.get(
                "gated",
                False,
            )
            and not {
                "u",
                "v",
                "d",
            }.issubset(
                geometry
            )
        ):

            errors.append(
                f"{model_id}: gated UVD model must "
                "declare u, v and d"
            )


    if errors:

        raise RuntimeError(
            "Demo model registry is invalid:\n"
            + "\n".join(
                f"- {error}"
                for error in errors
            )
        )


    return True