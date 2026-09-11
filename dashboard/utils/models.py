from pathlib import Path

import torch
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from datasets import (
    ObjectCentricDataset,
)

from src.dino_features import (
    get_device,
    load_dino_model,
)

from src.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.alignment_model import (
    PartQueryAlignmentSegmenter,
)

from src.crop_projection import (
    project_crop_prediction_to_full_view,
)

from dashboard.utils.demo_registry import (
    DEMO_MODELS,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = get_device()


# ============================================================
# LEGACY DASHBOARD MODEL LIST
#
# Keep this for the existing Streamlit dashboard.
#
# The actual demo/export/live backend should prefer model IDs
# from demo_registry.py.
# ============================================================

QUALITATIVE_MODELS = {
    "Crop + Alignment":
        "crop_alignment",

    "Crop + Alignment + Relative UV":
        "alignment_relative_uv",

    "Crop + Alignment + Fixed UVD":
        "crop_alignment_fixed_uvd",

    "Crop + Alignment + Query-Gated UVD":
        "crop_alignment_query_gated_uvd",
}


# ============================================================
# LEGACY MODES
#
# These allow older notebook/dashboard code to keep working.
#
# Registered demo models do NOT use these paths. Their exact
# checkpoint path comes from demo_registry.py.
# ============================================================

LEGACY_UVD_MODES = {
    "alignment_fixed_uvd",
    "alignment_query_gated_uvd",
}


# ============================================================
# DATASETS
# ============================================================

@st.cache_resource
def load_unseen_dataset():

    return ObjectCentricDataset(
        split="test_unseen",
        image_size=224,
        context_ratio=0.15,
    )


@st.cache_resource
def load_seen_dataset():

    return ObjectCentricDataset(
        split="test_seen",
        image_size=224,
        context_ratio=0.15,
    )


# ============================================================
# ENCODERS
#
# DINOv2 + CLIP are loaded once and reused.
# This is also what we want later for FastAPI.
# ============================================================

@st.cache_resource
def load_encoders():

    dino = load_dino_model(
        device=DEVICE
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=DEVICE
        )
    )

    return (
        dino,
        clip_model,
        tokenizer,
    )


# ============================================================
# MODEL SPEC RESOLUTION
#
# model_ref can be:
#
#     "crop_alignment"
#     "crop_alignment_fixed_uvd"
#     "crop_alignment_query_gated_uvd"
#
# It can also still be an older mode such as:
#
#     "alignment_mask"
#     "alignment_relative_uv"
#
# Registered models always get their checkpoint from
# demo_registry.py.
# ============================================================

def resolve_model_spec(
    model_ref,
):

    model_ref = str(
        model_ref
    )


    # --------------------------------------------------------
    # Preferred:
    # direct registry model ID
    # --------------------------------------------------------

    if model_ref in DEMO_MODELS:

        registered = (
            DEMO_MODELS[
                model_ref
            ]
        )

        return {
            "registered":
                True,

            "id":
                registered[
                    "id"
                ],

            "label":
                registered[
                    "label"
                ],

            "mode":
                registered[
                    "mode"
                ],

            "checkpoint":
                Path(
                    registered[
                        "checkpoint"
                    ]
                ),

            "geometry":
                list(
                    registered.get(
                        "geometry",
                        [],
                    )
                ),

            "gated":
                bool(
                    registered.get(
                        "gated",
                        False,
                    )
                ),
        }


    # --------------------------------------------------------
    # Backwards compatibility:
    # user passed a registered model's old mode instead
    # of its registry ID.
    # --------------------------------------------------------

    for registered in (
        DEMO_MODELS.values()
    ):

        if (
            registered[
                "mode"
            ]
            == model_ref
        ):

            return {
                "registered":
                    True,

                "id":
                    registered[
                        "id"
                    ],

                "label":
                    registered[
                        "label"
                    ],

                "mode":
                    registered[
                        "mode"
                    ],

                "checkpoint":
                    Path(
                        registered[
                            "checkpoint"
                        ]
                    ),

                "geometry":
                    list(
                        registered.get(
                            "geometry",
                            [],
                        )
                    ),

                "gated":
                    bool(
                        registered.get(
                            "gated",
                            False,
                        )
                    ),
            }


    # --------------------------------------------------------
    # Legacy fallback.
    #
    # This keeps older experiments such as
    # alignment_relative_uv usable even when they are not
    # currently part of the web-demo registry.
    # --------------------------------------------------------

    legacy_path = (
        PROJECT_ROOT
        / "outputs"
        / "object_zoom"
        / model_ref
        / "best.pt"
    )


    geometry = []

    if (
        "relative_uv"
        in model_ref
    ):
        geometry = [
            "u",
            "v",
        ]

    if (
        model_ref
        in LEGACY_UVD_MODES
    ):
        geometry = [
            "u",
            "v",
            "d",
        ]


    return {
        "registered":
            False,

        "id":
            model_ref,

        "label":
            model_ref,

        "mode":
            model_ref,

        "checkpoint":
            legacy_path,

        "geometry":
            geometry,

        "gated":
            (
                model_ref
                == "alignment_query_gated_uvd"
            ),
    }


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def checkpoint_path(
    model_ref,
):

    spec = resolve_model_spec(
        model_ref
    )

    return spec[
        "checkpoint"
    ]


def checkpoint_exists(
    model_ref,
):

    return checkpoint_path(
        model_ref
    ).is_file()


# ============================================================
# SEGMENTATION MODEL
#
# IMPORTANT:
#
# The checkpoint path for registered models now comes from
# demo_registry.py.
#
# This fixes the old behavior where "mode" silently decided
# which outputs/.../best.pt file was loaded.
# ============================================================

@st.cache_resource
def load_segmentation_model(
    model_ref,
):

    spec = resolve_model_spec(
        model_ref
    )

    mode = spec[
        "mode"
    ]

    path = spec[
        "checkpoint"
    ]


    if not path.is_file():

        print(
            "Checkpoint not found:",
            path,
        )

        return None


    (
        dino,
        _,
        _,
    ) = load_encoders()


    model = (
        PartQueryAlignmentSegmenter(
            dino_encoder=dino,
            mode=mode,
        )
        .to(
            DEVICE
        )
    )


    checkpoint = torch.load(
        path,
        map_location=DEVICE,
    )


    if (
        "model_state"
        not in checkpoint
    ):

        raise KeyError(
            f"Checkpoint does not contain "
            f"'model_state': {path}"
        )


    state = checkpoint[
        "model_state"
    ]


    # --------------------------------------------------------
    # Core learned modules
    # --------------------------------------------------------

    model.visual_projection.load_state_dict(
        state[
            "visual_projection"
        ]
    )


    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ]
    )


    model.text_decoder_projection.load_state_dict(
        state[
            "text_decoder_projection"
        ]
    )


    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )


    # --------------------------------------------------------
    # Optional geometry gate
    # --------------------------------------------------------

    if (
        model.geometry_gate
        is not None
    ):

        if (
            "geometry_gate"
            not in state
        ):

            raise KeyError(
                f"Model mode '{mode}' expects "
                f"geometry_gate weights, but "
                f"checkpoint does not contain them: "
                f"{path}"
            )


        model.geometry_gate.load_state_dict(
            state[
                "geometry_gate"
            ]
        )


    model.eval()


    return model


# ============================================================
# TEXT QUERY
# ============================================================

def encode_query(
    query,
):

    (
        _,
        clip_model,
        tokenizer,
    ) = load_encoders()


    return extract_clip_features(
        clip_model,
        tokenizer,
        query,
        device=DEVICE,
    )


# ============================================================
# PREDICT SAMPLE
#
# model_ref should preferably be the registry ID:
#
#     crop_alignment
#     crop_alignment_fixed_uvd
#     crop_alignment_query_gated_uvd
#
# Old mode strings still work for compatibility.
# ============================================================

def predict_sample(
    sample,
    model_ref,
):

    spec = resolve_model_spec(
        model_ref
    )


    mode = spec[
        "mode"
    ]

    geometry = set(
        spec[
            "geometry"
        ]
    )


    model = load_segmentation_model(
        model_ref
    )


    if model is None:
        return None


    # ========================================================
    # CLIP QUERY
    # ========================================================

    text = encode_query(
        sample[
            "query"
        ]
    )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    crop_image = (
        sample[
            "crop_image"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    crop_object = (
        sample[
            "crop_object_mask"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    crop_u = (
        sample[
            "crop_relative_u"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    crop_v = (
        sample[
            "crop_relative_v"
        ]
        .unsqueeze(0)
        .to(
            DEVICE
        )
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        # ----------------------------------------------------
        # Models using D require the sixth input.
        # ----------------------------------------------------

        if (
            "d"
            in geometry
        ):

            crop_d = (
                sample[
                    "crop_boundary_d"
                ]
                .unsqueeze(0)
                .to(
                    DEVICE
                )
            )


            logits, aux = model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
                crop_d,
            )


        # ----------------------------------------------------
        # Alignment / Relative UV family
        # ----------------------------------------------------

        else:

            logits, aux = model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
            )


    # ========================================================
    # CROP PROBABILITY
    # ========================================================

    crop_probability = (
        torch.sigmoid(
            logits
        )[0]
    )


    # ========================================================
    # PROJECT CROP PREDICTION BACK TO FULL 224 VIEW
    # ========================================================

    projected_probability = (
        project_crop_prediction_to_full_view(
            crop_probability,

            original_height=int(
                sample[
                    "original_height"
                ]
            ),

            original_width=int(
                sample[
                    "original_width"
                ]
            ),

            crop_x1=int(
                sample[
                    "crop_x1"
                ]
            ),

            crop_y1=int(
                sample[
                    "crop_y1"
                ]
            ),

            crop_side=int(
                sample[
                    "crop_side"
                ]
            ),

            target_size=224,
        )
    )


    # ========================================================
    # BINARY PREDICTION
    # ========================================================

    prediction = (
        projected_probability
        > 0.5
    )


    target = (
        sample[
            "full_part_mask"
        ]
        > 0.5
    )


    # ========================================================
    # IoU
    # ========================================================

    intersection = (
        prediction
        & target
    ).sum().float()


    union = (
        prediction
        | target
    ).sum().float()


    prediction_sum = (
        prediction
        .sum()
        .float()
    )


    target_sum = (
        target
        .sum()
        .float()
    )


    iou = (
        intersection
        / union.clamp_min(
            1.0
        )
    ).item()


    # ========================================================
    # DICE
    # ========================================================

    dice = (
        2.0
        * intersection
        / (
            prediction_sum
            + target_sum
        ).clamp_min(
            1.0
        )
    ).item()


    # ========================================================
    # RESULT
    # ========================================================

    return {
        "model_id":
            spec[
                "id"
            ],

        "mode":
            mode,

        "checkpoint":
            str(
                spec[
                    "checkpoint"
                ]
            ),

        "crop_probability":
            crop_probability,

        "projected_probability":
            projected_probability,

        "prediction":
            prediction.float(),

        "target":
            target.float(),

        "iou":
            iou,

        "dice":
            dice,

        "aux":
            aux,
    }