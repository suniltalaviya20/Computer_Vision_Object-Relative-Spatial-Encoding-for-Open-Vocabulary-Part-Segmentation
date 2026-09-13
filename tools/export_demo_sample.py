from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from final_model.demo_models import (
    load_unseen_dataset,
    predict_sample,
)

from final_model.demo_registry import (
    get_demo_models,
    get_frontend_models,
    validate_demo_models,
)


# ============================================================
# WEB PROJECT LOCATION
# ============================================================

WEB_ROOT = (
    PROJECT_ROOT
    / "web"
)

ASSETS_ROOT = (
    WEB_ROOT
    / "assets"
    / "examples"
)

CATALOGUE_PATH = (
    WEB_ROOT
    / "data"
    / "catalogue.json"
)

ASSETS_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CATALOGUE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def tensor_to_uint8_rgb(
    tensor,
):
    arr = (
        tensor
        .detach()
        .cpu()
        .permute(
            1,
            2,
            0,
        )
        .numpy()
    )

    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    return (
        arr * 255
    ).astype(
        np.uint8
    )


def tensor_to_uint8_gray(
    tensor,
):
    arr = (
        tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        .astype(
            np.float32
        )
    )

    arr = np.clip(
        arr,
        0.0,
        1.0,
    )

    return (
        arr * 255
    ).astype(
        np.uint8
    )


def save_rgb(
    tensor,
    path,
):
    Image.fromarray(
        tensor_to_uint8_rgb(
            tensor
        )
    ).save(
        path
    )


def save_gray(
    tensor,
    path,
):
    Image.fromarray(
        tensor_to_uint8_gray(
            tensor
        )
    ).save(
        path
    )


def save_binary_mask(
    tensor,
    path,
):
    arr = (
        tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
    )

    arr = (
        (arr > 0.5)
        .astype(
            np.uint8
        )
        * 255
    )

    Image.fromarray(
        arr
    ).save(
        path
    )


# ============================================================
# MODEL PREDICTION OVERLAY
#
# Red = predicted part
# ============================================================

def save_prediction_overlay(
    image_tensor,
    prediction_tensor,
    path,
):
    image = (
        tensor_to_uint8_rgb(
            image_tensor
        )
        .astype(
            np.float32
        )
    )

    prediction = (
        prediction_tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        > 0.5
    )

    overlay = image.copy()

    overlay[
        prediction,
        0
    ] = (
        0.55
        * overlay[
            prediction,
            0
        ]
        + 0.45 * 255
    )

    overlay[
        prediction,
        1
    ] *= 0.55

    overlay[
        prediction,
        2
    ] *= 0.55

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        overlay
    ).save(
        path
    )


# ============================================================
# GROUND TRUTH OVERLAY
#
# Green = true annotated part
# ============================================================

def save_ground_truth_overlay(
    image_tensor,
    mask_tensor,
    path,
):
    image = (
        tensor_to_uint8_rgb(
            image_tensor
        )
        .astype(
            np.float32
        )
    )

    mask = (
        mask_tensor
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        > 0.5
    )

    overlay = image.copy()

    overlay[
        mask,
        0
    ] *= 0.50

    overlay[
        mask,
        1
    ] = (
        0.50
        * overlay[
            mask,
            1
        ]
        + 0.50 * 255
    )

    overlay[
        mask,
        2
    ] *= 0.50

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        overlay
    ).save(
        path
    )


# ============================================================
# LEAKAGE
#
# Fraction of predicted foreground outside parent mask.
# ============================================================

def leakage_score(
    prediction,
    parent_mask,
):
    pred = (
        prediction
        .detach()
        .cpu()
        .squeeze()
        > 0.5
    )

    parent = (
        parent_mask
        .detach()
        .cpu()
        .squeeze()
        > 0.5
    )

    predicted_pixels = int(
        pred.sum().item()
    )

    if predicted_pixels == 0:
        return 0.0

    outside_pixels = int(
        (
            pred
            & (~parent)
        )
        .sum()
        .item()
    )

    return float(
        outside_pixels
        / predicted_pixels
    )


# ============================================================
# VERIFY MODEL REGISTRY
# ============================================================

print()
print(
    "=" * 70
)

print(
    "VALIDATING MODEL REGISTRY"
)

print(
    "=" * 70
)

validate_demo_models()

demo_models = (
    get_demo_models()
)

for model in demo_models:

    print(
        model["label"],
        "->",
        model["checkpoint"],
    )


# ============================================================
# LOAD ONE REAL TEST EXAMPLE
# ============================================================

dataset = (
    load_unseen_dataset()
)

sample_index = 10

sample = dataset[
    sample_index
]

sample_id = str(
    sample[
        "sample_id"
    ]
)

folder_id = (
    sample_id
    .replace(
        ":",
        "__",
    )
)

sample_dir = (
    ASSETS_ROOT
    / folder_id
)

sample_dir.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print(
    "=" * 70
)

print(
    "EXPORTING SAMPLE"
)

print(
    "=" * 70
)

print(
    "Sample:",
    sample_id,
)

print(
    "Object:",
    sample[
        "object_name"
    ],
)

print(
    "Part:",
    sample[
        "part_name"
    ],
)

print(
    "Query:",
    sample[
        "query"
    ],
)

print()


# ============================================================
# COMMON DATASET ASSETS
# ============================================================

save_rgb(
    sample[
        "full_display_image"
    ],
    sample_dir
    / "original.png",
)


save_binary_mask(
    sample[
        "full_object_mask"
    ],
    sample_dir
    / "parent.png",
)


save_binary_mask(
    sample[
        "full_part_mask"
    ],
    sample_dir
    / "ground_truth.png",
)


save_ground_truth_overlay(
    sample[
        "full_display_image"
    ],
    sample[
        "full_part_mask"
    ],
    sample_dir
    / "ground_truth_overlay.png",
)


# ============================================================
# RUN EVERY MODEL IN THE REGISTRY
# ============================================================

results = {}


for model in demo_models:

    frontend_id = (
        model[
            "id"
        ]
    )

    mode = (
        model[
            "mode"
        ]
    )

    geometry = set(
        model.get(
            "geometry",
            [],
        )
    )

    gated = bool(
        model.get(
            "gated",
            False,
        )
    )


    print(
        "Running:",
        model[
            "label"
        ],
    )

    print(
        "Mode:",
        mode,
    )


    result = (
        predict_sample(
            sample,
            frontend_id,
        )
    )


    if result is None:

        raise RuntimeError(
            f"Prediction failed for "
            f"{model['label']}"
        )


    model_dir = (
        sample_dir
        / frontend_id
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    save_prediction_overlay(
        sample[
            "full_display_image"
        ],
        result[
            "prediction"
        ],
        model_dir
        / "prediction.png",
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    save_gray(
        result[
            "projected_probability"
        ],
        model_dir
        / "probability.png",
    )


    result_assets = {

        "prediction": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"{frontend_id}/"
                f"prediction.png",

            "space":
                "full_image",

            "description":
                "Predicted part foreground "
                "overlaid on the original image.",
        },


        "probability": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"{frontend_id}/"
                f"probability.png",

            "space":
                "full_image",

            "description":
                "Projected foreground "
                "probability map.",
        },
    }


    # ========================================================
    # U GEOMETRY
    # ========================================================

    if (
        "u"
        in geometry
    ):

        save_gray(
            result[
                "u"
            ],
            model_dir
            / "u.png",
        )

        result_assets[
            "u"
        ] = {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"{frontend_id}/"
                f"u.png",

            "space":
                "full_image",

            "description":
                "U: horizontal position "
                "relative to the parent object.",
        }


    # ========================================================
    # V GEOMETRY
    # ========================================================

    if (
        "v"
        in geometry
    ):

        save_gray(
            result[
                "v"
            ],
            model_dir
            / "v.png",
        )

        result_assets[
            "v"
        ] = {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"{frontend_id}/"
                f"v.png",

            "space":
                "full_image",

            "description":
                "V: vertical position "
                "relative to the parent object.",
        }


    # ========================================================
    # D GEOMETRY
    # ========================================================

    if (
        "d"
        in geometry
    ):

        save_gray(
            result[
                "d"
            ],
            model_dir
            / "d.png",
        )

        result_assets[
            "d"
        ] = {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"{frontend_id}/"
                f"d.png",

            "space":
                "full_image",

            "description":
                "D: distance to the "
                "2D parent-mask boundary; "
                "not camera depth.",
        }


    # ========================================================
    # METRICS
    # ========================================================

    leakage = (
        leakage_score(
            result[
                "prediction"
            ],
            sample[
                "full_object_mask"
            ],
        )
    )


    entry = {

        "threshold":
            0.5,

        "checkpoint_id":
            str(
                Path(
                    model[
                        "checkpoint"
                    ]
                )
                .relative_to(
                    PROJECT_ROOT
                )
            ),

        "metrics": {

            "iou":
                float(
                    result[
                        "iou"
                    ]
                ),

            "dice":
                float(
                    result[
                        "dice"
                    ]
                ),

            "leakage":
                float(
                    leakage
                ),
        },

        "assets":
            result_assets,
    }


    # ========================================================
    # GATE WEIGHTS
    #
    # Only for models declared as gated.
    # ========================================================

    if gated:

        gate = (
            result[
                "aux"
            ]
            .get(
                "gate_weights"
            )
        )


        if gate is None:

            raise RuntimeError(
                f"{model['label']} "
                "is registered as gated "
                "but returned no gate weights."
            )


        weights = (
            gate[0]
            .detach()
            .cpu()
            .tolist()
        )


        if len(
            weights
        ) != 3:

            raise RuntimeError(
                "Expected U/V/D gate "
                "to contain exactly "
                "three weights."
            )


        entry[
            "gate_weights"
        ] = {

            "u":
                float(
                    weights[0]
                ),

            "v":
                float(
                    weights[1]
                ),

            "d":
                float(
                    weights[2]
                ),
        }


    results[
        frontend_id
    ] = entry


    print(
        "  IoU:",
        f"{result['iou']:.4f}",
    )

    print(
        "  Dice:",
        f"{result['dice']:.4f}",
    )

    print(
        "  Leakage:",
        f"{leakage:.4f}",
    )


    if gated:

        print(
            "  Gate:",
            entry[
                "gate_weights"
            ],
        )


    print()


# ============================================================
# SAMPLE ENTRY
# ============================================================

sample_entry = {

    "id":
        sample_id,

    "object":
        str(
            sample[
                "object_name"
            ]
        ),

    "part":
        str(
            sample[
                "part_name"
            ]
        ),

    "query":
        str(
            sample[
                "query"
            ]
        ),

    "split":
        "test_unseen",

    "assets": {

        "original": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"original.png",

            "space":
                "full_image",
        },


        "parent": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"parent.png",

            "space":
                "full_image",
        },


        "ground_truth": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"ground_truth.png",

            "space":
                "full_image",
        },


        "ground_truth_overlay": {

            "file":
                f"assets/examples/"
                f"{folder_id}/"
                f"ground_truth_overlay.png",

            "space":
                "full_image",

            "description":
                "Dataset ground-truth "
                "part overlaid in green.",
        },
    },

    "results":
        results,
}


# ============================================================
# CATALOGUE
# ============================================================

catalogue = {

    "schema_version":
        1,

    # Automatically comes from demo_registry.py
    "models":
        get_frontend_models(),

    "samples": [
        sample_entry
    ],
}


CATALOGUE_PATH.write_text(
    json.dumps(
        catalogue,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FINISHED
# ============================================================

print(
    "=" * 70
)

print(
    "EXPORT COMPLETE"
)

print(
    "=" * 70
)

print(
    "Sample:",
    sample_id,
)

print(
    "Assets:",
    sample_dir,
)

print(
    "Catalogue:",
    CATALOGUE_PATH,
)

print(
    "Models exported:",
    len(
        demo_models
    ),
)
