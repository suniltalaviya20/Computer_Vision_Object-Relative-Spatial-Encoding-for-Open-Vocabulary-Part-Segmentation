from pathlib import Path
from collections import defaultdict
import json
import random
import sys

import numpy as np
from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from dashboard.utils.models import (
    load_seen_dataset,
    load_unseen_dataset,
    predict_sample,
)

from dashboard.utils.demo_registry import (
    get_demo_models,
    get_frontend_models,
    validate_demo_models,
)


# ============================================================
# EXPORT CONFIGURATION
# ============================================================

PARTS_PER_PARENT = 4

EXAMPLES_PER_PART = 2

RANDOM_SEED = 42


# ============================================================
# WEB LOCATION
# ============================================================

WEB_ROOT = (
    Path.home()
    / "cv_project"
    / "Computer_Vision_Object-Relative-Spatial-Encoding-for-Open-Vocabulary-Part-Segmentation"
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
        (
            arr > 0.5
        ).astype(
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
# PREDICTION OVERLAY
#
# Red = prediction
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
# Green = annotation
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
# GROUP DATASET BY:
#
# parent -> part -> sample indices
# ============================================================

def build_groups(
    dataset,
):
    groups = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        parent = str(
            sample[
                "object_name"
            ]
        )

        part = str(
            sample[
                "part_name"
            ]
        )

        groups[
            parent
        ][
            part
        ].append(
            index
        )

    return groups


# ============================================================
# SELECT BALANCED DEMO EXAMPLES
#
# Strategy:
#
# 1. every parent appears
# 2. choose most frequent parts
# 3. choose two deterministic examples per part
# ============================================================

def select_samples(
    dataset,
    split_name,
):
    groups = build_groups(
        dataset
    )

    rng = random.Random(
        RANDOM_SEED
    )

    selected = []

    print()
    print(
        "=" * 70
    )

    print(
        "SELECTING",
        split_name.upper(),
    )

    print(
        "=" * 70
    )

    for parent in sorted(
        groups
    ):

        part_groups = (
            groups[
                parent
            ]
        )

        # Rank by sample count first.
        # Alphabetical name breaks ties deterministically.
        ranked_parts = sorted(
            part_groups.keys(),
            key=lambda part: (
                -len(
                    part_groups[
                        part
                    ]
                ),
                part,
            ),
        )

        chosen_parts = (
            ranked_parts[
                :PARTS_PER_PARENT
            ]
        )

        print()
        print(
            parent,
            "->",
            ", ".join(
                chosen_parts
            ),
        )

        for part in chosen_parts:

            indices = list(
                part_groups[
                    part
                ]
            )

            number_to_choose = min(
                EXAMPLES_PER_PART,
                len(
                    indices
                ),
            )

            if (
                len(
                    indices
                )
                <= number_to_choose
            ):
                chosen_indices = (
                    indices
                )

            else:
                chosen_indices = (
                    rng.sample(
                        indices,
                        number_to_choose,
                    )
                )

            chosen_indices = sorted(
                chosen_indices
            )

            for index in chosen_indices:

                selected.append(
                    {
                        "split":
                            split_name,

                        "index":
                            index,

                        "parent":
                            parent,

                        "part":
                            part,
                    }
                )

                print(
                    "   ",
                    part,
                    "sample index",
                    index,
                )

    return selected


# ============================================================
# EXPORT ONE SAMPLE
# ============================================================

def export_sample(
    sample,
    split_name,
    demo_models,
):

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


    # ========================================================
    # COMMON ASSETS
    # ========================================================

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


    results = {}


    # ========================================================
    # RUN REGISTERED MODELS
    # ========================================================

    for model in demo_models:

        model_id = (
            model[
                "id"
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


        result = predict_sample(
            sample,
            model_id,
        )


        if result is None:

            raise RuntimeError(
                f"Prediction failed for "
                f"{model['label']}"
            )


        model_dir = (
            sample_dir
            / model_id
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ====================================================
        # PREDICTION
        # ====================================================

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


        # ====================================================
        # PROBABILITY
        # ====================================================

        save_gray(
            result[
                "projected_probability"
            ],
            model_dir
            / "probability.png",
        )


        assets = {

            "prediction": {

                "file":
                    f"assets/examples/"
                    f"{folder_id}/"
                    f"{model_id}/"
                    f"prediction.png",

                "space":
                    "full_image",

                "description":
                    "Predicted part foreground "
                    "overlaid in red.",
            },


            "probability": {

                "file":
                    f"assets/examples/"
                    f"{folder_id}/"
                    f"{model_id}/"
                    f"probability.png",

                "space":
                    "full_image",

                "description":
                    "Projected part probability.",
            },
        }


        # ====================================================
        # U
        # ====================================================

        if "u" in geometry:

            save_gray(
                sample[
                    "crop_relative_u"
                ],
                model_dir
                / "u.png",
            )

            assets[
                "u"
            ] = {

                "file":
                    f"assets/examples/"
                    f"{folder_id}/"
                    f"{model_id}/"
                    f"u.png",

                "space":
                    "crop",

                "description":
                    "Horizontal position "
                    "relative to the parent object.",
            }


        # ====================================================
        # V
        # ====================================================

        if "v" in geometry:

            save_gray(
                sample[
                    "crop_relative_v"
                ],
                model_dir
                / "v.png",
            )

            assets[
                "v"
            ] = {

                "file":
                    f"assets/examples/"
                    f"{folder_id}/"
                    f"{model_id}/"
                    f"v.png",

                "space":
                    "crop",

                "description":
                    "Vertical position "
                    "relative to the parent object.",
            }


        # ====================================================
        # D
        # ====================================================

        if "d" in geometry:

            save_gray(
                sample[
                    "crop_boundary_d"
                ],
                model_dir
                / "d.png",
            )

            assets[
                "d"
            ] = {

                "file":
                    f"assets/examples/"
                    f"{folder_id}/"
                    f"{model_id}/"
                    f"d.png",

                "space":
                    "crop",

                "description":
                    "2D distance to the "
                    "parent-mask boundary.",
            }


        leakage = leakage_score(
            result[
                "prediction"
            ],
            sample[
                "full_object_mask"
            ],
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
                assets,
        }


        # ====================================================
        # TRUE GATE WEIGHTS
        # ====================================================

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
                    "returned no gate weights."
                )

            weights = (
                gate[0]
                .detach()
                .cpu()
                .tolist()
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
            model_id
        ] = entry


    # ========================================================
    # SAMPLE CATALOGUE ENTRY
    # ========================================================

    return {

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
            split_name,

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
# MAIN
# ============================================================

def main():

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
            model[
                "label"
            ],
            "->",
            model[
                "checkpoint"
            ],
        )


    # ========================================================
    # LOAD DATA
    # ========================================================

    seen_dataset = (
        load_seen_dataset()
    )

    unseen_dataset = (
        load_unseen_dataset()
    )


    # ========================================================
    # SELECT BALANCED EXAMPLES
    # ========================================================

    selected_seen = select_samples(
        seen_dataset,
        "test_seen",
    )

    selected_unseen = select_samples(
        unseen_dataset,
        "test_unseen",
    )


    selections = []

    for item in selected_seen:

        selections.append(
            (
                seen_dataset,
                item,
            )
        )

    for item in selected_unseen:

        selections.append(
            (
                unseen_dataset,
                item,
            )
        )


    print()
    print(
        "=" * 70
    )

    print(
        "EXPORT PLAN"
    )

    print(
        "=" * 70
    )

    print(
        "Seen examples:",
        len(
            selected_seen
        ),
    )

    print(
        "Unseen examples:",
        len(
            selected_unseen
        ),
    )

    print(
        "Total examples:",
        len(
            selections
        ),
    )

    print(
        "Models:",
        len(
            demo_models
        ),
    )


    # ========================================================
    # EXPORT
    # ========================================================

    catalogue_samples = []


    for position, (
        dataset,
        item,
    ) in enumerate(
        selections,
        start=1,
    ):

        sample = dataset[
            item[
                "index"
            ]
        ]


        print()
        print(
            f"[{position}/{len(selections)}]",
            item[
                "split"
            ],
            "|",
            item[
                "parent"
            ],
            "|",
            item[
                "part"
            ],
            "|",
            sample[
                "sample_id"
            ],
        )


        sample_entry = export_sample(
            sample,
            item[
                "split"
            ],
            demo_models,
        )


        catalogue_samples.append(
            sample_entry
        )


        for model in demo_models:

            metrics = (
                sample_entry[
                    "results"
                ][
                    model[
                        "id"
                    ]
                ][
                    "metrics"
                ]
            )

            print(
                "   ",
                model[
                    "label"
                ],
                "| IoU",
                f"{metrics['iou']:.3f}",
                "| Dice",
                f"{metrics['dice']:.3f}",
            )


    # ========================================================
    # WRITE CATALOGUE
    # ========================================================

    catalogue = {

        "schema_version":
            1,

        "models":
            get_frontend_models(),

        "samples":
            catalogue_samples,
    }


    CATALOGUE_PATH.write_text(
        json.dumps(
            catalogue,
            indent=2,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # DONE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "BATCH EXPORT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Seen:",
        len(
            selected_seen
        ),
    )

    print(
        "Unseen:",
        len(
            selected_unseen
        ),
    )

    print(
        "Total:",
        len(
            catalogue_samples
        ),
    )

    print(
        "Catalogue:",
        CATALOGUE_PATH,
    )

    print()


if __name__ == "__main__":
    main()