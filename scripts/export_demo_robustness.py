from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROBUSTNESS_PIPELINE_VERSION = 2
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

from datasets.object_centric_robustness_dataset import (
    RobustObjectCentricDataset,
    DEMO_ROBUSTNESS_CONDITIONS,
)


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


# ============================================================
# CONDITIONS
#
# Clean already exists in the normal catalogue.
#
# Therefore this script only has to generate the six
# additional robustness conditions.
# ============================================================

ROBUSTNESS_CONDITIONS = [
    condition
    for condition in DEMO_ROBUSTNESS_CONDITIONS
    if condition["id"] != "clean"
]


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
        )
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
# PREDICTION OVERLAY
#
# Red = model prediction
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
# Green = dataset GT
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
# CHECKPOINT ID
# ============================================================

def checkpoint_id(
    model,
):
    path = Path(
        model[
            "checkpoint"
        ]
    )

    try:
        return str(
            path.relative_to(
                PROJECT_ROOT
            )
        )

    except ValueError:
        return str(
            path
        )


# ============================================================
# WRITE CATALOGUE SAFELY
# ============================================================

def save_catalogue(
    catalogue,
):
    temporary_path = (
        CATALOGUE_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    temporary_path.write_text(
        json.dumps(
            catalogue,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        CATALOGUE_PATH
    )


# ============================================================
# BUILD SAMPLE-ID -> DATASET-INDEX MAP
#
# The existing catalogue stores sample IDs but not the dataset
# index. We find those indices once before robustness export.
# ============================================================

def build_index_map(
    dataset,
    wanted_ids,
    split_name,
):
    wanted_ids = set(
        wanted_ids
    )

    found = {}

    print()
    print(
        "Finding dataset indices for",
        split_name,
    )

    for index in range(
        len(dataset)
    ):

        sample = dataset[
            index
        ]

        sample_id = str(
            sample[
                "sample_id"
            ]
        )

        if sample_id in wanted_ids:

            if sample_id in found:

                raise RuntimeError(
                    f"Duplicate sample ID "
                    f"in {split_name}: "
                    f"{sample_id}"
                )

            found[
                sample_id
            ] = index


        if (
            len(found)
            == len(wanted_ids)
        ):
            break


        if (
            index > 0
            and index % 500 == 0
        ):

            print(
                f"  scanned {index}/"
                f"{len(dataset)}"
            )


    missing = (
        wanted_ids
        - set(
            found
        )
    )

    if missing:

        preview = sorted(
            missing
        )[:10]

        raise RuntimeError(
            f"Could not find "
            f"{len(missing)} catalogue samples "
            f"in {split_name}.\n"
            f"Examples: {preview}"
        )


    print(
        "  found:",
        len(
            found
        ),
    )

    return found


# ============================================================
# VERIFY CLEAN CATALOGUE MATCHES CURRENT REGISTRY
#
# This protects us later when you replace the models.
#
# After replacing models:
#
# 1. rerun export_demo_catalogue.py
# 2. rerun this script
#
# If you forget step 1, this script will stop instead of
# mixing old clean results with new robustness predictions.
# ============================================================

def validate_catalogue_against_registry(
    catalogue,
    demo_models,
):

    catalogue_model_ids = {
        model[
            "id"
        ]
        for model in catalogue.get(
            "models",
            []
        )
    }

    registry_model_ids = {
        model[
            "id"
        ]
        for model in demo_models
    }


    if (
        catalogue_model_ids
        != registry_model_ids
    ):

        raise RuntimeError(
            "catalogue.json does not match "
            "the current demo model registry.\n"
            "Run:\n"
            "python scripts/export_demo_catalogue.py\n"
            "before exporting robustness."
        )


    if not catalogue.get(
        "samples"
    ):

        raise RuntimeError(
            "catalogue.json contains no samples."
        )


    # --------------------------------------------------------
    # Check every clean sample.
    # --------------------------------------------------------

    for sample in catalogue[
        "samples"
    ]:

        clean_results = (
            sample.get(
                "results",
                {}
            )
        )


        for model in demo_models:

            model_id = (
                model[
                    "id"
                ]
            )

            expected_checkpoint = (
                checkpoint_id(
                    model
                )
            )


            if (
                model_id
                not in clean_results
            ):

                raise RuntimeError(
                    f"Clean result missing for "
                    f"{model_id} on "
                    f"{sample['id']}.\n"
                    "Re-run "
                    "export_demo_catalogue.py."
                )


            existing_checkpoint = (
                clean_results[
                    model_id
                ]
                .get(
                    "checkpoint_id"
                )
            )


            if (
                existing_checkpoint
                != expected_checkpoint
            ):

                raise RuntimeError(
                    f"Clean checkpoint mismatch "
                    f"for {model_id}.\n"
                    f"Catalogue: "
                    f"{existing_checkpoint}\n"
                    f"Registry: "
                    f"{expected_checkpoint}\n"
                    "Re-run "
                    "export_demo_catalogue.py."
                )


# ============================================================
# CHECK WHETHER AN EXISTING CONDITION IS CURRENT
#
# This makes the script resumable.
#
# If it is interrupted, running it again skips conditions
# already exported with the same checkpoints.
#
# If checkpoints change, those conditions are recomputed.
# ============================================================

def condition_is_current(
    entry,
    demo_models,
):

    if not entry:
        return False

    if (
        entry.get(
            "pipeline_version"
        )
        != ROBUSTNESS_PIPELINE_VERSION
    ):
        return False


    results = entry.get(
        "results",
        {}
    )


    for model in demo_models:

        model_id = (
            model[
                "id"
            ]
        )

        if (
            model_id
            not in results
        ):
            return False


        if (
            results[
                model_id
            ].get(
                "checkpoint_id"
            )
            != checkpoint_id(
                model
            )
        ):
            return False


    for asset in (
        entry.get(
            "assets",
            {}
        ).values()
    ):

        file_name = (
            asset.get(
                "file"
            )
        )

        if not file_name:
            return False

        if not (
            WEB_ROOT
            / file_name
        ).is_file():
            return False


    for result in results.values():

        for asset in (
            result.get(
                "assets",
                {}
            ).values()
        ):

            file_name = (
                asset.get(
                    "file"
                )
            )

            if not file_name:
                return False

            if not (
                WEB_ROOT
                / file_name
            ).is_file():
                return False


    return True


# ============================================================
# EXPORT ONE MODEL RESULT
# ============================================================

def export_model_result(
    sample,
    condition_dir,
    folder_id,
    condition_id,
    model,
):

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


    # ========================================================
    # REAL INFERENCE
    # ========================================================

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
        condition_dir
        / model_id
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


    base_path = (
        f"assets/examples/"
        f"{folder_id}/"
        f"robustness/"
        f"{condition_id}/"
        f"{model_id}/"
    )


    assets = {

        "prediction": {

            "file":
                base_path
                + "prediction.png",

            "space":
                "full_image",

            "description":
                "Prediction under the "
                "selected robustness condition.",
        },


        "probability": {

            "file":
                base_path
                + "probability.png",

            "space":
                "full_image",

            "description":
                "Part probability under the "
                "selected robustness condition.",
        },
    }


    # ========================================================
    # U
    # ========================================================

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
                base_path
                + "u.png",

            "space":
                "crop",

            "description":
                "U coordinate after "
                "the robustness perturbation.",
        }


    # ========================================================
    # V
    # ========================================================

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
                base_path
                + "v.png",

            "space":
                "crop",

            "description":
                "V coordinate after "
                "the robustness perturbation.",
        }


    # ========================================================
    # D
    # ========================================================

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
                base_path
                + "d.png",

            "space":
                "crop",

            "description":
                "Boundary-distance map after "
                "the robustness perturbation.",
        }


    # ========================================================
    # METRICS
    # ========================================================

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
            checkpoint_id(
                model
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


    # ========================================================
    # TRUE GATE WEIGHTS
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
                "Expected exactly three "
                "U/V/D gate weights."
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


    return entry


# ============================================================
# EXPORT ONE CONDITION
# ============================================================

def export_condition(
    sample,
    condition,
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


    condition_id = (
        condition[
            "id"
        ]
    )


    condition_dir = (
        ASSETS_ROOT
        / folder_id
        / "robustness"
        / condition_id
    )

    condition_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # COMMON CONDITION ASSETS
    # ========================================================

    save_rgb(
        sample[
            "full_display_image"
        ],
        condition_dir
        / "original.png",
    )


    save_binary_mask(
        sample[
            "full_object_mask"
        ],
        condition_dir
        / "parent.png",
    )


    save_binary_mask(
        sample[
            "full_part_mask"
        ],
        condition_dir
        / "ground_truth.png",
    )


    save_ground_truth_overlay(
        sample[
            "full_display_image"
        ],
        sample[
            "full_part_mask"
        ],
        condition_dir
        / "ground_truth_overlay.png",
    )


    base_path = (
        f"assets/examples/"
        f"{folder_id}/"
        f"robustness/"
        f"{condition_id}/"
    )

    results = {}


    # ========================================================
    # ALL CURRENT REGISTERED MODELS
    # ========================================================

    for model in demo_models:

        result_entry = (
            export_model_result(
                sample,
                condition_dir,
                folder_id,
                condition_id,
                model,
            )
        )


        results[
            model[
                "id"
            ]
        ] = (
            result_entry
        )


    # ========================================================
    # CONDITION CATALOGUE ENTRY
    # ========================================================

    return {

        "pipeline_version":
            ROBUSTNESS_PIPELINE_VERSION,

        "id":
            condition_id,

        "label":
            condition[
                "label"
            ],

        "type":
            condition[
                "type"
            ],

        "value":
            condition[
                "value"
            ],

        "assets": {

            "original": {

                "file":
                    base_path
                    + "original.png",

                "space":
                    "full_image",
            },


            "parent": {

                "file":
                    base_path
                    + "parent.png",

                "space":
                    "full_image",
            },


            "ground_truth": {

                "file":
                    base_path
                    + "ground_truth.png",

                "space":
                    "full_image",
            },


            "ground_truth_overlay": {

                "file":
                    base_path
                    + "ground_truth_overlay.png",

                "space":
                    "full_image",

                "description":
                    "Ground-truth part under "
                    "the selected condition.",
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
        "=" * 78
    )

    print(
        "DEMO ROBUSTNESS EXPORT"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # REGISTRY
    # ========================================================

    validate_demo_models()

    demo_models = (
        get_demo_models()
    )


    print()
    print(
        "Models:"
    )

    for model in demo_models:

        print(
            " ",
            model[
                "label"
            ],
        )


    # ========================================================
    # LOAD EXISTING 111-SAMPLE CATALOGUE
    # ========================================================

    if not CATALOGUE_PATH.is_file():

        raise FileNotFoundError(
            f"Catalogue not found:\n"
            f"{CATALOGUE_PATH}\n\n"
            "Run export_demo_catalogue.py first."
        )


    catalogue = json.loads(
        CATALOGUE_PATH.read_text(
            encoding="utf-8"
        )
    )


    validate_catalogue_against_registry(
        catalogue,
        demo_models,
    )


    catalogue_samples = (
        catalogue[
            "samples"
        ]
    )


    print()
    print(
        "Catalogue samples:",
        len(
            catalogue_samples
        ),
    )

    print(
        "Additional robustness conditions:",
        len(
            ROBUSTNESS_CONDITIONS
        ),
    )

    print(
        "Models:",
        len(
            demo_models
        ),
    )

    print(
        "Maximum new predictions:",
        (
            len(
                catalogue_samples
            )
            * len(
                ROBUSTNESS_CONDITIONS
            )
            * len(
                demo_models
            )
        ),
    )


    # ========================================================
    # ROBUSTNESS CONDITION METADATA
    #
    # Includes Clean even though Clean uses the existing
    # sample assets/results.
    # ========================================================

    catalogue[
        "robustness_conditions"
    ] = [

        {
            "id":
                condition[
                    "id"
                ],

            "label":
                condition[
                    "label"
                ],

            "type":
                condition[
                    "type"
                ],

            "value":
                condition[
                    "value"
                ],
        }

        for condition
        in DEMO_ROBUSTNESS_CONDITIONS
    ]


    # Keep frontend model metadata synchronized.
    catalogue[
        "models"
    ] = (
        get_frontend_models()
    )


    # ========================================================
    # LOAD NORMAL DATASETS
    #
    # Used only to locate the 111 catalogue samples.
    # ========================================================

    seen_dataset = (
        load_seen_dataset()
    )

    unseen_dataset = (
        load_unseen_dataset()
    )


    seen_ids = [

        sample[
            "id"
        ]

        for sample
        in catalogue_samples

        if (
            sample[
                "split"
            ]
            == "test_seen"
        )
    ]


    unseen_ids = [

        sample[
            "id"
        ]

        for sample
        in catalogue_samples

        if (
            sample[
                "split"
            ]
            == "test_unseen"
        )
    ]


    seen_index_map = (
        build_index_map(
            seen_dataset,
            seen_ids,
            "test_seen",
        )
    )


    unseen_index_map = (
        build_index_map(
            unseen_dataset,
            unseen_ids,
            "test_unseen",
        )
    )


    # ========================================================
    # CREATE ONE ROBUSTNESS DATASET PER CONDITION / SPLIT
    # ========================================================

    robustness_datasets = {

        "test_seen": {},

        "test_unseen": {},
    }


    for condition in (
        ROBUSTNESS_CONDITIONS
    ):

        condition_id = (
            condition[
                "id"
            ]
        )


        robustness_datasets[
            "test_seen"
        ][
            condition_id
        ] = (
            RobustObjectCentricDataset(
                split="test_seen",
                condition_id=condition_id,
                image_size=224,
                context_ratio=0.15,
            )
        )


        robustness_datasets[
            "test_unseen"
        ][
            condition_id
        ] = (
            RobustObjectCentricDataset(
                split="test_unseen",
                condition_id=condition_id,
                image_size=224,
                context_ratio=0.15,
            )
        )


    # ========================================================
    # EXPORT ALL 111 SAMPLES
    # ========================================================

    completed_predictions = 0
    skipped_conditions = 0


    for sample_number, catalogue_sample in enumerate(
        catalogue_samples,
        start=1,
    ):

        sample_id = (
            catalogue_sample[
                "id"
            ]
        )

        split_name = (
            catalogue_sample[
                "split"
            ]
        )


        if split_name == "test_seen":

            dataset_index = (
                seen_index_map[
                    sample_id
                ]
            )

        elif split_name == "test_unseen":

            dataset_index = (
                unseen_index_map[
                    sample_id
                ]
            )

        else:

            raise ValueError(
                f"Unknown split in catalogue: "
                f"{split_name}"
            )


        print()
        print(
            "=" * 78
        )

        print(
            f"[{sample_number}/"
            f"{len(catalogue_samples)}] "
            f"{split_name} | "
            f"{catalogue_sample['object']} | "
            f"{catalogue_sample['part']}"
        )

        print(
            sample_id
        )

        print(
            "=" * 78
        )


        # ====================================================
        # ENSURE ROBUSTNESS DICT EXISTS
        # ====================================================

        if (
            "robustness"
            not in catalogue_sample
        ):

            catalogue_sample[
                "robustness"
            ] = {}


        # ====================================================
        # SIX NON-CLEAN CONDITIONS
        # ====================================================

        for condition in (
            ROBUSTNESS_CONDITIONS
        ):

            condition_id = (
                condition[
                    "id"
                ]
            )


            existing_entry = (
                catalogue_sample[
                    "robustness"
                ].get(
                    condition_id
                )
            )


            # =================================================
            # RESUME SUPPORT
            # =================================================

            if condition_is_current(
                existing_entry,
                demo_models,
            ):

                print(
                    f"  {condition['label']}: "
                    "already current, skipping"
                )

                skipped_conditions += 1

                continue


            print(
                f"  {condition['label']}"
            )


            robust_dataset = (
                robustness_datasets[
                    split_name
                ][
                    condition_id
                ]
            )


            robust_sample = (
                robust_dataset[
                    dataset_index
                ]
            )


            # =================================================
            # SANITY CHECK
            # =================================================

            robust_sample_id = str(
                robust_sample[
                    "sample_id"
                ]
            )


            if (
                robust_sample_id
                != sample_id
            ):

                raise RuntimeError(
                    "Dataset index mismatch:\n"
                    f"Expected: {sample_id}\n"
                    f"Received: {robust_sample_id}"
                )


            # =================================================
            # EXPORT CONDITION
            # =================================================

            condition_entry = (
                export_condition(
                    robust_sample,
                    condition,
                    demo_models,
                )
            )


            catalogue_sample[
                "robustness"
            ][
                condition_id
            ] = (
                condition_entry
            )


            for model in demo_models:

                model_id = (
                    model[
                        "id"
                    ]
                )

                metrics = (
                    condition_entry[
                        "results"
                    ][
                        model_id
                    ][
                        "metrics"
                    ]
                )


                print(
                    "     ",
                    model[
                        "label"
                    ],
                    "| IoU",
                    f"{metrics['iou']:.3f}",
                    "| Dice",
                    f"{metrics['dice']:.3f}",
                )


                completed_predictions += 1


        # ====================================================
        # SAVE AFTER EVERY SAMPLE
        #
        # If the job is interrupted you lose at most one
        # sample, and normally less because its finished
        # conditions are already present in memory.
        # ====================================================

        save_catalogue(
            catalogue
        )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_catalogue(
        catalogue
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "=" * 78
    )

    print(
        "ROBUSTNESS EXPORT COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "Samples:",
        len(
            catalogue_samples
        ),
    )

    print(
        "Website conditions:",
        len(
            DEMO_ROBUSTNESS_CONDITIONS
        ),
    )

    print(
        "New predictions computed:",
        completed_predictions,
    )

    print(
        "Conditions skipped/resumed:",
        skipped_conditions,
    )

    print(
        "Catalogue:",
        CATALOGUE_PATH,
    )

    print()


if __name__ == "__main__":
    main()