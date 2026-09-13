from pathlib import Path
import sys


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
# IMPORTS
# ============================================================

from datasets.object_centric_robustness_dataset import (
    RobustObjectCentricDataset,
    DEMO_ROBUSTNESS_CONDITIONS,
)

from final_model.demo_registry import (
    get_demo_models,
    validate_demo_models,
)

from final_model.demo_models import (
    predict_sample,
)


# ============================================================
# TEST CONFIG
#
# Same unseen bird/head region we have used previously.
# ============================================================

SPLIT = "test_unseen"

SAMPLE_INDEX = 10


# ============================================================
# MAIN
# ============================================================

def main():

    validate_demo_models()

    models = (
        get_demo_models()
    )


    print()
    print(
        "=" * 78
    )

    print(
        "OBJECT-CENTRIC ROBUSTNESS TEST"
    )

    print(
        "=" * 78
    )


    for condition in (
        DEMO_ROBUSTNESS_CONDITIONS
    ):

        condition_id = (
            condition[
                "id"
            ]
        )


        dataset = (
            RobustObjectCentricDataset(
                split=SPLIT,
                condition_id=condition_id,
                image_size=224,
                context_ratio=0.15,
            )
        )


        sample = dataset[
            SAMPLE_INDEX
        ]


        print()
        print(
            "-" * 78
        )

        print(
            condition[
                "label"
            ]
        )

        print(
            "-" * 78
        )


        print(
            "Sample:",
            sample[
                "sample_id"
            ],
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

        print(
            "Condition:",
            condition_id,
        )

        print(
            "Crop:",
            (
                sample[
                    "crop_x1"
                ],
                sample[
                    "crop_y1"
                ],
                sample[
                    "crop_side"
                ],
            ),
        )

        print()


        # ====================================================
        # RUN ALL CURRENT REGISTERED MODELS
        # ====================================================

        for model in models:

            model_id = (
                model[
                    "id"
                ]
            )


            result = predict_sample(
                sample,
                model_id,
            )


            if result is None:

                print(
                    model[
                        "label"
                    ],
                    "FAILED",
                )

                continue


            print(
                f"{model['label']}"
            )

            print(
                f"    IoU:  "
                f"{result['iou']:.4f}"
            )

            print(
                f"    Dice: "
                f"{result['dice']:.4f}"
            )


            if (
                model.get(
                    "gated",
                    False,
                )
            ):

                gate = (
                    result[
                        "aux"
                    ]
                    .get(
                        "gate_weights"
                    )
                )

                if gate is not None:

                    weights = (
                        gate[0]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    print(
                        "    Gate:"
                    )

                    print(
                        f"        U: "
                        f"{weights[0]:.4f}"
                    )

                    print(
                        f"        V: "
                        f"{weights[1]:.4f}"
                    )

                    print(
                        f"        D: "
                        f"{weights[2]:.4f}"
                    )


    print()
    print(
        "=" * 78
    )

    print(
        "ROBUSTNESS TEST COMPLETE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()