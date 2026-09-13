from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


BASELINE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
)


GEOMETRY_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "geometry"
)


ALIGNMENT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "part_query_alignment"
)


ZOOM_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "object_zoom"
)


FINAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_analysis"
)


FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPERIMENTS = {
    "baseline/part_only":
        BASELINE_ROOT
        / "part_only"
        / "history.json",

    "baseline/object_mask":
        BASELINE_ROOT
        / "object_mask"
        / "history.json",

    "geometry/object_mask":
        GEOMETRY_ROOT
        / "object_mask"
        / "history.json",

    "geometry/absolute_xy":
        GEOMETRY_ROOT
        / "absolute_xy"
        / "history.json",

    "geometry/relative_uv":
        GEOMETRY_ROOT
        / "relative_uv"
        / "history.json",

    "alignment/mask_baseline":
        ALIGNMENT_ROOT
        / "mask_baseline"
        / "history.json",

    "alignment/alignment_mask":
        ALIGNMENT_ROOT
        / "alignment_mask"
        / "history.json",

    "alignment/alignment_relative_uv":
        ALIGNMENT_ROOT
        / "alignment_relative_uv"
        / "history.json",

    "object_zoom/mask_baseline":
        ZOOM_ROOT
        / "mask_baseline"
        / "history.json",

    "object_zoom/alignment_mask":
        ZOOM_ROOT
        / "alignment_mask"
        / "history.json",

    "object_zoom/alignment_relative_uv":
        ZOOM_ROOT
        / "alignment_relative_uv"
        / "history.json",
}


TEST_RESULTS_PATH = (
    FINAL_ROOT
    / "test_results.json"
)


def load_json(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def get_value(
    row,
    names,
):
    for name in names:
        if name in row:
            return row[name]

    return None


def find_best_epoch(
    history,
):
    if not history:
        return None

    best_row = None
    best_iou = None

    for row in history:
        val_iou = get_value(
            row,
            [
                "val_iou",
            ],
        )

        if val_iou is None:
            validation = row.get(
                "validation"
            )

            if isinstance(
                validation,
                dict,
            ):
                val_iou = validation.get(
                    "iou"
                )

        if val_iou is None:
            continue

        if (
            best_iou is None
            or val_iou > best_iou
        ):
            best_iou = val_iou
            best_row = row

    return best_row


def make_experiment_row(
    name,
    history_path,
):
    if not history_path.is_file():
        return {
            "experiment":
                name,

            "status":
                "missing",

            "best_epoch":
                None,

            "val_loss":
                None,

            "val_iou":
                None,

            "val_dice":
                None,
        }

    history = load_json(
        history_path
    )

    best_row = find_best_epoch(
        history
    )

    if best_row is None:
        return {
            "experiment":
                name,

            "status":
                "no_valid_epoch",

            "best_epoch":
                None,

            "val_loss":
                None,

            "val_iou":
                None,

            "val_dice":
                None,
        }

    val_loss = get_value(
        best_row,
        [
            "val_loss",
        ],
    )

    val_iou = get_value(
        best_row,
        [
            "val_iou",
        ],
    )

    val_dice = get_value(
        best_row,
        [
            "val_dice",
        ],
    )

    if (
        val_loss is None
        or val_iou is None
        or val_dice is None
    ):
        validation = best_row.get(
            "validation"
        )

        if isinstance(
            validation,
            dict,
        ):
            if val_loss is None:
                val_loss = validation.get(
                    "loss"
                )

            if val_iou is None:
                val_iou = validation.get(
                    "iou"
                )

            if val_dice is None:
                val_dice = validation.get(
                    "dice"
                )

    return {
        "experiment":
            name,

        "status":
            "available",

        "best_epoch":
            best_row.get(
                "epoch"
            ),

        "val_loss":
            val_loss,

        "val_iou":
            val_iou,

        "val_dice":
            val_dice,
    }


def build_validation_table():
    rows = []

    for name, history_path in (
        EXPERIMENTS.items()
    ):
        rows.append(
            make_experiment_row(
                name,
                history_path,
            )
        )

    return pd.DataFrame(
        rows
    )


def print_validation_table(
    table,
):
    display_table = table.copy()

    for column in [
        "val_loss",
        "val_iou",
        "val_dice",
    ]:
        display_table[column] = (
            display_table[column]
            .apply(
                lambda value:
                    (
                        f"{value:.4f}"
                        if pd.notna(value)
                        else "-"
                    )
            )
        )

    display_table[
        "best_epoch"
    ] = (
        display_table[
            "best_epoch"
        ].apply(
            lambda value:
                (
                    int(value)
                    if pd.notna(value)
                    else "-"
                )
        )
    )

    print(
        display_table.to_string(
            index=False
        )
    )


def build_test_table():
    if not TEST_RESULTS_PATH.is_file():
        return None

    results = load_json(
        TEST_RESULTS_PATH
    )

    rows = []

    for result in results:
        row = {
            "experiment":
                result.get(
                    "experiment"
                ),

            "status":
                result.get(
                    "status",
                    "unknown",
                ),

            "seen_iou":
                result.get(
                    "test_seen_iou"
                ),

            "unseen_iou":
                result.get(
                    "test_unseen_iou"
                ),

            "seen_dice":
                result.get(
                    "test_seen_dice"
                ),

            "unseen_dice":
                result.get(
                    "test_unseen_dice"
                ),

            "seen_outside_ratio":
                result.get(
                    "test_seen_outside_ratio"
                ),

            "unseen_outside_ratio":
                result.get(
                    "test_unseen_outside_ratio"
                ),

            "seen_samples":
                result.get(
                    "test_seen_samples"
                ),

            "unseen_samples":
                result.get(
                    "test_unseen_samples"
                ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def print_test_table(
    table,
):
    display_table = table.copy()

    for column in [
        "seen_iou",
        "unseen_iou",
        "seen_dice",
        "unseen_dice",
        "seen_outside_ratio",
        "unseen_outside_ratio",
    ]:
        display_table[
            column
        ] = (
            display_table[
                column
            ].apply(
                lambda value:
                    (
                        f"{value:.4f}"
                        if pd.notna(value)
                        else "-"
                    )
            )
        )

    print(
        display_table[
            [
                "experiment",
                "status",
                "seen_iou",
                "unseen_iou",
                "seen_dice",
                "unseen_dice",
                "seen_outside_ratio",
                "unseen_outside_ratio",
            ]
        ].to_string(
            index=False
        )
    )


def main():
    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Final output:",
        FINAL_ROOT,
    )

    print()
    print(
        "Validation model selection"
    )

    print(
        "--------------------------"
    )

    validation_table = (
        build_validation_table()
    )

    print_validation_table(
        validation_table
    )

    validation_output = (
        FINAL_ROOT
        / "validation_summary.csv"
    )

    validation_table.to_csv(
        validation_output,
        index=False,
    )

    print()
    print(
        "Saved:",
        validation_output,
    )


    available = validation_table[
        validation_table[
            "status"
        ] == "available"
    ]

    if len(available) > 0:
        best_index = (
            available[
                "val_iou"
            ]
            .astype(float)
            .idxmax()
        )

        best = available.loc[
            best_index
        ]

        print()
        print(
            "Best available model by "
            "validation IoU:"
        )

        print(
            best[
                "experiment"
            ]
        )

        print(
            "Validation IoU:",
            best[
                "val_iou"
            ],
        )


    print()
    print(
        "Seen / unseen test results"
    )

    print(
        "--------------------------"
    )

    test_table = build_test_table()


    if test_table is None:
        print(
            "Test result file not found:"
        )

        print(
            TEST_RESULTS_PATH
        )

        print()
        print(
            "Run:"
        )

        print(
            "python experiments/"
            "evaluate_models.py"
        )

        return


    print_test_table(
        test_table
    )


    test_output = (
        FINAL_ROOT
        / "test_summary.csv"
    )


    test_table.to_csv(
        test_output,
        index=False,
    )


    print()
    print(
        "Saved:",
        test_output,
    )


if __name__ == "__main__":
    main()