from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

DASHBOARD_DIR = OUTPUT_ROOT / "dashboard"

DASHBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPERIMENTS = [
    # Baselines
    (
        "baseline",
        "part_only",
        OUTPUT_ROOT
        / "experiments"
        / "part_only"
        / "history.json",
    ),
    (
        "baseline",
        "object_mask",
        OUTPUT_ROOT
        / "experiments"
        / "object_mask"
        / "history.json",
    ),

    # Geometry
    (
        "geometry",
        "object_mask",
        OUTPUT_ROOT
        / "geometry"
        / "object_mask"
        / "history.json",
    ),
    (
        "geometry",
        "absolute_xy",
        OUTPUT_ROOT
        / "geometry"
        / "absolute_xy"
        / "history.json",
    ),
    (
        "geometry",
        "relative_uv",
        OUTPUT_ROOT
        / "geometry"
        / "relative_uv"
        / "history.json",
    ),

    # Alignment
    (
        "alignment",
        "mask_baseline",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "mask_baseline"
        / "history.json",
    ),
    (
        "alignment",
        "alignment_mask",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "alignment_mask"
        / "history.json",
    ),
    (
        "alignment",
        "alignment_relative_uv",
        OUTPUT_ROOT
        / "experiments"
        / "part_query_alignment"
        / "alignment_relative_uv"
        / "history.json",
    ),

    # Full-image UVD
    (
        "uvd",
        "fixed_uvd",
        OUTPUT_ROOT
        / "experiments"
        / "query_gated_uvd"
        / "fixed_uvd"
        / "history.json",
    ),
    (
        "uvd",
        "query_gated_uvd",
        OUTPUT_ROOT
        / "experiments"
        / "query_gated_uvd"
        / "query_gated_uvd"
        / "history.json",
    ),

    # Object-centric crop
    (
        "object_zoom",
        "mask_baseline",
        OUTPUT_ROOT
        / "object_zoom"
        / "mask_baseline"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_mask",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_mask"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_relative_uv",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_relative_uv"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_fixed_uvd",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_fixed_uvd"
        / "history.json",
    ),
    (
        "object_zoom",
        "alignment_query_gated_uvd",
        OUTPUT_ROOT
        / "object_zoom"
        / "alignment_query_gated_uvd"
        / "history.json",
    ),
]


summary_rows = []
epoch_rows = []


for family, mode, history_path in EXPERIMENTS:

    if not history_path.is_file():
        print(
            "Missing:",
            history_path,
        )
        continue

    history = json.loads(
        history_path.read_text()
    )

    if not history:
        print(
            "Empty history:",
            history_path,
        )
        continue

    best = max(
        history,
        key=lambda row: row["val_iou"],
    )

    runtime_seconds = sum(
        float(
            row.get(
                "seconds",
                0.0,
            )
        )
        for row in history
    )

    summary_rows.append(
        {
            "family":
                family,

            "mode":
                mode,

            "best_epoch":
                best["epoch"],

            "best_val_iou":
                best["val_iou"],

            "best_val_dice":
                best.get(
                    "val_dice"
                ),

            "best_val_loss":
                best.get(
                    "val_loss"
                ),

            "train_iou_at_best":
                best.get(
                    "train_iou"
                ),

            "train_loss_at_best":
                best.get(
                    "train_loss"
                ),

            "runtime_seconds":
                runtime_seconds,

            "epochs":
                len(history),
        }
    )

    for row in history:

        epoch_row = {
            "family":
                family,

            "mode":
                mode,
        }

        epoch_row.update(
            row
        )

        epoch_rows.append(
            epoch_row
        )


def write_csv(
    path,
    rows,
):
    if not rows:
        print(
            "No rows for:",
            path,
        )
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


summary_path = (
    DASHBOARD_DIR
    / "experiment_summary.csv"
)

history_path = (
    DASHBOARD_DIR
    / "epoch_history.csv"
)


write_csv(
    summary_path,
    summary_rows,
)

write_csv(
    history_path,
    epoch_rows,
)


# ============================================================
# Final seen / unseen evaluation
# ============================================================

test_rows = []


# Existing selected alignment-mask evaluation
alignment_mask_path = (
    OUTPUT_ROOT
    / "object_zoom_evaluation"
    / "object_zoom_results.json"
)

if alignment_mask_path.is_file():

    raw = json.loads(
        alignment_mask_path.read_text()
    )

    for row in raw:

        crop = row.get(
            "object_crop",
            {}
        )

        test_rows.append(
            {
                "model":
                    "Object Zoom + Alignment Mask",

                "mode":
                    "alignment_mask",

                "split":
                    row["split"],

                "iou":
                    crop.get("iou"),

                "dice":
                    crop.get("dice"),
            }
        )


# New crop-UVD evaluations
uvd_test_files = [
    (
        "Object Zoom + Alignment Fixed UVD",
        "alignment_fixed_uvd",
        OUTPUT_ROOT
        / "object_zoom_evaluation"
        / "alignment_fixed_uvd_results.json",
    ),

    (
        "Object Zoom + Alignment Query-Gated UVD",
        "alignment_query_gated_uvd",
        OUTPUT_ROOT
        / "object_zoom_evaluation"
        / "alignment_query_gated_uvd_results.json",
    ),
]


for display_name, mode, path in uvd_test_files:

    if not path.is_file():
        print(
            "Missing final test result:",
            path,
        )
        continue

    raw = json.loads(
        path.read_text()
    )

    for row in raw[
        "results"
    ]:

        test_rows.append(
            {
                "model":
                    display_name,

                "mode":
                    mode,

                "split":
                    row["split"],

                "iou":
                    row["iou"],

                "dice":
                    row["dice"],
            }
        )


test_output = (
    DASHBOARD_DIR
    / "final_test_results.csv"
)


write_csv(
    test_output,
    test_rows,
)


print()
print("Created:")
print(summary_path)
print(history_path)

if test_output.is_file():
    print(test_output)
