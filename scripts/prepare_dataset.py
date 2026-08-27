from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metadata import (  # noqa: E402
    OBJECT_NAMES,
    PART_CATEGORIES,
    get_part_category,
)


IGNORE_LABEL = 255
VALID_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}

MANIFEST_NAMES = (
    "train",
    "validation",
    "test",
    "train_seen",
    "validation_seen",
    "test_seen",
    "test_unseen",
)

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


def visible_files_by_stem(
    directory: Path,
) -> dict[str, Path]:
    """Return valid files while ignoring hidden files."""

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Required directory not found: {directory}"
        )

    files: dict[str, Path] = {}

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        # Ignores .DS_Store and macOS ._* resource files.
        if path.name.startswith("."):
            continue

        if path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue

        if path.stem in files:
            raise RuntimeError(
                f"Duplicate filename stem '{path.stem}' "
                f"inside {directory}"
            )

        files[path.stem] = path

    return files


def project_relative(
    path: Path,
) -> str:
    """Convert a path to a project-relative path."""

    resolved_path = path.resolve()
    resolved_root = PROJECT_ROOT.resolve()

    try:
        return str(
            resolved_path.relative_to(resolved_root)
        )
    except ValueError:
        return str(resolved_path)


def load_label_map(
    path: Path,
) -> np.ndarray:
    """Load a semantic annotation as a 2-D array."""

    with Image.open(path) as annotation:
        labels = np.asarray(annotation)

    if labels.ndim != 2:
        raise ValueError(
            f"Expected a 2-D annotation at {path}, "
            f"received shape {labels.shape}"
        )

    return labels.astype(
        np.int64,
        copy=False,
    )


def validate_label_values(
    labels: np.ndarray,
    maximum_label: int,
    path: Path,
) -> None:
    """Check that annotation IDs are valid."""

    present_values = {
        int(value)
        for value in np.unique(labels)
        if int(value) != IGNORE_LABEL
    }

    invalid_values = sorted(
        value
        for value in present_values
        if value < 0 or value > maximum_label
    )

    if invalid_values:
        raise ValueError(
            f"Unexpected labels in {path}: "
            f"{invalid_values}"
        )


def records_for_image(
    image_id: str,
    source_split: str,
    image_path: Path,
    object_annotation_path: Path,
    part_annotation_path: Path,
) -> list[dict]:
    """Create one query record for every present part."""

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    object_labels = load_label_map(
        object_annotation_path
    )

    part_labels = load_label_map(
        part_annotation_path
    )

    expected_shape = (
        image_height,
        image_width,
    )

    if object_labels.shape != expected_shape:
        raise ValueError(
            f"Image and object annotation dimensions differ "
            f"for {image_id}: {expected_shape} and "
            f"{object_labels.shape}"
        )

    if part_labels.shape != expected_shape:
        raise ValueError(
            f"Image and part annotation dimensions differ "
            f"for {image_id}: {expected_shape} and "
            f"{part_labels.shape}"
        )

    validate_label_values(
        labels=object_labels,
        maximum_label=len(OBJECT_NAMES) - 1,
        path=object_annotation_path,
    )

    validate_label_values(
        labels=part_labels,
        maximum_label=len(PART_CATEGORIES) - 1,
        path=part_annotation_path,
    )

    present_part_ids = sorted(
        int(value)
        for value in np.unique(part_labels)
        if int(value) != IGNORE_LABEL
    )

    records: list[dict] = []

    for part_id in present_part_ids:
        category = get_part_category(part_id)

        object_id = int(
            category["object_id"]
        )

        object_mask = (
            object_labels == object_id
        )

        raw_part_mask = (
            part_labels == part_id
        )

        target_part_mask = (
            raw_part_mask
            & object_mask
        )

        object_pixels = int(
            object_mask.sum()
        )

        raw_part_pixels = int(
            raw_part_mask.sum()
        )

        target_part_pixels = int(
            target_part_mask.sum()
        )

        if object_pixels == 0:
            raise ValueError(
                f"Part {part_id} exists in {image_id}, "
                f"but parent object {object_id} is absent"
            )

        if target_part_pixels == 0:
            raise ValueError(
                f"Part {part_id} has no overlap with "
                f"parent object {object_id} in {image_id}"
            )

        containment_ratio = (
            target_part_pixels
            / max(raw_part_pixels, 1)
        )

        records.append(
            {
                "sample_id": (
                    f"{source_split}:"
                    f"{image_id}:"
                    f"part_{part_id}"
                ),
                "image_id": image_id,
                "source_split": source_split,
                "image_path": project_relative(
                    image_path
                ),
                "object_annotation_path": (
                    project_relative(
                        object_annotation_path
                    )
                ),
                "part_annotation_path": (
                    project_relative(
                        part_annotation_path
                    )
                ),
                "width": image_width,
                "height": image_height,
                "object_id": object_id,
                "object_name": (
                    category["object_name"]
                ),
                "part_id": part_id,
                "part_name": (
                    category["part_name"]
                ),
                "full_part_name": (
                    category["full_name"]
                ),
                "query": category["part_name"],
                "evaluation_split": (
                    category["evaluation_split"]
                ),
                "object_pixels": object_pixels,
                "raw_part_pixels": raw_part_pixels,
                "target_part_pixels": (
                    target_part_pixels
                ),
                "containment_ratio": (
                    containment_ratio
                ),
            }
        )

    return records


def collect_source_split(
    raw_root: Path,
    source_split: str,
) -> dict[str, list[dict]]:
    """Validate and collect one source split."""

    image_files = visible_files_by_stem(
        raw_root
        / "images"
        / source_split
    )

    object_files = visible_files_by_stem(
        raw_root
        / "annotations_detectron2_obj"
        / source_split
    )

    part_files = visible_files_by_stem(
        raw_root
        / "annotations_detectron2_part"
        / source_split
    )

    image_stems = set(image_files)
    object_stems = set(object_files)
    part_stems = set(part_files)

    matched_stems = sorted(
        image_stems
        & object_stems
        & part_stems
    )

    missing_object_masks = sorted(
        image_stems - object_stems
    )

    missing_part_masks = sorted(
        image_stems - part_stems
    )

    object_masks_without_images = sorted(
        object_stems - image_stems
    )

    part_masks_without_images = sorted(
        part_stems - image_stems
    )

    print()
    print(f"Source split: {source_split}")
    print("-" * 40)
    print(
        f"Images:                 "
        f"{len(image_files):6d}"
    )
    print(
        f"Object annotations:     "
        f"{len(object_files):6d}"
    )
    print(
        f"Part annotations:       "
        f"{len(part_files):6d}"
    )
    print(
        f"Matched triplets:       "
        f"{len(matched_stems):6d}"
    )

    if missing_object_masks:
        raise RuntimeError(
            f"{len(missing_object_masks)} images are "
            f"missing object annotations"
        )

    if missing_part_masks:
        raise RuntimeError(
            f"{len(missing_part_masks)} images are "
            f"missing part annotations"
        )

    if object_masks_without_images:
        raise RuntimeError(
            f"{len(object_masks_without_images)} object "
            f"annotations have no matching image"
        )

    if part_masks_without_images:
        raise RuntimeError(
            f"{len(part_masks_without_images)} part "
            f"annotations have no matching image"
        )

    if not matched_stems:
        raise RuntimeError(
            f"No matched data found for split "
            f"'{source_split}'"
        )

    records_by_image: dict[
        str,
        list[dict],
    ] = {}

    for index, image_id in enumerate(
        matched_stems,
        start=1,
    ):
        records_by_image[image_id] = (
            records_for_image(
                image_id=image_id,
                source_split=source_split,
                image_path=image_files[
                    image_id
                ],
                object_annotation_path=(
                    object_files[image_id]
                ),
                part_annotation_path=(
                    part_files[image_id]
                ),
            )
        )

        if (
            index % 500 == 0
            or index == len(matched_stems)
        ):
            print(
                f"Processed:              "
                f"{index:6d}/"
                f"{len(matched_stems)}"
            )

    return records_by_image


def flatten_records(
    records_by_image: dict[
        str,
        list[dict],
    ],
    image_ids: list[str],
) -> list[dict]:
    """Combine records from selected image IDs."""

    return [
        record
        for image_id in image_ids
        for record in records_by_image[
            image_id
        ]
    ]


def filter_evaluation_split(
    records: list[dict],
    evaluation_split: str,
) -> list[dict]:
    """Filter records by seen/unseen status."""

    return [
        record
        for record in records
        if record["evaluation_split"]
        == evaluation_split
    ]


def write_json(
    path: Path,
    value,
) -> None:
    """Write formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_split_file(
    path: Path,
    image_ids: list[str],
) -> None:
    """Write one image ID per line."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "".join(
            f"{image_id}\n"
            for image_id in image_ids
        ),
        encoding="utf-8",
    )


def summarize_records(
    name: str,
    records: list[dict],
) -> dict:
    """Calculate and print manifest statistics."""

    image_ids = {
        record["image_id"]
        for record in records
    }

    object_counts = Counter(
        record["object_name"]
        for record in records
    )

    part_counts = Counter(
        record["full_part_name"]
        for record in records
    )

    containment_values = [
        record["containment_ratio"]
        for record in records
    ]

    summary = {
        "images": len(image_ids),
        "queries": len(records),
        "object_categories": len(
            object_counts
        ),
        "part_categories": len(
            part_counts
        ),
        "minimum_containment_ratio": (
            min(containment_values)
            if containment_values
            else None
        ),
        "mean_containment_ratio": (
            float(
                np.mean(
                    containment_values
                )
            )
            if containment_values
            else None
        ),
        "object_query_counts": dict(
            sorted(
                object_counts.items()
            )
        ),
        "part_query_counts": dict(
            sorted(
                part_counts.items()
            )
        ),
    }

    print()
    print(name)
    print("-" * len(name))
    print(
        f"Images:             "
        f"{summary['images']:6d}"
    )
    print(
        f"Part queries:       "
        f"{summary['queries']:6d}"
    )
    print(
        f"Object categories:  "
        f"{summary['object_categories']:6d}"
    )
    print(
        f"Part categories:    "
        f"{summary['part_categories']:6d}"
    )

    return summary


def prepared_outputs_exist(
    output_dir: Path,
    splits_dir: Path,
    seed: int,
    validation_fraction: float,
    raw_root: Path,
) -> bool:
    """Check that all generated files exist and are valid."""

    manifest_paths = [
        output_dir / f"{name}.json"
        for name in MANIFEST_NAMES
    ]

    metadata_path = (
        output_dir / "metadata.json"
    )

    statistics_path = (
        output_dir
        / "dataset_statistics.json"
    )

    split_paths = [
        splits_dir / f"{name}.txt"
        for name in SPLIT_NAMES
    ]

    all_paths = (
        manifest_paths
        + [
            metadata_path,
            statistics_path,
        ]
        + split_paths
    )

    missing_paths = [
        path
        for path in all_paths
        if (
            not path.is_file()
            or path.stat().st_size == 0
        )
    ]

    if missing_paths:
        print(
            "Prepared dataset is incomplete."
        )

        for path in missing_paths:
            print("  Missing:", path)

        return False

    try:
        for manifest_path in manifest_paths:
            records = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                not isinstance(records, list)
                or not records
            ):
                print(
                    "Invalid or empty manifest:",
                    manifest_path,
                )
                return False

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        statistics = json.loads(
            statistics_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(metadata, dict):
            print(
                "Generated metadata is invalid."
            )
            return False

        if not isinstance(statistics, dict):
            print(
                "Generated statistics are invalid."
            )
            return False

        configuration = statistics.get(
            "configuration",
            {},
        )

        if (
            configuration.get("seed")
            != seed
        ):
            print(
                "Existing data uses a "
                "different random seed."
            )
            return False

        existing_fraction = (
            configuration.get(
                "validation_fraction"
            )
        )

        if (
            existing_fraction is None
            or not np.isclose(
                existing_fraction,
                validation_fraction,
            )
        ):
            print(
                "Existing data uses a "
                "different validation fraction."
            )
            return False

        expected_raw_root = (
            project_relative(raw_root)
        )

        if (
            configuration.get("raw_root")
            != expected_raw_root
        ):
            print(
                "Existing data uses a "
                "different raw dataset path."
            )
            return False

        split_sets: dict[
            str,
            set[str],
        ] = {}

        for split_name, split_path in zip(
            SPLIT_NAMES,
            split_paths,
        ):
            image_ids = {
                line.strip()
                for line in split_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }

            if not image_ids:
                print(
                    f"Empty split file: "
                    f"{split_path}"
                )
                return False

            split_sets[
                split_name
            ] = image_ids

        if not split_sets[
            "train"
        ].isdisjoint(
            split_sets["validation"]
        ):
            print(
                "Train and validation "
                "splits overlap."
            )
            return False

        if not split_sets[
            "train"
        ].isdisjoint(
            split_sets["test"]
        ):
            print(
                "Train and test "
                "splits overlap."
            )
            return False

        if not split_sets[
            "validation"
        ].isdisjoint(
            split_sets["test"]
        ):
            print(
                "Validation and test "
                "splits overlap."
            )
            return False

    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "Existing prepared data "
            "is invalid:",
            error,
        )
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Pascal-Part-116 and "
            "generate query-level manifests."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "raw"
            / "PascalPart116"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
        ),
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "splits"
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate all manifests even "
            "when valid outputs already exist."
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.validation_fraction
        < 1.0
    ):
        parser.error(
            "--validation-fraction must "
            "be between 0 and 1"
        )

    if not args.raw_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: "
            f"{args.raw_root.resolve()}"
        )

    if (
        not args.force
        and prepared_outputs_exist(
            output_dir=args.output_dir,
            splits_dir=args.splits_dir,
            seed=args.seed,
            validation_fraction=(
                args.validation_fraction
            ),
            raw_root=args.raw_root,
        )
    ):
        print()
        print("=" * 40)
        print(
            "Prepared dataset already exists."
        )
        print(
            "Generation was skipped."
        )
        print("=" * 40)
        print()
        print("Next command:")
        print("pytest -q")
        print()
        print("Then inspect a sample:")
        print(
            "python scripts/"
            "inspect_dataset.py "
            "--split train"
        )
        print()
        print(
            "Use --force to regenerate:"
        )
        print(
            "python scripts/"
            "prepare_dataset.py --force"
        )
        return

    print(
        "Pascal-Part-116 preparation"
    )
    print("=" * 40)
    print(
        "Project root:",
        PROJECT_ROOT.resolve(),
    )
    print(
        "Raw dataset:",
        args.raw_root.resolve(),
    )
    print(
        "Output directory:",
        args.output_dir.resolve(),
    )
    print("Random seed:", args.seed)
    print(
        "Validation fraction:",
        args.validation_fraction,
    )

    source_train_records = (
        collect_source_split(
            raw_root=args.raw_root,
            source_split="train",
        )
    )

    source_test_records = (
        collect_source_split(
            raw_root=args.raw_root,
            source_split="val",
        )
    )

    source_train_ids = sorted(
        source_train_records
    )

    shuffled_train_ids = (
        source_train_ids.copy()
    )

    random.Random(
        args.seed
    ).shuffle(shuffled_train_ids)

    validation_count = max(
        1,
        round(
            len(shuffled_train_ids)
            * args.validation_fraction
        ),
    )

    validation_ids = sorted(
        shuffled_train_ids[
            :validation_count
        ]
    )

    training_ids = sorted(
        shuffled_train_ids[
            validation_count:
        ]
    )

    test_ids = sorted(
        source_test_records
    )

    training_records = flatten_records(
        source_train_records,
        training_ids,
    )

    validation_records = flatten_records(
        source_train_records,
        validation_ids,
    )

    test_records = flatten_records(
        source_test_records,
        test_ids,
    )

    train_seen_records = (
        filter_evaluation_split(
            training_records,
            "seen",
        )
    )

    validation_seen_records = (
        filter_evaluation_split(
            validation_records,
            "seen",
        )
    )

    test_seen_records = (
        filter_evaluation_split(
            test_records,
            "seen",
        )
    )

    test_unseen_records = (
        filter_evaluation_split(
            test_records,
            "unseen",
        )
    )

    manifests = {
        "train": training_records,
        "validation": (
            validation_records
        ),
        "test": test_records,
        "train_seen": (
            train_seen_records
        ),
        "validation_seen": (
            validation_seen_records
        ),
        "test_seen": (
            test_seen_records
        ),
        "test_unseen": (
            test_unseen_records
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.splits_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for manifest_name, records in (
        manifests.items()
    ):
        manifest_path = (
            args.output_dir
            / f"{manifest_name}.json"
        )

        write_json(
            manifest_path,
            records,
        )

        print(
            f"Saved {manifest_name}: "
            f"{manifest_path}"
        )

    write_split_file(
        args.splits_dir / "train.txt",
        training_ids,
    )

    write_split_file(
        args.splits_dir
        / "validation.txt",
        validation_ids,
    )

    write_split_file(
        args.splits_dir / "test.txt",
        test_ids,
    )

    category_metadata = {
        "ignore_label": IGNORE_LABEL,
        "object_names": (
            OBJECT_NAMES
        ),
        "part_categories": (
            PART_CATEGORIES
        ),
    }

    write_json(
        args.output_dir
        / "metadata.json",
        category_metadata,
    )

    statistics = {
        manifest_name: (
            summarize_records(
                manifest_name,
                records,
            )
        )
        for manifest_name, records
        in manifests.items()
    }

    statistics["configuration"] = {
        "seed": args.seed,
        "validation_fraction": (
            args.validation_fraction
        ),
        "raw_root": project_relative(
            args.raw_root
        ),
        "ignore_label": (
            IGNORE_LABEL
        ),
    }

    write_json(
        args.output_dir
        / "dataset_statistics.json",
        statistics,
    )

    training_set = set(
        training_ids
    )

    validation_set = set(
        validation_ids
    )

    test_set = set(
        test_ids
    )

    if not training_set.isdisjoint(
        validation_set
    ):
        raise RuntimeError(
            "Training and validation "
            "image IDs overlap"
        )

    if not training_set.isdisjoint(
        test_set
    ):
        raise RuntimeError(
            "Training and test "
            "image IDs overlap"
        )

    if not validation_set.isdisjoint(
        test_set
    ):
        raise RuntimeError(
            "Validation and test "
            "image IDs overlap"
        )

    print()
    print("=" * 40)
    print(
        "Dataset preparation "
        "completed successfully."
    )
    print("=" * 40)
    print()
    print("Next command:")
    print(
        "python scripts/"
        "inspect_dataset.py "
        "--split train"
    )


if __name__ == "__main__":
    main()