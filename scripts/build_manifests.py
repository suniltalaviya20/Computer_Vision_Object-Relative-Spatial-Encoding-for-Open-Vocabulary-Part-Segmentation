from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Pascal-Part-116"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pascal_part_116_categories.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"

IGNORE_LABEL = 255
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def valid_files(directory: Path) -> list[Path]:
    """Return valid non-hidden image files."""

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VALID_IMAGE_SUFFIXES
            and not path.name.startswith(".")
            and not path.name.startswith("._")
        )
    )


def build_stem_map(directory: Path) -> dict[str, Path]:
    """Create a mapping from filename stem to path."""

    return {
        path.stem: path
        for path in valid_files(directory)
    }


def load_label_map(path: Path) -> np.ndarray:
    """Load a single-channel integer semantic label map."""

    image = Image.open(path)
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D label map at {path}, "
            f"but received shape {array.shape}."
        )

    return array.astype(np.int64)


def relative_path(path: Path) -> str:
    """Store paths relative to the project root."""

    return str(path.relative_to(PROJECT_ROOT))


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {METADATA_PATH}"
        )

    return json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )


def create_records_for_stem(
    stem: str,
    image_path: Path,
    object_annotation_path: Path,
    part_annotation_path: Path,
    metadata: dict[str, Any],
    source_split: str,
) -> list[dict[str, Any]]:
    """Create one record for every annotated part in an image."""

    object_labels = load_label_map(object_annotation_path)
    part_labels = load_label_map(part_annotation_path)

    if object_labels.shape != part_labels.shape:
        raise ValueError(
            f"Object and part annotations have different shapes "
            f"for {stem}: {object_labels.shape} and {part_labels.shape}"
        )

    object_records = metadata["objects"]
    part_records = metadata["object_parts"]

    object_name_to_id = {
        record["name"]: int(record["id"])
        for record in object_records
    }

    present_part_ids = [
        int(value)
        for value in np.unique(part_labels)
        if int(value) != IGNORE_LABEL
    ]

    records: list[dict[str, Any]] = []

    for part_id in present_part_ids:
        if not 0 <= part_id < len(part_records):
            print(
                f"Warning: skipping unexpected part label "
                f"{part_id} in {part_annotation_path}"
            )
            continue

        part_record = part_records[part_id]

        object_name = part_record["object_name"]
        part_name = part_record["part_name"]
        full_part_name = part_record["full_name"]
        evaluation_split = part_record["split"]

        object_id = object_name_to_id[object_name]

        object_mask = object_labels == object_id
        part_mask = part_labels == part_id

        object_pixels = int(object_mask.sum())
        part_pixels = int(part_mask.sum())

        # The part must overlap its corresponding parent-object class.
        overlap_pixels = int(
            np.count_nonzero(object_mask & part_mask)
        )

        if object_pixels == 0:
            print(
                f"Warning: {full_part_name} exists in {stem}, "
                f"but parent object {object_name} is absent."
            )
            continue

        if part_pixels == 0 or overlap_pixels == 0:
            continue

        containment_ratio = overlap_pixels / max(part_pixels, 1)

        records.append(
            {
                "sample_id": f"{source_split}:{stem}:part_{part_id}",
                "image_stem": stem,
                "source_split": source_split,
                "image_path": relative_path(image_path),
                "object_annotation_path": relative_path(
                    object_annotation_path
                ),
                "part_annotation_path": relative_path(
                    part_annotation_path
                ),
                "object_id": object_id,
                "object_name": object_name,
                "part_id": part_id,
                "part_name": part_name,
                "full_part_name": full_part_name,
                "evaluation_split": evaluation_split,
                "object_pixels": object_pixels,
                "part_pixels": part_pixels,
                "overlap_pixels": overlap_pixels,
                "containment_ratio": containment_ratio,
            }
        )

    return records


def collect_split_records(
    split_name: str,
    metadata: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Collect records, grouped by image stem."""

    image_dir = DATASET_ROOT / "images" / split_name

    object_dir = (
        DATASET_ROOT
        / "annotations_detectron2_obj"
        / split_name
    )

    part_dir = (
        DATASET_ROOT
        / "annotations_detectron2_part"
        / split_name
    )

    image_map = build_stem_map(image_dir)
    object_map = build_stem_map(object_dir)
    part_map = build_stem_map(part_dir)

    common_stems = sorted(
        set(image_map)
        & set(object_map)
        & set(part_map)
    )

    missing_objects = sorted(
        set(image_map) - set(object_map)
    )

    missing_parts = sorted(
        set(image_map) - set(part_map)
    )

    print(f"\nSource split: {split_name}")
    print("Images:", len(image_map))
    print("Object annotations:", len(object_map))
    print("Part annotations:", len(part_map))
    print("Complete matched samples:", len(common_stems))
    print("Images missing object annotations:", len(missing_objects))
    print("Images missing part annotations:", len(missing_parts))

    records_by_stem: dict[str, list[dict[str, Any]]] = {}

    for index, stem in enumerate(common_stems, start=1):
        records = create_records_for_stem(
            stem=stem,
            image_path=image_map[stem],
            object_annotation_path=object_map[stem],
            part_annotation_path=part_map[stem],
            metadata=metadata,
            source_split=split_name,
        )

        if records:
            records_by_stem[stem] = records

        if index % 500 == 0:
            print(
                f"Processed {index}/{len(common_stems)} images"
            )

    return records_by_stem


def flatten_records(
    records_by_stem: dict[str, list[dict[str, Any]]],
    stems: list[str],
) -> list[dict[str, Any]]:
    """Combine all records belonging to selected image stems."""

    return [
        record
        for stem in stems
        for record in records_by_stem.get(stem, [])
    ]


def filter_by_evaluation_split(
    records: list[dict[str, Any]],
    evaluation_split: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["evaluation_split"] == evaluation_split
    ]


def write_manifest(
    filename: str,
    records: list[dict[str, Any]],
) -> None:
    output_path = OUTPUT_DIR / filename

    output_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved {len(records)} records to {output_path}")


def print_statistics(
    name: str,
    records: list[dict[str, Any]],
) -> None:
    object_counts = Counter(
        record["object_name"]
        for record in records
    )

    part_counts = Counter(
        record["part_name"]
        for record in records
    )

    image_count = len(
        {
            record["image_stem"]
            for record in records
        }
    )

    print(f"\n{name}")
    print("-" * len(name))
    print("Records:", len(records))
    print("Unique images:", image_count)
    print("Object categories:", len(object_counts))
    print("Generic part names:", len(part_counts))

    print("Most common objects:")
    for object_name, count in object_counts.most_common(10):
        print(f"  {object_name}: {count}")

    print("Most common parts:")
    for part_name, count in part_counts.most_common(10):
        print(f"  {part_name}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of official training images reserved "
            "for validation."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError(
            "--validation-fraction must be between 0 and 1."
        )

    metadata = load_metadata()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_by_stem = collect_split_records(
        split_name="train",
        metadata=metadata,
    )

    val_by_stem = collect_split_records(
        split_name="val",
        metadata=metadata,
    )

    # Only seen parent-object categories are used for training.
    seen_train_by_stem = {
        stem: filter_by_evaluation_split(records, "seen")
        for stem, records in train_by_stem.items()
    }

    seen_train_by_stem = {
        stem: records
        for stem, records in seen_train_by_stem.items()
        if records
    }

    stems = sorted(seen_train_by_stem)

    random_generator = random.Random(args.seed)
    random_generator.shuffle(stems)

    validation_size = max(
        1,
        round(len(stems) * args.validation_fraction),
    )

    validation_stems = sorted(stems[:validation_size])
    training_stems = sorted(stems[validation_size:])

    train_records = flatten_records(
        seen_train_by_stem,
        training_stems,
    )

    validation_records = flatten_records(
        seen_train_by_stem,
        validation_stems,
    )

    official_val_records = flatten_records(
        val_by_stem,
        sorted(val_by_stem),
    )

    test_seen_records = filter_by_evaluation_split(
        official_val_records,
        "seen",
    )

    test_unseen_records = filter_by_evaluation_split(
        official_val_records,
        "unseen",
    )

    write_manifest("train.json", train_records)
    write_manifest("validation.json", validation_records)
    write_manifest("test_seen.json", test_seen_records)
    write_manifest("test_unseen.json", test_unseen_records)

    split_info = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "training_image_stems": training_stems,
        "validation_image_stems": validation_stems,
        "unseen_parent_objects": metadata["unseen_objects"],
    }

    write_manifest(
        "split_info.json",
        [split_info],
    )

    print_statistics(
        "Training set",
        train_records,
    )

    print_statistics(
        "Validation set",
        validation_records,
    )

    print_statistics(
        "Seen-object test set",
        test_seen_records,
    )

    print_statistics(
        "Unseen-object test set",
        test_unseen_records,
    )


if __name__ == "__main__":
    main()