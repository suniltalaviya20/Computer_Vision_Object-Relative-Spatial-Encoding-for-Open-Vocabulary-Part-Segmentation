from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "pascal_part"
    / "register_pascal_part_116.py"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "pascal_part_116_categories.json"
)


def extract_literal_assignment(
    tree: ast.Module,
    variable_name: str,
) -> Any:
    """Extract a literal variable assignment from a Python source file."""

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == variable_name
            ):
                return ast.literal_eval(node.value)

    raise KeyError(
        f"Could not find literal assignment for {variable_name}"
    )


def split_object_part_name(full_name: str) -> tuple[str, str]:
    """Split a name such as bicycle's wheel into two components."""

    separator = "'s "

    if separator not in full_name:
        raise ValueError(
            f"Unexpected object-part class name: {full_name}"
        )

    object_name, part_name = full_name.split(separator, maxsplit=1)

    return object_name, part_name


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Official metadata source was not found: {SOURCE_PATH}"
        )

    source_code = SOURCE_PATH.read_text(encoding="utf-8")
    syntax_tree = ast.parse(source_code)

    object_class_names = extract_literal_assignment(
        syntax_tree,
        "OBJ_CLASS_NAMES",
    )

    object_part_class_names = extract_literal_assignment(
        syntax_tree,
        "CLASS_NAMES",
    )

    # Official generalized zero-shot split used by OV-PARTS.
    unseen_object_names = {
        "bird",
        "car",
        "dog",
        "sheep",
        "motorbike",
    }

    object_records = []

    for class_id, object_name in enumerate(object_class_names):
        split = (
            "unseen"
            if object_name in unseen_object_names
            else "seen"
        )

        object_records.append(
            {
                "id": class_id,
                "name": object_name,
                "split": split,
            }
        )

    object_part_records = []
    generic_part_names = set()

    for class_id, full_name in enumerate(object_part_class_names):
        object_name, part_name = split_object_part_name(full_name)

        generic_part_names.add(part_name)

        split = (
            "unseen"
            if object_name in unseen_object_names
            else "seen"
        )

        object_part_records.append(
            {
                "id": class_id,
                "full_name": full_name,
                "object_name": object_name,
                "part_name": part_name,
                "split": split,
            }
        )

    if len(object_records) != 20:
        raise RuntimeError(
            f"Expected 20 object classes, found {len(object_records)}"
        )

    if len(object_part_records) != 116:
        raise RuntimeError(
            "Expected 116 object-part classes, "
            f"found {len(object_part_records)}"
        )

    metadata = {
        "ignore_label": 255,
        "objects": object_records,
        "object_parts": object_part_records,
        "generic_part_names": sorted(generic_part_names),
        "unseen_objects": sorted(unseen_object_names),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved metadata:", OUTPUT_PATH)
    print("Object classes:", len(object_records))
    print("Object-part classes:", len(object_part_records))
    print("Generic part names:", len(generic_part_names))
    print("Unseen objects:", sorted(unseen_object_names))

    print("\nFirst five object classes:")

    for record in object_records[:5]:
        print(record)

    print("\nFirst five object-part classes:")

    for record in object_part_records[:5]:
        print(record)


if __name__ == "__main__":
    main()