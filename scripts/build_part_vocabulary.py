from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "train.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "part_vocabulary.json"
)


def main() -> None:
    records = json.loads(
        TRAIN_MANIFEST.read_text(encoding="utf-8")
    )

    part_names = sorted(
        {
            record["part_name"]
            for record in records
        }
    )

    part_to_index = {
        part_name: index
        for index, part_name in enumerate(part_names)
    }

    metadata = {
        "part_names": part_names,
        "part_to_index": part_to_index,
        "number_of_parts": len(part_names),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved vocabulary:", OUTPUT_PATH)
    print("Number of generic part names:", len(part_names))

    print("\nPart vocabulary:")

    for part_name, index in part_to_index.items():
        print(f"{index:02d}: {part_name}")


if __name__ == "__main__":
    main()