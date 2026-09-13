from pathlib import Path
from collections import defaultdict
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

from dashboard.utils.models import (
    load_seen_dataset,
    load_unseen_dataset,
)


# ============================================================
# INSPECT ONE SPLIT
# ============================================================

def inspect_split(
    name,
    dataset,
):

    parents = defaultdict(
        lambda: {
            "parts": set(),
            "samples": 0,
        }
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


        parents[
            parent
        ][
            "parts"
        ].add(
            part
        )


        parents[
            parent
        ][
            "samples"
        ] += 1


    # ========================================================
    # PRINT SPLIT SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
    )

    print(
        "Dataset samples:",
        len(dataset),
    )

    print(
        "Unique parents:",
        len(parents),
    )

    print()


    # ========================================================
    # PRINT EACH PARENT
    # ========================================================

    for parent in sorted(
        parents
    ):

        parts = sorted(
            parents[
                parent
            ][
                "parts"
            ]
        )

        count = (
            parents[
                parent
            ][
                "samples"
            ]
        )


        print(
            f"{parent:15s} | "
            f"{len(parts):2d} parts | "
            f"{count:4d} samples"
        )

        print(
            "   ",
            ", ".join(
                parts
            ),
        )


    # ========================================================
    # PARENT WITH MOST PARTS
    # ========================================================

    max_parent = max(
        parents,
        key=lambda parent: len(
            parents[
                parent
            ][
                "parts"
            ]
        ),
    )


    print()

    print(
        "Maximum parts:",
        max_parent,
        "->",
        len(
            parents[
                max_parent
            ][
                "parts"
            ]
        ),
    )


    return parents


# ============================================================
# LOAD DATASETS
# ============================================================

seen_dataset = (
    load_seen_dataset()
)

unseen_dataset = (
    load_unseen_dataset()
)


# ============================================================
# INSPECT
# ============================================================

seen = inspect_split(
    "TEST SEEN",
    seen_dataset,
)

unseen = inspect_split(
    "TEST UNSEEN",
    unseen_dataset,
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

all_parents = (
    set(
        seen
    )
    | set(
        unseen
    )
)


print()
print(
    "=" * 70
)

print(
    "COMBINED"
)

print(
    "=" * 70
)


print(
    "Seen parents:",
    len(
        seen
    ),
)

print(
    "Unseen parents:",
    len(
        unseen
    ),
)

print(
    "Total unique parents:",
    len(
        all_parents
    ),
)


print(
    "Total seen samples:",
    sum(
        value[
            "samples"
        ]
        for value in seen.values()
    ),
)


print(
    "Total unseen samples:",
    sum(
        value[
            "samples"
        ]
        for value in unseen.values()
    ),
)