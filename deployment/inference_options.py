"""Shared object and part-query options used by the API and web client."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONS_PATH = PROJECT_ROOT / "web" / "data" / "inference_options.json"


def _load_options() -> dict:
    with OPTIONS_PATH.open(encoding="utf-8") as handle:
        options = json.load(handle)

    required = {
        "parent_categories",
        "part_queries",
        "part_suggestion_groups",
        "parent_part_groups",
    }
    missing = required.difference(options)
    if missing:
        raise RuntimeError(
            f"Inference options are missing: {', '.join(sorted(missing))}"
        )
    return options


OPTIONS = _load_options()
PARENT_CATEGORIES = tuple(OPTIONS["parent_categories"])
PART_QUERIES = tuple(OPTIONS["part_queries"])
PART_SUGGESTION_GROUPS = {
    name: tuple(parts)
    for name, parts in OPTIONS["part_suggestion_groups"].items()
}
PARENT_PART_GROUPS = dict(OPTIONS["parent_part_groups"])


def suggested_parts_for_parent(
    category: str,
    evaluated_parts: dict[str, set[str]],
) -> tuple[str, ...]:
    """Return exact evaluated parts or the curated unseen-category suggestions."""
    if category in evaluated_parts:
        return tuple(sorted(evaluated_parts[category]))

    group_name = PARENT_PART_GROUPS.get(category)
    return PART_SUGGESTION_GROUPS.get(group_name, ())
