"""Regenerate the static web catalogue from deployed final-study checkpoints.

Generation always happens in training_results_corrected/web_export first. Pass
--publish to copy the validated catalogue and assets into web/ afterwards.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import PascalPart116Dataset
from final_model.inference import load_predictor
from final_model.training_core import (
    preprocess_image,
    preprocess_mask,
    relative_uvd,
    resize_info,
)


WEB_ROOT = PROJECT_ROOT / "web"
TEMPLATE_PATH = WEB_ROOT / "data" / "catalogue.json"
REGISTRY_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_study"
    / "model_registry.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "training_results_corrected"
    / "web_export"
)
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--parts-per-parent",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--examples-per-part",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="For example cuda:0 or cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    publishing = parser.add_mutually_exclusive_group()

    publishing.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Generate, validate, and then copy the export "
            "into web/."
        ),
    )

    publishing.add_argument(
        "--publish-staged",
        action="store_true",
        help=(
            "Publish an existing validated --output export "
            "without rerunning inference."
        ),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return value


def model_specs(
    template: dict[str, Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    frontend = template.get("models")
    registered = registry.get("models")

    if (
        not isinstance(frontend, list)
        or not isinstance(registered, dict)
    ):
        raise ValueError(
            "Invalid catalogue template or model registry"
        )

    specs = []
    frontend_names = set()

    for item in frontend:
        model_id = str(item["id"])
        name = model_id.removeprefix("final_")

        frontend_names.add(name)

        if name not in registered:
            raise ValueError(
                "Frontend model is absent from registry: "
                f"{name}"
            )

        checkpoint = (
            PROJECT_ROOT
            / registered[name]["checkpoint"]
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint is missing: {checkpoint}"
            )

        specs.append(
            {
                **item,
                "name": name,
                "checkpoint": checkpoint,
            }
        )

    extra = set(registered) - frontend_names

    if extra:
        raise ValueError(
            "Registry models are absent from the frontend "
            f"catalogue: {sorted(extra)}"
        )

    return specs


def select_samples(
    dataset: PascalPart116Dataset,
    split: str,
    parts_per_parent: int,
    examples_per_part: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[
        str,
        dict[str, list[int]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for index, record in enumerate(dataset.records):
        groups[
            str(record["object_name"])
        ][
            str(record["part_name"])
        ].append(index)

    rng = random.Random(seed)
    selected = []

    for parent in sorted(groups):
        part_groups = groups[parent]

        ranked_parts = sorted(
            part_groups,
            key=lambda part: (
                -len(part_groups[part]),
                part,
            ),
        )

        for part in ranked_parts[:parts_per_parent]:
            indices = part_groups[part]

            count = min(
                examples_per_part,
                len(indices),
            )

            if len(indices) <= count:
                chosen = indices
            else:
                chosen = rng.sample(
                    indices,
                    count,
                )

            for index in sorted(chosen):
                selected.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "index": index,
                        "parent": parent,
                        "part": part,
                    }
                )

    return selected


def sample_folder(sample_id: str) -> str:
    return sample_id.replace(":", "__")


def image_array(
    image: torch.Tensor,
) -> np.ndarray:
    return (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        .astype(np.uint8)
    )


def gray_array(
    value: torch.Tensor,
) -> np.ndarray:
    array = (
        value.detach()
        .float()
        .cpu()
        .squeeze()
        .numpy()
    )

    return (
        np.clip(array, 0, 1) * 255
    ).round().astype(np.uint8)


def binary_array(
    value: torch.Tensor,
) -> np.ndarray:
    return (
        (
            value.detach()
            .cpu()
            .squeeze()
            .numpy()
            > 0.5
        ).astype(np.uint8)
        * 255
    )


def save_rgb(
    image: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image_array(image),
        "RGB",
    ).save(path)


def save_gray(
    value: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        gray_array(value),
        "L",
    ).save(path)


def save_binary(
    value: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        binary_array(value),
        "L",
    ).save(path)


def save_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    path: Path,
    colour: np.ndarray,
) -> None:
    base = image_array(image).astype(
        np.float32
    )

    active = (
        mask.detach()
        .cpu()
        .squeeze()
        .numpy()
        > 0.5
    )

    base[active] = (
        0.42 * base[active]
        + 0.58 * colour
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        np.clip(
            base,
            0,
            255,
        ).astype(np.uint8),
        "RGB",
    ).save(path)


def restore_map(
    value: torch.Tensor,
    height: int,
    width: int,
    image_size: int,
) -> torch.Tensor:
    info = resize_info(
        height,
        width,
        image_size,
    )

    top = int(info["top"])
    left = int(info["left"])
    new_h = int(info["new_h"])
    new_w = int(info["new_w"])

    cropped = value[
        :,
        :,
        top : top + new_h,
        left : left + new_w,
    ]

    return F.interpolate(
        cropped,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu()


@torch.inference_mode()
def predict_details(
    predictor: Any,
    sample: dict[str, Any],
) -> dict[str, Any]:
    image = sample["image"]
    parent = sample["object_mask"]

    height, width = image.shape[-2:]

    _, model_image = preprocess_image(
        image.cpu(),
        predictor.config.image_size,
    )

    model_mask = preprocess_mask(
        parent.cpu(),
        height,
        width,
        predictor.config.image_size,
    ).float()

    u, v, d = relative_uvd(
        model_mask.bool()
    )

    values = [
        tensor.unsqueeze(0).to(
            predictor.device
        )
        for tensor in (
            model_image,
            model_mask,
            u,
            v,
            d,
        )
    ]

    context = (
        torch.autocast(
            "cuda",
            dtype=torch.float16,
        )
        if predictor.device.type == "cuda"
        else nullcontext()
    )

    with context:
        logits, auxiliary = predictor.model(
            values[0],
            predictor.text(
                [sample["query"]]
            ),
            values[1],
            values[2],
            values[3],
            values[4],
        )

    probability = restore_map(
        torch.sigmoid(logits).float(),
        height,
        width,
        predictor.config.image_size,
    )

    geometry = [
        restore_map(
            value.unsqueeze(0),
            height,
            width,
            predictor.config.image_size,
        )
        for value in (u, v, d)
    ]

    return {
        "probability": probability,
        "prediction": (
            probability >= THRESHOLD
        ),
        "u": geometry[0],
        "v": geometry[1],
        "d": geometry[2],
        "gates": (
            auxiliary["effective_gates"][0]
            .float()
            .cpu()
        ),
    }


def metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    parent: torch.Tensor,
) -> dict[str, float]:
    predicted = prediction.bool().cpu()
    target = truth.bool().cpu()
    parent = parent.bool().cpu()

    intersection = int(
        (predicted & target).sum()
    )

    union = int(
        (predicted | target).sum()
    )

    predicted_count = int(
        predicted.sum()
    )

    target_count = int(
        target.sum()
    )

    return {
        "iou": (
            intersection
            / max(union, 1)
        ),
        "dice": (
            2 * intersection
            / max(
                predicted_count
                + target_count,
                1,
            )
        ),
        "leakage": (
            int(
                (
                    predicted
                    & ~parent
                ).sum()
            )
            / max(
                predicted_count,
                1,
            )
        ),
    }


def common_entry(
    item: dict[str, Any],
    assets_root: Path,
) -> dict[str, Any]:
    sample = item["dataset"][
        item["index"]
    ]

    folder = sample_folder(
        str(sample["sample_id"])
    )

    directory = assets_root / folder

    save_rgb(
        sample["image"],
        directory / "original.png",
    )

    save_binary(
        sample["object_mask"],
        directory / "parent.png",
    )

    save_binary(
        sample["part_mask"],
        directory / "ground_truth.png",
    )

    save_overlay(
        sample["image"],
        sample["part_mask"],
        directory
        / "ground_truth_overlay.png",
        np.array(
            [34, 197, 94],
            dtype=np.float32,
        ),
    )

    prefix = (
        f"assets/examples/{folder}"
    )

    return {
        "id": str(sample["sample_id"]),
        "object": str(
            sample["object_name"]
        ),
        "part": str(
            sample["part_name"]
        ),
        "query": str(
            sample["query"]
        ),
        "split": item["split"],
        "assets": {
            "original": {
                "file": (
                    f"{prefix}/original.png"
                ),
                "space": "full_image",
            },
            "parent": {
                "file": (
                    f"{prefix}/parent.png"
                ),
                "space": "full_image",
            },
            "ground_truth": {
                "file": (
                    f"{prefix}/ground_truth.png"
                ),
                "space": "full_image",
            },
            "ground_truth_overlay": {
                "file": (
                    f"{prefix}/"
                    "ground_truth_overlay.png"
                ),
                "space": "full_image",
                "description": (
                    "Dataset ground-truth "
                    "part overlaid in green."
                ),
            },
        },
        "results": {},
    }


def export_model(
    spec: dict[str, Any],
    selections: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    assets_root: Path,
    device: str | None,
) -> None:
    print(
        f"Loading {spec['label']}: "
        f"{spec['checkpoint']}"
    )

    predictor = load_predictor(
        spec["checkpoint"],
        device=device,
    )

    model_id = str(spec["id"])
    geometry = set(
        spec.get("geometry", [])
    )

    for position, (
        item,
        entry,
    ) in enumerate(
        zip(selections, entries),
        start=1,
    ):
        sample = item["dataset"][
            item["index"]
        ]

        output = predict_details(
            predictor,
            sample,
        )

        folder = sample_folder(
            str(sample["sample_id"])
        )

        directory = (
            assets_root
            / folder
            / model_id
        )

        save_overlay(
            sample["image"],
            output["prediction"],
            directory
            / "prediction.png",
            np.array(
                [239, 68, 68],
                dtype=np.float32,
            ),
        )

        save_gray(
            output["probability"],
            directory
            / "probability.png",
        )

        prefix = (
            f"assets/examples/"
            f"{folder}/{model_id}"
        )

        result_assets: dict[
            str,
            Any,
        ] = {
            "prediction": {
                "file": (
                    f"{prefix}/"
                    "prediction.png"
                ),
                "space": "full_image",
                "description": (
                    "Predicted part "
                    "foreground overlaid "
                    "in red."
                ),
            },
            "probability": {
                "file": (
                    f"{prefix}/"
                    "probability.png"
                ),
                "space": "full_image",
                "description": (
                    "Projected part "
                    "probability."
                ),
            },
        }

        descriptions = {
            "u": (
                "Horizontal position "
                "relative to the parent "
                "object."
            ),
            "v": (
                "Vertical position "
                "relative to the parent "
                "object."
            ),
            "d": (
                "2D distance to the "
                "parent-mask boundary."
            ),
        }

        for key in ("u", "v", "d"):
            if key not in geometry:
                continue

            save_gray(
                output[key],
                directory
                / f"{key}.png",
            )

            result_assets[key] = {
                "file": (
                    f"{prefix}/{key}.png"
                ),
                "space": "full_image",
                "description": (
                    descriptions[key]
                ),
            }

        result: dict[str, Any] = {
            "threshold": THRESHOLD,
            "checkpoint_id": str(
                spec["checkpoint"]
                .relative_to(
                    PROJECT_ROOT
                )
            ),
            "metrics": metrics(
                output["prediction"],
                sample["part_mask"],
                sample["object_mask"],
            ),
            "assets": result_assets,
        }

        if spec.get(
            "gated",
            False,
        ):
            gate = output[
                "gates"
            ].tolist()

            result["gate_weights"] = {
                "u": gate[0],
                "v": gate[1],
                "d": gate[2],
            }

        entry["results"][
            model_id
        ] = result

        if (
            position % 10 == 0
            or position
            == len(selections)
        ):
            print(
                f"  {model_id}: "
                f"{position}/"
                f"{len(selections)}"
            )

    del predictor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_catalogue(
    catalogue: dict[str, Any],
    stage_root: Path,
) -> None:
    models = [
        str(model["id"])
        for model
        in catalogue["models"]
    ]

    if not catalogue["samples"]:
        raise ValueError(
            "No samples were exported"
        )

    references = []

    for sample in catalogue["samples"]:
        if (
            set(sample["results"])
            != set(models)
        ):
            raise ValueError(
                "Incomplete model results "
                f"for {sample['id']}"
            )

        for asset in (
            sample["assets"].values()
        ):
            references.append(
                asset["file"]
            )

        for result in (
            sample["results"].values()
        ):
            for asset in (
                result["assets"].values()
            ):
                references.append(
                    asset["file"]
                )

    missing = [
        value
        for value in references
        if not (
            stage_root / value
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Generated catalogue "
            "references missing assets: "
            f"{missing[:10]}"
        )

    print(
        f"Validated "
        f"{len(catalogue['samples'])} "
        f"samples and "
        f"{len(references)} assets"
    )


def publish(
    stage_root: Path,
) -> None:
    source_assets = (
        stage_root
        / "assets"
        / "examples"
    )

    destination_assets = (
        WEB_ROOT
        / "assets"
        / "examples"
    )

    # IMPORTANT:
    # Replace the complete examples directory
    # instead of merging with the old one.
    #
    # This removes stale folders from previous
    # experiments such as crop_alignment and
    # robustness outputs.
    if destination_assets.exists():
        print(
            "Removing old web assets: "
            f"{destination_assets}"
        )

        shutil.rmtree(
            destination_assets
        )

    shutil.copytree(
        source_assets,
        destination_assets,
    )

    destination_catalogue = (
        WEB_ROOT
        / "data"
        / "catalogue.json"
    )

    temporary = (
        destination_catalogue
        .with_suffix(".json.tmp")
    )

    shutil.copy2(
        stage_root
        / "data"
        / "catalogue.json",
        temporary,
    )

    temporary.replace(
        destination_catalogue
    )

    print(
        "Published clean assets to "
        f"{destination_assets}"
    )

    print(
        "Published catalogue to "
        f"{destination_catalogue}"
    )


def main() -> None:
    args = parse_args()

    if (
        args.parts_per_parent < 1
        or args.examples_per_part < 1
    ):
        raise ValueError(
            "Sample-selection counts "
            "must be positive"
        )

    stage_root = (
        args.output.resolve()
    )

    assets_root = (
        stage_root
        / "assets"
        / "examples"
    )

    catalogue_path = (
        stage_root
        / "data"
        / "catalogue.json"
    )

    if args.publish_staged:
        catalogue = load_json(
            catalogue_path
        )

        validate_catalogue(
            catalogue,
            stage_root,
        )

        publish(
            stage_root
        )

        return

    assets_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalogue_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = load_json(
        TEMPLATE_PATH
    )

    registry = load_json(
        REGISTRY_PATH
    )

    specs = model_specs(
        template,
        registry,
    )

    datasets = {
        split: PascalPart116Dataset(
            split=split
        )
        for split in (
            "test_seen",
            "test_unseen",
        )
    }

    selections = []

    for split, dataset in (
        datasets.items()
    ):
        selected = select_samples(
            dataset,
            split,
            args.parts_per_parent,
            args.examples_per_part,
            args.seed,
        )

        print(
            f"Selected "
            f"{len(selected)} "
            f"{split} examples"
        )

        selections.extend(
            selected
        )

    entries = [
        common_entry(
            item,
            assets_root,
        )
        for item in selections
    ]

    for spec in specs:
        export_model(
            spec,
            selections,
            entries,
            assets_root,
            args.device,
        )

    catalogue = {
        "schema_version": 1,
        "object_parts": (
            template["object_parts"]
        ),
        "models": (
            template["models"]
        ),
        "samples": entries,
    }

    catalogue_path.write_text(
        json.dumps(
            catalogue,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    validate_catalogue(
        catalogue,
        stage_root,
    )

    print(
        "Staged catalogue: "
        f"{catalogue_path}"
    )

    if args.publish:
        publish(
            stage_root
        )
    else:
        print(
            "Review the staged export, "
            "then use --publish-staged "
            "to update web/."
        )


if __name__ == "__main__":
    main()