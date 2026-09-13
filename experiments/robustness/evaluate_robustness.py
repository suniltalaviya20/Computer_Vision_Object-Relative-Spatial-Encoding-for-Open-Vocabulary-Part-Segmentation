from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import RobustnessDataset

from src.features.dino_features import (
    get_device,
    load_dino_model,
)

from src.features.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.geometry_comparison.geometry_model import (
    GeometryPartSegmenter,
)

from src.features.metrics import (
    dice_score,
    iou_score,
)

from src.robustness.robustness import (
    ROTATION_ANGLES,
    MASK_NOISE_CONDITIONS,
)


DEVICE = get_device()

BATCH_SIZE = 16
NUM_WORKERS = 4

MASK_THRESHOLD = 0.5


MODES = [
    "object_mask",
    "absolute_xy",
    "relative_uv",
]


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "robustness"
)


def encode_queries(
    clip_model,
    tokenizer,
    queries,
    device,
):
    features = []

    for query in queries:
        feature = extract_clip_features(
            clip_model,
            tokenizer,
            query,
            device=device,
        )

        features.append(
            feature
        )

    return torch.cat(
        features,
        dim=0,
    )


def load_geometry_checkpoint(
    mode,
    dino_model,
):
    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "geometry"
        / mode
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model = GeometryPartSegmenter(
        dino_encoder=dino_model,
        mode=mode,
    ).to(
        DEVICE
    )

    state = checkpoint[
        "model_state"
    ]

    model.visual_projection.load_state_dict(
        state[
            "visual_projection"
        ]
    )

    model.text_projection.load_state_dict(
        state[
            "text_projection"
        ]
    )

    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )

    model.eval()

    return model


def evaluate_dataset(
    model,
    dataset,
    clip_model,
    tokenizer,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "image"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            object_masks = batch[
                "object_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            part_masks = batch[
                "part_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            absolute_x = batch[
                "absolute_x"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            absolute_y = batch[
                "absolute_y"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            relative_u = batch[
                "relative_u"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            relative_v = batch[
                "relative_v"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            text_embeddings = (
                encode_queries(
                    clip_model,
                    tokenizer,
                    batch["query"],
                    DEVICE,
                )
            )

            logits = model(
                images,
                text_embeddings,
                object_masks,
                absolute_x,
                absolute_y,
                relative_u,
                relative_v,
            )

            batch_iou = iou_score(
                logits,
                part_masks,
                threshold=MASK_THRESHOLD,
            )

            batch_dice = dice_score(
                logits,
                part_masks,
                threshold=MASK_THRESHOLD,
            )

            batch_size = (
                images.shape[0]
            )

            total_iou += (
                batch_iou.item()
                * batch_size
            )

            total_dice += (
                batch_dice.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    return {
        "iou":
            total_iou
            / total_samples,

        "dice":
            total_dice
            / total_samples,

        "samples":
            total_samples,
    }


def evaluate_rotation(
    model,
    mode,
    clip_model,
    tokenizer,
):
    results = []

    print()
    print(
        "Rotation robustness"
    )

    print(
        "-------------------"
    )

    for angle in ROTATION_ANGLES:
        dataset = RobustnessDataset(
            split="test_unseen",
            perturbation_type="rotation",
            perturbation_value=angle,
        )

        metrics = evaluate_dataset(
            model,
            dataset,
            clip_model,
            tokenizer,
        )

        row = {
            "mode":
                mode,

            "perturbation":
                "rotation",

            "value":
                angle,

            "iou":
                metrics[
                    "iou"
                ],

            "dice":
                metrics[
                    "dice"
                ],

            "samples":
                metrics[
                    "samples"
                ],
        }

        results.append(
            row
        )

        print(
            f"{angle:3d} degrees | "
            f"IoU: "
            f"{metrics['iou']:.4f} | "
            f"Dice: "
            f"{metrics['dice']:.4f}"
        )

    return results


def evaluate_mask_noise(
    model,
    mode,
    clip_model,
    tokenizer,
):
    results = []

    print()
    print(
        "Mask robustness"
    )

    print(
        "---------------"
    )

    for condition in MASK_NOISE_CONDITIONS:
        dataset = RobustnessDataset(
            split="test_unseen",
            perturbation_type="mask_noise",
            perturbation_value=condition,
        )

        metrics = evaluate_dataset(
            model,
            dataset,
            clip_model,
            tokenizer,
        )

        row = {
            "mode":
                mode,

            "perturbation":
                "mask_noise",

            "value":
                condition,

            "iou":
                metrics[
                    "iou"
                ],

            "dice":
                metrics[
                    "dice"
                ],

            "samples":
                metrics[
                    "samples"
                ],
        }

        results.append(
            row
        )

        print(
            f"{condition:12s} | "
            f"IoU: "
            f"{metrics['iou']:.4f} | "
            f"Dice: "
            f"{metrics['dice']:.4f}"
        )

    return results


def main():
    print(
        "Device:",
        DEVICE,
    )

    print()
    print(
        "Loading DINOv2..."
    )

    dino_model = load_dino_model(
        device=DEVICE
    )

    print(
        "Loading CLIP..."
    )

    clip_model, tokenizer = (
        load_clip_model(
            device=DEVICE
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for mode in MODES:
        print()
        print(
            "========================================"
        )

        print(
            "Mode:",
            mode,
        )

        print(
            "========================================"
        )

        model = load_geometry_checkpoint(
            mode,
            dino_model,
        )

        rotation_results = (
            evaluate_rotation(
                model,
                mode,
                clip_model,
                tokenizer,
            )
        )

        mask_results = (
            evaluate_mask_noise(
                model,
                mode,
                clip_model,
                tokenizer,
            )
        )

        all_results.extend(
            rotation_results
        )

        all_results.extend(
            mask_results
        )

    output_path = (
        OUTPUT_DIR
        / "robustness_results.json"
    )

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved results:",
        output_path,
    )

    print()
    print(
        "Robustness evaluation completed."
    )


if __name__ == "__main__":
    main()