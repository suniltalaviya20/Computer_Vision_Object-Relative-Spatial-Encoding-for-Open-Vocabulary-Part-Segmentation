from pathlib import Path
import argparse
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


from datasets import ObjectCentricDataset

from src.features.dino_features import (
    get_device,
    load_dino_model,
)

from src.features.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from src.part_query_alignment.alignment_model import (
    PartQueryAlignmentSegmenter,
)

from src.object_centric_zoom.crop_projection import (
    project_crop_prediction_to_full_view,
)


DEVICE = get_device()

BATCH_SIZE = 1
NUM_WORKERS = 2
MASK_THRESHOLD = 0.5

VALID_MODES = {
    "alignment_fixed_uvd",
    "alignment_query_gated_uvd",
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "object_zoom_evaluation"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(
            VALID_MODES
        ),
    )

    return parser.parse_args()


def encode_query(
    clip_model,
    tokenizer,
    query,
):
    return extract_clip_features(
        clip_model,
        tokenizer,
        query,
        device=DEVICE,
    )


def load_model(
    mode,
    checkpoint_path,
    dino_model,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model = PartQueryAlignmentSegmenter(
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

    model.text_decoder_projection.load_state_dict(
        state[
            "text_decoder_projection"
        ]
    )

    model.decoder.load_state_dict(
        state[
            "decoder"
        ]
    )

    if (
        model.geometry_gate is not None
        and "geometry_gate" in state
    ):
        model.geometry_gate.load_state_dict(
            state[
                "geometry_gate"
            ]
        )

    model.eval()

    return model


def evaluate_split(
    split,
    model,
    clip_model,
    tokenizer,
):
    dataset = ObjectCentricDataset(
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    iou_total = 0.0
    dice_total = 0.0
    total_samples = 0


    with torch.no_grad():

        for index, batch in enumerate(
            loader
        ):
            query = batch[
                "query"
            ][0]

            text = encode_query(
                clip_model,
                tokenizer,
                query,
            )


            crop_image = batch[
                "crop_image"
            ].to(
                DEVICE
            )

            crop_object = batch[
                "crop_object_mask"
            ].to(
                DEVICE
            )

            crop_u = batch[
                "crop_relative_u"
            ].to(
                DEVICE
            )

            crop_v = batch[
                "crop_relative_v"
            ].to(
                DEVICE
            )

            crop_d = batch[
                "crop_boundary_d"
            ].to(
                DEVICE
            )


            crop_logits, _ = model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
                crop_d,
            )


            crop_probability = (
                torch.sigmoid(
                    crop_logits
                )[0]
            )


            projected = (
                project_crop_prediction_to_full_view(
                    crop_probability,
                    original_height=int(
                        batch[
                            "original_height"
                        ][0]
                    ),
                    original_width=int(
                        batch[
                            "original_width"
                        ][0]
                    ),
                    crop_x1=int(
                        batch[
                            "crop_x1"
                        ][0]
                    ),
                    crop_y1=int(
                        batch[
                            "crop_y1"
                        ][0]
                    ),
                    crop_side=int(
                        batch[
                            "crop_side"
                        ][0]
                    ),
                    target_size=224,
                )
            )


            projected = (
                projected
                .unsqueeze(0)
                .to(
                    DEVICE
                )
            )


            full_part = batch[
                "full_part_mask"
            ].to(
                DEVICE
            )


            prediction = (
                projected
                > MASK_THRESHOLD
            )

            target = (
                full_part
                > MASK_THRESHOLD
            )


            intersection = (
                prediction
                & target
            ).sum().float()


            union = (
                prediction
                | target
            ).sum().float()


            prediction_sum = (
                prediction
                .sum()
                .float()
            )

            target_sum = (
                target
                .sum()
                .float()
            )


            iou = (
                intersection
                / union.clamp_min(
                    1.0
                )
            )


            dice = (
                2.0
                * intersection
                / (
                    prediction_sum
                    + target_sum
                ).clamp_min(
                    1.0
                )
            )


            iou_total += (
                iou.item()
            )

            dice_total += (
                dice.item()
            )

            total_samples += 1


            if (
                index + 1
            ) % 100 == 0:
                print(
                    f"{split}: "
                    f"{index + 1}/"
                    f"{len(dataset)}"
                )


    return {
        "split":
            split,

        "samples":
            total_samples,

        "iou":
            (
                iou_total
                / total_samples
            ),

        "dice":
            (
                dice_total
                / total_samples
            ),
    }


def main():
    args = parse_args()

    mode = args.mode

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Mode:",
        mode,
    )


    checkpoint_path = (
        PROJECT_ROOT
        / "outputs"
        / "object_zoom"
        / mode
        / "best.pt"
    )


    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )


    print(
        "Checkpoint:",
        checkpoint_path,
    )


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


    print(
        "Loading crop model..."
    )

    model = load_model(
        mode,
        checkpoint_path,
        dino_model,
    )


    results = []


    for split in [
        "test_seen",
        "test_unseen",
    ]:
        print()
        print(
            "Evaluating:",
            split,
        )

        result = evaluate_split(
            split,
            model,
            clip_model,
            tokenizer,
        )

        results.append(
            result
        )

        print(
            "IoU:",
            result["iou"],
        )

        print(
            "Dice:",
            result["dice"],
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        OUTPUT_DIR
        / f"{mode}_results.json"
    )


    output_path.write_text(
        json.dumps(
            {
                "mode":
                    mode,

                "checkpoint":
                    str(
                        checkpoint_path
                    ),

                "results":
                    results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
