from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets import ObjectCentricDataset

from experiments.implementations.features.dino_features import (
    get_device,
    load_dino_model,
)

from experiments.implementations.features.clip_features import (
    load_clip_model,
    extract_clip_features,
)

from experiments.implementations.part_query_alignment.alignment_model import (
    PartQueryAlignmentSegmenter,
)

from experiments.implementations.object_centric_zoom.crop_projection import (
    project_crop_prediction_to_full_view,
)

from experiments.implementations.features.metrics import (
    dice_score,
    iou_score,
)


DEVICE = get_device()

BATCH_SIZE = 1

NUM_WORKERS = 2

MASK_THRESHOLD = 0.5

MODE = "alignment_mask"


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "object_zoom_evaluation"
)


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
    checkpoint_path,
    dino_model,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model = PartQueryAlignmentSegmenter(
        dino_encoder=dino_model,
        mode=MODE,
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

    model.eval()

    return model


def evaluate_split(
    split,
    full_model,
    crop_model,
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

    full_iou_total = 0.0
    full_dice_total = 0.0

    crop_iou_total = 0.0
    crop_dice_total = 0.0

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


            full_image = batch[
                "full_image"
            ].to(
                DEVICE
            )

            full_object = batch[
                "full_object_mask"
            ].to(
                DEVICE
            )

            full_part = batch[
                "full_part_mask"
            ].to(
                DEVICE
            )

            full_u = batch[
                "full_relative_u"
            ].to(
                DEVICE
            )

            full_v = batch[
                "full_relative_v"
            ].to(
                DEVICE
            )


            full_logits, _ = full_model(
                full_image,
                text,
                full_object,
                full_u,
                full_v,
            )


            full_iou = iou_score(
                full_logits,
                full_part,
                threshold=MASK_THRESHOLD,
            )

            full_dice = dice_score(
                full_logits,
                full_part,
                threshold=MASK_THRESHOLD,
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


            crop_logits, _ = crop_model(
                crop_image,
                text,
                crop_object,
                crop_u,
                crop_v,
            )


            crop_probability = torch.sigmoid(
                crop_logits
            )[0]


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


            projected_binary = (
                projected
                >= MASK_THRESHOLD
            )

            target_binary = (
                full_part
                >= 0.5
            )

            intersection = (
                projected_binary
                & target_binary
            ).sum(
                dim=(
                    1,
                    2,
                    3,
                )
            ).float()

            union = (
                projected_binary
                | target_binary
            ).sum(
                dim=(
                    1,
                    2,
                    3,
                )
            ).float()

            crop_iou = (
                (
                    intersection
                    + 1.0
                )
                /
                (
                    union
                    + 1.0
                )
            ).mean()

            dice_denominator = (
                projected_binary.sum(
                    dim=(
                        1,
                        2,
                        3,
                    )
                ).float()
                + target_binary.sum(
                    dim=(
                        1,
                        2,
                        3,
                    )
                ).float()
            )

            crop_dice = (
                (
                    2.0 * intersection
                    + 1.0
                )
                /
                (
                    dice_denominator
                    + 1.0
                )
            ).mean()


            full_iou_total += (
                full_iou.item()
            )

            full_dice_total += (
                full_dice.item()
            )

            crop_iou_total += (
                crop_iou.item()
            )

            crop_dice_total += (
                crop_dice.item()
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

        "full_image":
            {
                "iou":
                    full_iou_total
                    / total_samples,

                "dice":
                    full_dice_total
                    / total_samples,
            },

        "object_crop":
            {
                "iou":
                    crop_iou_total
                    / total_samples,

                "dice":
                    crop_dice_total
                    / total_samples,
            },
    }


def main():
    print(
        "Device:",
        DEVICE,
    )

    full_checkpoint = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "part_query_alignment"
        / MODE
        / "best.pt"
    )

    crop_checkpoint = (
        PROJECT_ROOT
        / "outputs"
        / "object_zoom"
        / MODE
        / "best.pt"
    )


    if not full_checkpoint.is_file():
        raise FileNotFoundError(
            f"Full-image checkpoint "
            f"not found: "
            f"{full_checkpoint}"
        )


    if not crop_checkpoint.is_file():
        raise FileNotFoundError(
            f"Crop checkpoint "
            f"not found: "
            f"{crop_checkpoint}"
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
        "Loading full-image model..."
    )

    full_model = load_model(
        full_checkpoint,
        dino_model,
    )


    print(
        "Loading crop model..."
    )

    crop_model = load_model(
        crop_checkpoint,
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

        split_results = (
            evaluate_split(
                split,
                full_model,
                crop_model,
                clip_model,
                tokenizer,
            )
        )

        results.append(
            split_results
        )


        print(
            "Full-image IoU:",
            split_results[
                "full_image"
            ][
                "iou"
            ],
        )

        print(
            "Crop IoU:",
            split_results[
                "object_crop"
            ][
                "iou"
            ],
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path = (
        OUTPUT_DIR
        / "object_zoom_results.json"
    )


    output_path.write_text(
        json.dumps(
            results,
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