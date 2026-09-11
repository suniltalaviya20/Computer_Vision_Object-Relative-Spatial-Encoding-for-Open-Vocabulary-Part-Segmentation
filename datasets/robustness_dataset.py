from torch.utils.data import Dataset

from datasets import SegmentationDataset

from src.geometry import (
    create_absolute_xy,
    create_relative_uv,
)

from src.robustness import (
    rotate_sample,
    corrupt_object_mask,
)


class RobustnessDataset(Dataset):

    def __init__(
        self,
        split,
        perturbation_type,
        perturbation_value,
        image_size=224,
    ):
        self.base_dataset = SegmentationDataset(
            split=split,
            image_size=image_size,
        )

        valid_types = {
            "rotation",
            "mask_noise",
        }

        if perturbation_type not in valid_types:
            raise ValueError(
                f"Unknown perturbation type: "
                f"{perturbation_type}"
            )

        self.perturbation_type = (
            perturbation_type
        )

        self.perturbation_value = (
            perturbation_value
        )


    def __len__(
        self,
    ):
        return len(
            self.base_dataset
        )


    def __getitem__(
        self,
        index,
    ):
        sample = self.base_dataset[
            index
        ]

        image = sample[
            "image"
        ]

        display_image = sample[
            "display_image"
        ]

        object_mask = (
            sample[
                "object_mask"
            ] > 0.5
        )

        part_mask = (
            sample[
                "part_mask"
            ] > 0.5
        )


        if (
            self.perturbation_type
            == "rotation"
        ):
            (
                display_image,
                image,
                object_mask,
                part_mask,
            ) = rotate_sample(
                display_image,
                image,
                object_mask,
                part_mask,
                angle=(
                    self.perturbation_value
                ),
            )


        elif (
            self.perturbation_type
            == "mask_noise"
        ):
            object_mask = (
                corrupt_object_mask(
                    object_mask,
                    condition=(
                        self.perturbation_value
                    ),
                    index=index,
                )
            )


        object_mask = (
            object_mask.float()
        )

        part_mask = (
            part_mask.float()
        )


        (
            absolute_x,
            absolute_y,
        ) = create_absolute_xy(
            object_mask
        )


        (
            relative_u,
            relative_v,
        ) = create_relative_uv(
            object_mask
        )


        return {
            "sample_id":
                sample[
                    "sample_id"
                ],

            "image_id":
                sample[
                    "image_id"
                ],

            "image":
                image,

            "display_image":
                display_image,

            "object_mask":
                object_mask,

            "part_mask":
                part_mask,

            "absolute_x":
                absolute_x,

            "absolute_y":
                absolute_y,

            "relative_u":
                relative_u,

            "relative_v":
                relative_v,

            "query":
                sample[
                    "query"
                ],

            "part_name":
                sample[
                    "part_name"
                ],

            "object_name":
                sample[
                    "object_name"
                ],

            "evaluation_split":
                sample[
                    "evaluation_split"
                ],

            "perturbation_type":
                self.perturbation_type,

            "perturbation_value":
                self.perturbation_value,
        }