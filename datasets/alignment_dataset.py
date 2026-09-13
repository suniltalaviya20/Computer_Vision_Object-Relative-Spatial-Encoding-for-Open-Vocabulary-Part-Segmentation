from torch.utils.data import Dataset

from datasets import PascalPart116Dataset

from src.preprocessing import (
    preprocess_segmentation_image,
    preprocess_mask,
)

from src.geometry import (
    create_relative_uv,
)


class AlignmentDataset(Dataset):

    def __init__(
        self,
        split,
        image_size=224,
    ):
        self.base_dataset = (
            PascalPart116Dataset(
                split=split
            )
        )

        self.image_size = (
            image_size
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
        sample = (
            self.base_dataset[
                index
            ]
        )


        (
            display_image,
            model_image,
            resize_info,
        ) = preprocess_segmentation_image(
            sample["image"],
            target_size=self.image_size,
        )


        object_mask = preprocess_mask(
            sample["object_mask"],
            resize_info,
            preserve_positive=True,
        )


        part_mask = preprocess_mask(
            sample["part_mask"],
            resize_info,
            preserve_positive=True,
        )


        overlap = (
            object_mask
            & part_mask
        )


        if (
            part_mask.any()
            and not overlap.any()
        ):
            object_mask = (
                object_mask
                | part_mask
            )


        part_mask = (
            part_mask
            & object_mask
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
                model_image,

            "display_image":
                display_image,

            "object_mask":
                object_mask.float(),

            "part_mask":
                part_mask.float(),

            "relative_u":
                relative_u,

            "relative_v":
                relative_v,

            "query":
                sample[
                    "query"
                ],

            "object_name":
                sample[
                    "object_name"
                ],

            "part_name":
                sample[
                    "part_name"
                ],

            "full_part_name":
                sample[
                    "full_part_name"
                ],

            "part_id":
                sample[
                    "part_id"
                ],

            "evaluation_split":
                sample[
                    "evaluation_split"
                ],
        }