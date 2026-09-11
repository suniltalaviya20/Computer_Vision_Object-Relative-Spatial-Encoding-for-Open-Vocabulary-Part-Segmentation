from torch.utils.data import Dataset

from datasets import PascalPart116Dataset

from src.preprocessing import (
    preprocess_segmentation_image,
    preprocess_mask,
)


class SegmentationDataset(Dataset):

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

        self.image_size = image_size

    def __len__(self):
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
            image,
            resize_info,
        ) = (
            preprocess_segmentation_image(
                sample["image"],
                self.image_size,
            )
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
            part_mask
            & object_mask
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

        return {
            "sample_id":
                sample["sample_id"],

            "image_id":
                sample["image_id"],

            "image":
                image,

            "display_image":
                display_image,

            "object_mask":
                object_mask.float(),

            "part_mask":
                part_mask.float(),

            "query":
                sample["query"],

            "part_name":
                sample["part_name"],

            "object_name":
                sample["object_name"],

            "evaluation_split":
                sample[
                    "evaluation_split"
                ],
        }