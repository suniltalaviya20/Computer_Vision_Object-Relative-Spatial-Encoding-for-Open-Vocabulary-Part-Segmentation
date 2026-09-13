from torch.utils.data import Dataset

from datasets import SegmentationDataset

from src.geometry import (
    create_absolute_xy,
    create_relative_uv,
)


class GeometryDataset(Dataset):

    def __init__(
        self,
        split,
        image_size=224,
    ):
        self.base_dataset = (
            SegmentationDataset(
                split=split,
                image_size=image_size,
            )
        )

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

        object_mask = sample[
            "object_mask"
        ]

        absolute_x, absolute_y = (
            create_absolute_xy(
                object_mask
            )
        )

        relative_u, relative_v = (
            create_relative_uv(
                object_mask
            )
        )

        return {
            "sample_id":
                sample["sample_id"],

            "image_id":
                sample["image_id"],

            "image":
                sample["image"],

            "display_image":
                sample["display_image"],

            "object_mask":
                object_mask,

            "part_mask":
                sample["part_mask"],

            "absolute_x":
                absolute_x,

            "absolute_y":
                absolute_y,

            "relative_u":
                relative_u,

            "relative_v":
                relative_v,

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