from torch.utils.data import Dataset

from datasets import PascalPart116Dataset

from src.preprocessing import (
    preprocess_segmentation_image,
    preprocess_mask,
)

from src.geometry import (
    create_relative_uv,
    create_boundary_distance,
)

from src.object_zoom import (
    get_square_object_crop,
    square_crop_with_padding,
    prepare_crop_image,
    prepare_crop_mask,
)


class ObjectCentricDataset(Dataset):

    def __init__(
        self,
        split,
        image_size=224,
        context_ratio=0.15,
    ):
        self.base_dataset = (
            PascalPart116Dataset(
                split=split
            )
        )

        self.image_size = (
            image_size
        )

        self.context_ratio = (
            context_ratio
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


        original_image = (
            sample["image"]
        )

        original_object = (
            sample["object_mask"]
        )

        original_part = (
            sample["part_mask"]
        )


        (
            _,
            original_height,
            original_width,
        ) = original_image.shape


        # =====================================
        # Full-image representation
        # =====================================

        (
            full_display,
            full_image,
            full_resize_info,
        ) = preprocess_segmentation_image(
            original_image,
            target_size=self.image_size,
        )


        full_object = preprocess_mask(
            original_object,
            full_resize_info,
            preserve_positive=True,
        )


        full_part = preprocess_mask(
            original_part,
            full_resize_info,
            preserve_positive=True,
        )


        if (
            full_part.any()
            and not (
                full_part
                & full_object
            ).any()
        ):
            full_object = (
                full_object
                | full_part
            )


        full_part = (
            full_part
            & full_object
        )


        (
            full_u,
            full_v,
        ) = create_relative_uv(
            full_object
        )


        full_d = (
            create_boundary_distance(
                full_object
            )
        )


        # =====================================
        # Object-centric crop
        # =====================================

        crop_info = (
            get_square_object_crop(
                original_object,
                context_ratio=(
                    self.context_ratio
                ),
            )
        )


        x1 = crop_info[
            "x1"
        ]

        y1 = crop_info[
            "y1"
        ]

        side = crop_info[
            "side"
        ]


        crop_rgb_raw = (
            square_crop_with_padding(
                original_image,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        crop_object_raw = (
            square_crop_with_padding(
                original_object,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        crop_part_raw = (
            square_crop_with_padding(
                original_part,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        (
            crop_display,
            crop_image,
        ) = prepare_crop_image(
            crop_rgb_raw,
            target_size=(
                self.image_size
            ),
        )


        crop_object = (
            prepare_crop_mask(
                crop_object_raw,
                target_size=(
                    self.image_size
                ),
                preserve_positive=True,
            )
        )


        crop_part = (
            prepare_crop_mask(
                crop_part_raw,
                target_size=(
                    self.image_size
                ),
                preserve_positive=True,
            )
        )


        if (
            crop_part.any()
            and not (
                crop_part
                & crop_object
            ).any()
        ):
            crop_object = (
                crop_object
                | crop_part
            )


        crop_part = (
            crop_part
            & crop_object
        )


        (
            crop_u,
            crop_v,
        ) = create_relative_uv(
            crop_object
        )


        crop_d = (
            create_boundary_distance(
                crop_object
            )
        )


        part_to_object_ratio = (
            full_part.sum().float()
            /
            full_object
            .sum()
            .float()
            .clamp_min(1)
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


            # =========================
            # Full-image view
            # =========================

            "full_display_image":
                full_display,

            "full_image":
                full_image,

            "full_object_mask":
                full_object.float(),

            "full_part_mask":
                full_part.float(),

            "full_relative_u":
                full_u,

            "full_relative_v":
                full_v,

            "full_boundary_d":
                full_d,


            # =========================
            # Crop view
            # =========================

            "crop_display_image":
                crop_display,

            "crop_image":
                crop_image,

            "crop_object_mask":
                crop_object.float(),

            "crop_part_mask":
                crop_part.float(),

            "crop_relative_u":
                crop_u,

            "crop_relative_v":
                crop_v,

            "crop_boundary_d":
                crop_d,


            # =========================
            # Mapping information
            # =========================

            "original_height":
                original_height,

            "original_width":
                original_width,

            "crop_x1":
                x1,

            "crop_y1":
                y1,

            "crop_side":
                side,


            "part_to_object_ratio":
                part_to_object_ratio,
        }
    