from torch.utils.data import Dataset

from datasets import (
    PascalPart116Dataset,
    ObjectCentricDataset,
)

from src.features.preprocessing import (
    preprocess_segmentation_image,
    preprocess_mask,
)

from src.geometry_comparison.geometry import (
    create_relative_uv,
    create_boundary_distance,
)

from src.object_centric_zoom.object_zoom import (
    get_square_object_crop,
    square_crop_with_padding,
    prepare_crop_image,
    prepare_crop_mask,
)

from src.robustness.robustness import (
    rotate_sample,
    erode_mask,
    dilate_mask,
    deterministic_shift,
)


# ============================================================
# WEBSITE CONDITIONS
# ============================================================

DEMO_ROBUSTNESS_CONDITIONS = [

    {
        "id": "clean",
        "label": "Clean",
        "type": "clean",
        "value": None,
    },

    {
        "id": "rotate_15",
        "label": "Rotate 15°",
        "type": "rotation",
        "value": 15,
    },

    {
        "id": "rotate_45",
        "label": "Rotate 45°",
        "type": "rotation",
        "value": 45,
    },

    {
        "id": "rotate_90",
        "label": "Rotate 90°",
        "type": "rotation",
        "value": 90,
    },

    {
        "id": "erode_5",
        "label": "Erode parent mask",
        "type": "mask_noise",
        "value": "erode_5",
    },

    {
        "id": "dilate_5",
        "label": "Dilate parent mask",
        "type": "mask_noise",
        "value": "dilate_5",
    },

    {
        "id": "shift_5",
        "label": "Shift parent mask",
        "type": "mask_noise",
        "value": "shift_5",
    },
]


# ============================================================
# CONDITION LOOKUP
# ============================================================

def get_robustness_condition(
    condition_id,
):

    for condition in DEMO_ROBUSTNESS_CONDITIONS:

        if condition["id"] == condition_id:
            return condition

    raise KeyError(
        f"Unknown robustness condition: "
        f"{condition_id}"
    )


# ============================================================
# OBJECT-CENTRIC ROBUSTNESS DATASET
#
# IMPORTANT:
#
# Perturbations happen BEFORE object-centric preprocessing.
#
# This means:
#
# RAW IMAGE
#     ↓
# perturb
#     ↓
# normal full-image preprocessing
#     ↓
# normal object crop
#     ↓
# resize crop to 224
#     ↓
# recompute U/V/D
#
# This avoids cropping from an already-downsampled 224 image.
# ============================================================

class RobustObjectCentricDataset(
    Dataset
):

    def __init__(
        self,
        split,
        condition_id,
        image_size=224,
        context_ratio=0.15,
    ):

        # ----------------------------------------------------
        # Normal dataset.
        #
        # Clean condition returns this directly so clean
        # predictions remain EXACTLY the normal predictions.
        # ----------------------------------------------------

        self.clean_dataset = (
            ObjectCentricDataset(
                split=split,
                image_size=image_size,
                context_ratio=context_ratio,
            )
        )


        # ----------------------------------------------------
        # Raw Pascal-Part data.
        #
        # Non-clean perturbations are applied here BEFORE
        # preprocessing and cropping.
        # ----------------------------------------------------

        self.raw_dataset = (
            PascalPart116Dataset(
                split=split
            )
        )


        self.condition = (
            get_robustness_condition(
                condition_id
            )
        )

        self.condition_id = (
            condition_id
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
            self.raw_dataset
        )


    # ========================================================
    # SCALE A PIXEL PERTURBATION
    #
    # Notebook robustness uses values such as 5 pixels in
    # the prepared 224 view.
    #
    # Here perturbation happens in the raw image.
    #
    # Therefore convert 5 prepared pixels into approximately
    # the equivalent distance in raw-image coordinates.
    # ========================================================

    def _raw_shift_magnitude(
        self,
        prepared_pixels,
        height,
        width,
    ):

        resize_scale = min(
            self.image_size / height,
            self.image_size / width,
        )


        raw_pixels = round(
            prepared_pixels
            / resize_scale
        )


        return max(
            1,
            int(
                raw_pixels
            ),
        )


    # ========================================================
    # SCALE MORPHOLOGY KERNEL
    #
    # Preserve approximately the same morphological radius
    # as a 5x5 operation in the 224 prepared view.
    # ========================================================

    def _raw_kernel_size(
        self,
        prepared_kernel,
        height,
        width,
    ):

        resize_scale = min(
            self.image_size / height,
            self.image_size / width,
        )


        prepared_radius = (
            prepared_kernel - 1
        ) / 2.0


        raw_radius = round(
            prepared_radius
            / resize_scale
        )


        raw_radius = max(
            1,
            int(
                raw_radius
            ),
        )


        return (
            2 * raw_radius
            + 1
        )


    # ========================================================
    # CORRUPT RAW PARENT MASK
    # ========================================================

    def _corrupt_raw_mask(
        self,
        mask,
        condition,
        index,
        height,
        width,
    ):

        if condition == "erode_5":

            kernel_size = (
                self._raw_kernel_size(
                    5,
                    height,
                    width,
                )
            )

            return erode_mask(
                mask,
                kernel_size=kernel_size,
            )


        if condition == "dilate_5":

            kernel_size = (
                self._raw_kernel_size(
                    5,
                    height,
                    width,
                )
            )

            return dilate_mask(
                mask,
                kernel_size=kernel_size,
            )


        if condition == "shift_5":

            magnitude = (
                self._raw_shift_magnitude(
                    5,
                    height,
                    width,
                )
            )

            shifted = (
                deterministic_shift(
                    mask
                    .float()
                    .unsqueeze(0),
                    magnitude=magnitude,
                    index=index,
                )
            )

            return (
                shifted
                .squeeze(0)
                .bool()
            )

            return deterministic_shift(
                mask,
                magnitude=magnitude,
                index=index,
            )


        raise ValueError(
            f"Unknown mask condition: "
            f"{condition}"
        )


    # ========================================================
    # SAFE CROP INFO
    # ========================================================

    def _get_crop_info(
        self,
        object_mask,
        height,
        width,
    ):

        if object_mask.any():

            return get_square_object_crop(
                object_mask,
                context_ratio=(
                    self.context_ratio
                ),
            )


        # Extremely corrupted mask fallback.
        side = max(
            height,
            width,
        )


        return {
            "x1": 0,
            "y1": 0,
            "side": side,
        }


    # ========================================================
    # SAMPLE
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        condition_type = (
            self.condition[
                "type"
            ]
        )

        condition_value = (
            self.condition[
                "value"
            ]
        )


        # ====================================================
        # CLEAN
        #
        # Return the exact normal ObjectCentricDataset sample.
        # ====================================================

        if condition_type == "clean":

            output = dict(
                self.clean_dataset[
                    index
                ]
            )

            output[
                "robustness_condition"
            ] = "clean"

            output[
                "robustness_type"
            ] = "clean"

            output[
                "robustness_value"
            ] = None

            return output


        # ====================================================
        # RAW SAMPLE
        # ====================================================

        raw = (
            self.raw_dataset[
                index
            ]
        )


        image = (
            raw[
                "image"
            ]
            .clone()
        )


        object_mask = (
            raw[
                "object_mask"
            ]
            .clone()
            > 0.5
        )


        part_mask = (
            raw[
                "part_mask"
            ]
            .clone()
            > 0.5
        )


        (
            _,
            original_height,
            original_width,
        ) = image.shape


        # ====================================================
        # ROTATION
        #
        # Rotate RAW image + RAW parent + RAW GT.
        #
        # The model prediction will therefore be recomputed
        # from an actually rotated input.
        # ====================================================

        if condition_type == "rotation":

            # rotate_sample() expects masks as:
            # [1, H, W]
            #
            # Raw Pascal-Part masks are:
            # [H, W]
            #
            # Temporarily add the channel dimension for rotation,
            # then remove it again so the rest of the raw
            # object-centric pipeline stays unchanged.

            object_mask_for_rotation = (
                object_mask
                .float()
                .unsqueeze(0)
            )

            part_mask_for_rotation = (
                part_mask
                .float()
                .unsqueeze(0)
            )


            (
                image,
                _,
                rotated_object,
                rotated_part,
            ) = rotate_sample(
                image,
                image,
                object_mask_for_rotation,
                part_mask_for_rotation,
                angle=condition_value,
            )


            object_mask = (
                rotated_object
                .squeeze(0)
                .bool()
            )

            part_mask = (
                rotated_part
                .squeeze(0)
                .bool()
            )


        # ====================================================
        # PARENT MASK NOISE
        #
        # RGB unchanged.
        # GT unchanged.
        # Only supplied parent mask changes.
        # ====================================================

        elif condition_type == "mask_noise":

            object_mask = (
                self._corrupt_raw_mask(
                    object_mask,
                    condition=condition_value,
                    index=index,
                    height=original_height,
                    width=original_width,
                )
            )


        else:

            raise ValueError(
                f"Unsupported robustness type: "
                f"{condition_type}"
            )


        object_mask = (
            object_mask.bool()
        )

        part_mask = (
            part_mask.bool()
        )


        # ====================================================
        # FULL-IMAGE PREPROCESSING
        #
        # Same operation used by ObjectCentricDataset.
        # ====================================================

        (
            full_display,
            full_image,
            full_resize_info,
        ) = preprocess_segmentation_image(
            image,
            target_size=self.image_size,
        )


        full_object = preprocess_mask(
            object_mask,
            full_resize_info,
            preserve_positive=True,
        )


        full_part = preprocess_mask(
            part_mask,
            full_resize_info,
            preserve_positive=True,
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We deliberately do NOT intersect full_part with the
        # noisy parent mask.
        #
        # Ground truth must stay the true part annotation.
        # ----------------------------------------------------


        # ====================================================
        # FULL U / V / D
        # ====================================================

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


        # ====================================================
        # OBJECT-CENTRIC CROP
        #
        # Calculated from the RAW perturbed parent mask,
        # exactly where it belongs in the pipeline.
        # ====================================================

        crop_info = (
            self._get_crop_info(
                object_mask,
                original_height,
                original_width,
            )
        )


        x1 = int(
            crop_info[
                "x1"
            ]
        )

        y1 = int(
            crop_info[
                "y1"
            ]
        )

        side = int(
            crop_info[
                "side"
            ]
        )


        # ====================================================
        # RAW RGB CROP
        # ====================================================

        crop_rgb_raw = (
            square_crop_with_padding(
                image,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        # ====================================================
        # RAW PARENT CROP
        # ====================================================

        crop_object_raw = (
            square_crop_with_padding(
                object_mask,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        # ====================================================
        # RAW GT CROP
        # ====================================================

        crop_part_raw = (
            square_crop_with_padding(
                part_mask,
                x1,
                y1,
                side,
                fill_value=0,
            )
        )


        # ====================================================
        # PREPARE MODEL CROP
        #
        # Same operation as normal ObjectCentricDataset.
        # ====================================================

        (
            crop_display,
            crop_image,
        ) = prepare_crop_image(
            crop_rgb_raw,
            target_size=self.image_size,
        )


        crop_object = (
            prepare_crop_mask(
                crop_object_raw,
                target_size=self.image_size,
                preserve_positive=True,
            )
        )


        crop_part = (
            prepare_crop_mask(
                crop_part_raw,
                target_size=self.image_size,
                preserve_positive=True,
            )
        )


        # ----------------------------------------------------
        # Again, do not intersect GT with a corrupted parent
        # mask. GT is the evaluation target.
        # ----------------------------------------------------


        # ====================================================
        # RECOMPUTE CROP U / V / D
        # ====================================================

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


        # ====================================================
        # PART SIZE
        # ====================================================

        part_to_object_ratio = (
            full_part
            .sum()
            .float()
            /
            full_object
            .sum()
            .float()
            .clamp_min(
                1
            )
        )


        # ====================================================
        # RETURN
        # ====================================================

        return {

            "sample_id":
                raw[
                    "sample_id"
                ],

            "image_id":
                raw[
                    "image_id"
                ],

            "query":
                raw[
                    "query"
                ],

            "object_name":
                raw[
                    "object_name"
                ],

            "part_name":
                raw[
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
            # Object crop
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
            # Correct RAW mapping info
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


            # =========================
            # Robustness metadata
            # =========================

            "robustness_condition":
                self.condition_id,

            "robustness_type":
                condition_type,

            "robustness_value":
                condition_value,
        }