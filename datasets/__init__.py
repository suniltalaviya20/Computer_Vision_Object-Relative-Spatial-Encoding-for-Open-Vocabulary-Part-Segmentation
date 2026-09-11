from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .pascal_part116 import (
        PascalPart116Dataset,
    )

    from .segmentation_dataset import (
        SegmentationDataset,
    )

    from .geometry_dataset import (
        GeometryDataset,
    )
    from .robustness_dataset import (
        RobustnessDataset,
    )

    from .alignment_dataset import (
        AlignmentDataset,
    )

    from .object_centric_dataset import (
        ObjectCentricDataset,
    )


__all__ = [
    "PascalPart116Dataset",
    "SegmentationDataset",
    "GeometryDataset",
    "AlignmentDataset",
    "ObjectCentricDataset",
]


def __getattr__(name):
    if name == "PascalPart116Dataset":
        from .pascal_part116 import (
            PascalPart116Dataset,
        )

        return PascalPart116Dataset

    if name == "SegmentationDataset":
        from .segmentation_dataset import (
            SegmentationDataset,
        )

        return SegmentationDataset

    if name == "GeometryDataset":
        from .geometry_dataset import (
            GeometryDataset,
        )

        return GeometryDataset

    if name == "RobustnessDataset":
        from .robustness_dataset import (
            RobustnessDataset,
        )

        return RobustnessDataset

    if name == "AlignmentDataset":
        from .alignment_dataset import (
            AlignmentDataset,
        )

        return AlignmentDataset

    if name == "ObjectCentricDataset":
        from .object_centric_dataset import (
            ObjectCentricDataset,
        )

        return ObjectCentricDataset

    raise AttributeError(
        f"module {__name__!r} "
        f"has no attribute {name!r}"
    )