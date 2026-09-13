from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .pascal_part116 import (
        PascalPart116Dataset,
    )


__all__ = [
    "PascalPart116Dataset",
]


def __getattr__(name):
    if name == "PascalPart116Dataset":
        from .pascal_part116 import (
            PascalPart116Dataset,
        )

        return PascalPart116Dataset

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )