from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.geometry import create_relative_coordinate_maps


def main() -> None:
    height = 256
    width = 256

    # Synthetic parent-object mask.
    object_mask = torch.zeros((height, width), dtype=torch.float32)

    # Create a rectangular object:
    # rows 60–200 and columns 40–180.
    object_mask[60:200, 40:180] = 1.0

    # Add a smaller extension to make the shape non-rectangular.
    object_mask[100:160, 180:220] = 1.0

    u_map, v_map = create_relative_coordinate_maps(object_mask)

    output_dir = PROJECT_ROOT / "outputs" / "geometry_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(object_mask.numpy(), cmap="gray")
    axes[0].set_title("Parent-object mask")

    image_u = axes[1].imshow(u_map.numpy(), vmin=0, vmax=1)
    axes[1].set_title("Relative horizontal map U")
    figure.colorbar(image_u, ax=axes[1], fraction=0.046)

    image_v = axes[2].imshow(v_map.numpy(), vmin=0, vmax=1)
    axes[2].set_title("Relative vertical map V")
    figure.colorbar(image_v, ax=axes[2], fraction=0.046)

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    output_path = output_dir / "relative_coordinates.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved visualization to: {output_path}")
    print(f"U range inside object: "
          f"{u_map[object_mask.bool()].min():.3f} - "
          f"{u_map[object_mask.bool()].max():.3f}")
    print(f"V range inside object: "
          f"{v_map[object_mask.bool()].min():.3f} - "
          f"{v_map[object_mask.bool()].max():.3f}")
    print(
        "Maximum value outside object:",
        max(
            u_map[~object_mask.bool()].max().item(),
            v_map[~object_mask.bool()].max().item(),
        ),
    )


if __name__ == "__main__":
    main()