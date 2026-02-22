"""
Offset discretization for NUDGE.

Maps continuous offsets (meters) to discrete classes {-3..+3} and back.
Per-axis exponential thresholds (base 3x) matched to each axis's natural range:

XY thresholds [2, 6, 18]mm — gripper lateral corrections:
| Class | Label   | Offset range  |
|-------|---------|---------------|
|   0   | aligned | < 2mm         |
|   1   | nudge   | 2 - 6mm       |
|   2   | shift   | 6 - 18mm      |
|   3   | jump    | > 18mm        |

Z thresholds [8, 24, 72]mm — approach descent corrections:
| Class | Label   | Offset range  |
|-------|---------|---------------|
|   0   | aligned | < 8mm         |
|   1   | nudge   | 8 - 24mm      |
|   2   | shift   | 24 - 72mm     |
|   3   | jump    | > 72mm        |

Sign determines direction: negative classes = negative offset direction.
Class index in [0, 6] maps to signed class in [-3, +3] via: signed = index - 3.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DiscretizeConfig:
    """Thresholds for magnitude bins (in meters), per axis."""
    # Boundaries between magnitude classes (ascending, in meters)
    # Class 0: [0, thresholds[0])
    # Class 1: [thresholds[0], thresholds[1])
    # Class 2: [thresholds[1], thresholds[2])
    # Class 3: [thresholds[2], inf)
    thresholds_xy: List[float] = field(default_factory=lambda: [0.002, 0.006, 0.018])
    thresholds_z: List[float] = field(default_factory=lambda: [0.008, 0.024, 0.072])

    # Representative magnitudes for each class (meters), used for class->continuous
    # Midpoint of each bin (last bin uses 1.5x the upper threshold)
    representatives_xy: List[float] = field(default_factory=lambda: [0.0, 0.004, 0.012, 0.025])
    representatives_z: List[float] = field(default_factory=lambda: [0.0, 0.016, 0.048, 0.100])


DEFAULT_CONFIG = DiscretizeConfig()


def continuous_to_class(
    offset_m: float,
    axis: str = "x",
    config: DiscretizeConfig = DEFAULT_CONFIG,
) -> int:
    """
    Convert a continuous offset (meters, signed) to a class index [0, 6].

    Args:
        offset_m: Signed offset in meters.
        axis: One of "x", "y", "z". X and Y use lateral thresholds, Z uses descent thresholds.
        config: Discretization config.

    Returns:
        Class index 3 = aligned (0). Index 0 = -3 (large negative). Index 6 = +3 (large positive).
    """
    thresholds = config.thresholds_z if axis == "z" else config.thresholds_xy

    sign = 1 if offset_m >= 0 else -1
    mag = abs(offset_m)

    # Determine magnitude class (0-3)
    mag_class = 0
    for i, thresh in enumerate(thresholds):
        if mag >= thresh:
            mag_class = i + 1
        else:
            break

    # Convert signed magnitude class to index [0, 6]
    signed_class = sign * mag_class  # range [-3, +3]
    return signed_class + 3  # range [0, 6]


def class_to_continuous(
    cls_index: int,
    axis: str = "x",
    config: DiscretizeConfig = DEFAULT_CONFIG,
) -> float:
    """
    Convert a class index [0, 6] to a representative offset in meters.

    Returns signed value: negative for classes 0-2, zero for class 3, positive for classes 4-6.
    """
    reps = config.representatives_z if axis == "z" else config.representatives_xy

    signed = cls_index - 3  # [-3, +3]
    sign = 1 if signed >= 0 else -1
    mag_class = abs(signed)  # [0, 3]
    return sign * reps[mag_class]


def class_label(cls_index: int) -> str:
    """Human-readable label for a class index."""
    signed = cls_index - 3
    mag = abs(signed)
    labels = ["aligned", "nudge", "shift", "jump"]
    direction = "" if signed == 0 else ("+" if signed > 0 else "-")
    return f"{direction}{labels[mag]}"


if __name__ == "__main__":
    print("=== XY discretization ===")
    test_offsets = [0.0, 0.001, -0.003, 0.005, -0.008, 0.012, -0.020, 0.030]
    for offset in test_offsets:
        cls = continuous_to_class(offset, axis="x")
        back = class_to_continuous(cls, axis="x")
        label = class_label(cls)
        print(f"  {offset*1000:+7.1f}mm -> class {cls} ({label:>8s}) -> {back*1000:+7.1f}mm")

    print("\n=== Z discretization ===")
    test_offsets = [0.0, 0.005, -0.010, 0.020, -0.030, 0.050, -0.080, 0.100]
    for offset in test_offsets:
        cls = continuous_to_class(offset, axis="z")
        back = class_to_continuous(cls, axis="z")
        label = class_label(cls)
        print(f"  {offset*1000:+7.1f}mm -> class {cls} ({label:>8s}) -> {back*1000:+7.1f}mm")

    print("\n=== All XY classes ===")
    for i in range(7):
        val = class_to_continuous(i, axis="x")
        label = class_label(i)
        print(f"  class {i} = {label:>8s} = {val*1000:+6.1f}mm")

    print("\n=== All Z classes ===")
    for i in range(7):
        val = class_to_continuous(i, axis="z")
        label = class_label(i)
        print(f"  class {i} = {label:>8s} = {val*1000:+6.1f}mm")
