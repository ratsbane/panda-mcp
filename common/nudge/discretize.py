"""
Offset discretization for NUDGE.

3-class scheme: negative / aligned / positive per axis.

XY threshold: 4mm
| Class | Label    | Offset range     |
|-------|----------|------------------|
|   0   | negative | offset < -4mm    |
|   1   | aligned  | |offset| <= 4mm  |
|   2   | positive | offset > +4mm    |

Z threshold: 15mm
| Class | Label    | Offset range     |
|-------|----------|------------------|
|   0   | negative | offset < -15mm   |
|   1   | aligned  | |offset| <= 15mm |
|   2   | positive | offset > +15mm   |

Representative magnitudes: 8mm for XY, 30mm for Z (used for class->continuous).
"""

from dataclasses import dataclass, field
from typing import List

NUM_CLASSES = 3


@dataclass
class DiscretizeConfig:
    """Single threshold per axis separating negative / aligned / positive."""
    threshold_xy: float = 0.004   # 4mm
    threshold_z: float = 0.015    # 15mm

    # Representative magnitude for non-aligned classes (meters)
    representative_xy: float = 0.008   # 8mm
    representative_z: float = 0.030    # 30mm


DEFAULT_CONFIG = DiscretizeConfig()


def continuous_to_class(
    offset_m: float,
    axis: str = "x",
    config: DiscretizeConfig = DEFAULT_CONFIG,
) -> int:
    """
    Convert a continuous offset (meters, signed) to a class index [0, 2].

    Returns:
        0 = negative (needs negative correction)
        1 = aligned (within threshold)
        2 = positive (needs positive correction)
    """
    threshold = config.threshold_z if axis == "z" else config.threshold_xy

    if offset_m < -threshold:
        return 0  # negative
    elif offset_m > threshold:
        return 2  # positive
    else:
        return 1  # aligned


def class_to_continuous(
    cls_index: int,
    axis: str = "x",
    config: DiscretizeConfig = DEFAULT_CONFIG,
) -> float:
    """
    Convert a class index [0, 2] to a representative offset in meters.

    Returns: negative value for class 0, zero for class 1, positive for class 2.
    """
    rep = config.representative_z if axis == "z" else config.representative_xy

    if cls_index == 0:
        return -rep
    elif cls_index == 2:
        return rep
    else:
        return 0.0


def class_label(cls_index: int) -> str:
    """Human-readable label for a class index."""
    labels = ["negative", "aligned", "positive"]
    if 0 <= cls_index < len(labels):
        return labels[cls_index]
    return f"unknown({cls_index})"


if __name__ == "__main__":
    print("=== XY discretization (3-class) ===")
    test_offsets = [0.0, 0.001, -0.003, 0.005, -0.008, 0.012, -0.020, 0.030]
    for offset in test_offsets:
        cls = continuous_to_class(offset, axis="x")
        back = class_to_continuous(cls, axis="x")
        label = class_label(cls)
        print(f"  {offset*1000:+7.1f}mm -> class {cls} ({label:>8s}) -> {back*1000:+7.1f}mm")

    print("\n=== Z discretization (3-class) ===")
    test_offsets = [0.0, 0.005, -0.010, 0.020, -0.030, 0.050, -0.080, 0.100]
    for offset in test_offsets:
        cls = continuous_to_class(offset, axis="z")
        back = class_to_continuous(cls, axis="z")
        label = class_label(cls)
        print(f"  {offset*1000:+7.1f}mm -> class {cls} ({label:>8s}) -> {back*1000:+7.1f}mm")

    print("\n=== All classes ===")
    for i in range(NUM_CLASSES):
        val_xy = class_to_continuous(i, axis="x")
        val_z = class_to_continuous(i, axis="z")
        label = class_label(i)
        print(f"  class {i} = {label:>8s} = XY:{val_xy*1000:+6.1f}mm  Z:{val_z*1000:+6.1f}mm")
