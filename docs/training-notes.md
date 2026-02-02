# Gripper Localizer Training Notes

## Overview

Training a CNN to predict gripper (x, y, z) position from RGB camera images.
Goal: Enable visual servoing and object manipulation without explicit position sensing.

## Dataset v1: Single Viewpoint (2026-02-02)

- **Location**: `~/datasets/gripper_v1/20260202_081430/`
- **Samples**: 6,551 images
- **Collection method**: Random poses in workspace
- **Camera**: Fixed position USB camera
- **Lighting**: Consistent

### Training Results

All models converged to ~200mm error, which equals predicting the mean position.
This confirms insufficient information from a single fixed viewpoint.

| Backbone | Parameters | Best Val Error | Checkpoint |
|----------|------------|----------------|------------|
| mobilenetv3_small_100 | ~2.5M | 200.83mm | `spark:~/runs/mobilenetv3_small_100_20260202_163344/` |
| mobilenetv3_large_100 | ~5.4M | 201.91mm | `spark:~/runs/mobilenetv3_large_100_20260202_164248/` |
| efficientnet_b0 | ~5.3M | 202.36mm | `spark:~/runs/efficientnet_b0_20260202_165612/` |
| efficientnet_b2 | ~9.1M | 202.66mm | `spark:~/runs/efficientnet_b2_20260202_171652/` |

### Key Findings

1. **Model capacity doesn't help**: Smallest model (MobileNetV3-Small) performed best
2. **Data limitation confirmed**: All models predict near-mean position (~200mm error)
3. **Spatial augmentation removed**: Flip/rotate/crop breaks position labels; only color jitter used

## Dataset v2: Multi-Viewpoint (2026-02-02, in progress)

- **Location**: `~/datasets/gripper_v2/multi_viewpoint/`
- **Target samples**: 3,000 images
- **Collection method**: Random poses with camera moved during collection
- **Camera**: USB camera moved to multiple positions
- **Lighting**: Varied during collection

### Hypothesis

Multiple viewpoints should provide:
- Parallax information for depth estimation
- More robust features that generalize across camera positions
- Better 3D understanding vs memorizing single viewpoint

## Training Infrastructure

- **Collection**: Raspberry Pi 5 + Franka Panda
- **Training**: Nvidia DGX Spark (GB10 GPU, 119GB RAM)
- **Framework**: PyTorch 2.10 + timm

## Next Steps

1. Complete multi-viewpoint dataset collection
2. Train on combined v1 + v2 datasets
3. Integrate PhotoNeo depth camera for ground truth
4. Develop grasp prediction model

## Files

- `scripts/train_gripper_localizer.py` - Training script
- `scripts/train_all_backbones.py` - Multi-backbone comparison
- `scripts/collect_data.py` - Data collection script
- `scripts/infer_gripper_position.py` - Inference script
