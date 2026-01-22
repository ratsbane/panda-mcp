# Visuomotor Policy Models

This directory contains model architectures for learning visuomotor control
from collected training data.

## Architecture Overview

### Level 1: Gripper Localization

A simple model that predicts the end-effector position from an image.
This is trained on our collected (image, robot_state) pairs.

```
Image (640x480) → CNN Backbone → FC Layers → (x, y, z) position
```

**Use case**: Given a camera view, where is my gripper?

### Level 2: Visual Servoing Policy

A model that takes (current_image, target_location) and predicts actions.

```
Current Image + Target (pixel coords) → CNN → Action (dx, dy, dz)
```

**Use case**: Move the gripper toward pixel location (320, 240)

### Level 3: Goal-Conditioned Policy (VLA-style)

A model that takes (image, goal_description) and predicts action sequences.

```
Image + "pick up the red block" → Vision-Language Model → Action Sequence
```

**Use case**: High-level commands like "grasp the elephant"

## Training Strategy

### Phase 1: Supervised Learning
- Collect diverse (image, robot_state) pairs
- Train Level 1 model to predict gripper position
- Validate on held-out test set

### Phase 2: Behavioral Cloning
- Collect demonstration trajectories
- Train Level 2 model on (start, goal, action) tuples
- Add data augmentation for robustness

### Phase 3: Fine-tuning (optional)
- Use reinforcement learning to improve beyond demonstrations
- Define reward based on task success

## Model Files

- `gripper_localizer.py` - Level 1 model
- `visual_servo.py` - Level 2 model
- `policy_vla.py` - Level 3 model (future)

## Hardware Requirements

- Training: GPU with 8GB+ VRAM (can use cloud)
- Inference: Raspberry Pi 5 (with optimizations)

## Inference Optimization

For Pi deployment:
- Use ONNX export
- Quantize to INT8
- Use smaller backbone (MobileNet, EfficientNet-Lite)
- Target <100ms inference time
