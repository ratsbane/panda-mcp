"""
Level 1: Gripper Localization Model

Predicts end-effector position (x, y, z) from a camera image.
Trained on collected (image, robot_state) pairs.

This is the foundation for visual servoing - knowing where the gripper
is in the image allows computing relative movements to targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GripperLocalizer(nn.Module):
    """
    CNN that predicts gripper 3D position from image.

    Architecture:
    - Backbone: MobileNetV3-Small (efficient for Pi)
    - Head: FC layers to regress (x, y, z)

    Input: RGB image (224x224)
    Output: (x, y, z) in robot base frame
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        output_dim: int = 3,  # x, y, z
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone

        # Load backbone
        if backbone == "mobilenet_v3_small":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            backbone_out = 576  # MobileNetV3-Small output features

            # Remove the classifier
            self.backbone.classifier = nn.Identity()

        elif backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            backbone_out = 1280

            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(backbone_out, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

        # Output normalization parameters (to be set from dataset)
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.ones(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Image tensor (B, 3, 224, 224)

        Returns:
            Position tensor (B, 3) in normalized coordinates
        """
        features = self.backbone(x)
        position = self.head(features)
        return position

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict and denormalize output."""
        normalized = self.forward(x)
        return normalized * self.output_std + self.output_mean

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set output normalization parameters from dataset statistics."""
        self.output_mean = mean
        self.output_std = std


class GripperLocalizerWithGripper(GripperLocalizer):
    """
    Extended model that also predicts gripper width.

    Output: (x, y, z, gripper_width)
    """

    def __init__(self, **kwargs):
        kwargs['output_dim'] = 4
        super().__init__(**kwargs)


class GripperLocalizerWithOrientation(GripperLocalizer):
    """
    Extended model that predicts full 6DOF pose.

    Output: (x, y, z, roll, pitch, yaw)
    """

    def __init__(self, **kwargs):
        kwargs['output_dim'] = 6
        super().__init__(**kwargs)


# Training utilities

def compute_dataset_statistics(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of positions in dataset."""
    positions = []
    for sample in dataset:
        positions.append(sample['position'])

    positions = torch.stack(positions)
    mean = positions.mean(dim=0)
    std = positions.std(dim=0)

    # Prevent division by zero
    std = torch.clamp(std, min=1e-6)

    return mean, std


def create_model(
    backbone: str = "mobilenet_v3_small",
    pretrained: bool = True,
    include_gripper: bool = False,
    include_orientation: bool = False,
) -> GripperLocalizer:
    """Factory function for creating models."""

    if include_orientation:
        return GripperLocalizerWithOrientation(
            backbone=backbone,
            pretrained=pretrained,
        )
    elif include_gripper:
        return GripperLocalizerWithGripper(
            backbone=backbone,
            pretrained=pretrained,
        )
    else:
        return GripperLocalizer(
            backbone=backbone,
            pretrained=pretrained,
        )


# Loss functions

class PositionLoss(nn.Module):
    """Combined loss for position prediction."""

    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights or torch.ones(3)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Can weight different axes differently (e.g., penalize Z errors more).
        """
        diff = pred - target
        weighted_diff = diff * self.weights
        return (weighted_diff ** 2).mean()


# ONNX export for Pi deployment

def export_to_onnx(
    model: GripperLocalizer,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
):
    """Export model to ONNX format for Pi deployment."""
    model.eval()

    dummy_input = torch.randn(1, 3, *input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['position'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'position': {0: 'batch_size'},
        },
        opset_version=11,
    )

    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model: {model.backbone_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
