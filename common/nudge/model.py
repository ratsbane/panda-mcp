"""
NUDGENet v2 -- CNN + gripper position → continuous offset regression.

The key insight: the camera is static and the target mask doesn't change
between frames of an approach. Without gripper position, the model sees
identical inputs for different labels and can only learn the average bias.

Architecture:
- 4 conv blocks extract visual features (where is the block?)
- Gripper XYZ provides current arm position (where is the gripper going?)
- Trunk fuses both to predict (dx, dy, dz) in mm

Input:
    image: (B, 4, 224, 224) -- RGB [0,1] + binary target mask
    gripper_xyz: (B, 3) -- gripper position in robot frame (meters)

Output:
    offsets: (B, 3) -- predicted (dx, dy, dz) correction in millimeters
"""

import torch
import torch.nn as nn
from typing import Tuple


class NUDGENet(nn.Module):
    """Predicts continuous (dx, dy, dz) corrections in millimeters."""

    NUM_CLASSES = 3  # kept for compat with dataset/training code

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x4 -> 56x56x24
            nn.Conv2d(4, 24, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            # Block 2: 56x56x24 -> 14x14x48
            nn.Conv2d(24, 48, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            # Block 3: 14x14x48 -> 7x7x64
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fuse CNN features (64) + gripper position (3)
        self.trunk = nn.Sequential(
            nn.Linear(64 + 3, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # Single regression head: (dx, dy, dz) in mm
        self.head = nn.Linear(16, 3)

    def forward(
        self, image: torch.Tensor, gripper_xyz: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 4, 224, 224)
            gripper_xyz: (B, 3) gripper position in meters. If None, uses zeros.

        Returns:
            pred_x, pred_y, pred_z: each (B, 1) offset in mm
            (returned as 3 separate tensors for API compat with training code)
        """
        vis = self.features(image)
        vis = self.pool(vis).flatten(1)  # (B, 64)

        if gripper_xyz is None:
            gripper_xyz = torch.zeros(vis.shape[0], 3, device=vis.device)

        # Concatenate visual features + gripper position
        fused = torch.cat([vis, gripper_xyz], dim=1)  # (B, 67)
        h = self.trunk(fused)  # (B, 16)

        out = self.head(h)  # (B, 3) -- dx, dy, dz in mm
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def predict(self, image: torch.Tensor, gripper_xyz: torch.Tensor = None) -> Tuple[float, float, float]:
        """Predict continuous offsets for a single sample (no batch)."""
        self.eval()
        with torch.no_grad():
            if gripper_xyz is not None:
                gripper_xyz = gripper_xyz.unsqueeze(0)
            dx, dy, dz = self.forward(image.unsqueeze(0), gripper_xyz)
            return dx.item(), dy.item(), dz.item()

    @staticmethod
    def param_count() -> str:
        m = NUDGENet()
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"{total:,} total, {trainable:,} trainable"


class _NUDGENetDeploy(nn.Module):
    """Wrapper for ONNX deployment -- outputs mm offsets as 3 scalars."""

    def __init__(self, model: "NUDGENet"):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, gripper_xyz: torch.Tensor):
        dx, dy, dz = self.model(image, gripper_xyz)
        return dx.squeeze(1), dy.squeeze(1), dz.squeeze(1)


def export_onnx(
    model: "NUDGENet",
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
):
    """Export NUDGENet to ONNX with regression outputs (mm offsets)."""
    model.eval()
    wrapper = _NUDGENetDeploy(model)
    wrapper.eval()

    dummy_img = torch.randn(1, 4, *input_size)
    dummy_xyz = torch.randn(1, 3)

    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_xyz),
        output_path,
        input_names=["image", "gripper_xyz"],
        output_names=["dx_mm", "dy_mm", "dz_mm"],
        dynamic_axes={
            "image": {0: "batch"},
            "gripper_xyz": {0: "batch"},
            "dx_mm": {0: "batch"},
            "dy_mm": {0: "batch"},
            "dz_mm": {0: "batch"},
        },
        opset_version=13,
    )
    print(f"NUDGE model exported to {output_path}")


if __name__ == "__main__":
    print(f"NUDGENet params: {NUDGENet.param_count()}")

    model = NUDGENet()
    img = torch.randn(2, 4, 224, 224)
    xyz = torch.randn(2, 3)
    dx, dy, dz = model(img, xyz)
    print(f"Input: img={img.shape}, xyz={xyz.shape}")
    print(f"Outputs: dx={dx.shape}, dy={dy.shape}, dz={dz.shape}")
    print(f"Values: dx={dx.squeeze().tolist()}, dy={dy.squeeze().tolist()}, dz={dz.squeeze().tolist()}")
