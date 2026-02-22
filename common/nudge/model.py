"""
NUDGENet -- ab initio CNN for discrete spatial corrections.

4-channel input (RGB + binary target mask), 5 conv blocks, 3 classification
heads predicting {-3..+3} per axis. ~300K params, runs on Pi CPU or Hailo.
"""

import torch
import torch.nn as nn
from typing import Tuple


class NUDGENet(nn.Module):
    """
    Predicts discrete correction classes for x, y, z axes.

    Input:
        image: (B, 4, 224, 224) -- RGB [0,1] + binary target mask

    Output:
        logits_x: (B, 7) -- classes {-3, -2, -1, 0, +1, +2, +3}
        logits_y: (B, 7)
        logits_z: (B, 7)
    """

    NUM_CLASSES = 7  # {-3, -2, -1, 0, +1, +2, +3}

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x4 -> 112x112x16
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2: 112x112x16 -> 56x56x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 56x56x32 -> 28x28x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 28x28x64 -> 14x14x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 5: 14x14x128 -> 7x7x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.trunk = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_x = nn.Linear(64, self.NUM_CLASSES)
        self.head_y = nn.Linear(64, self.NUM_CLASSES)
        self.head_z = nn.Linear(64, self.NUM_CLASSES)

    def forward(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 4, 224, 224)

        Returns:
            logits_x, logits_y, logits_z: each (B, 7)
        """
        x = self.features(image)
        x = self.pool(x).flatten(1)  # (B, 128)
        h = self.trunk(x)            # (B, 64)
        return self.head_x(h), self.head_y(h), self.head_z(h)

    def predict(self, image: torch.Tensor) -> Tuple[int, int, int]:
        """Predict argmax class for each axis (single sample, no batch)."""
        self.eval()
        with torch.no_grad():
            lx, ly, lz = self.forward(image.unsqueeze(0))
            return lx.argmax(1).item(), ly.argmax(1).item(), lz.argmax(1).item()

    @staticmethod
    def param_count() -> str:
        m = NUDGENet()
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"{total:,} total, {trainable:,} trainable"


class _NUDGENetArgmax(nn.Module):
    """Wrapper that outputs argmax class indices for ONNX deployment."""

    def __init__(self, model: "NUDGENet"):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        lx, ly, lz = self.model(image)
        return lx.argmax(1), ly.argmax(1), lz.argmax(1)


def export_onnx(
    model: "NUDGENet",
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
):
    """Export NUDGENet to ONNX with argmax outputs (class indices)."""
    model.eval()
    wrapper = _NUDGENetArgmax(model)
    wrapper.eval()

    dummy = torch.randn(1, 4, *input_size)

    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        input_names=["image"],
        output_names=["class_x", "class_y", "class_z"],
        dynamic_axes={
            "image": {0: "batch"},
            "class_x": {0: "batch"},
            "class_y": {0: "batch"},
            "class_z": {0: "batch"},
        },
        opset_version=13,
    )
    print(f"NUDGE model exported to {output_path}")


if __name__ == "__main__":
    print(f"NUDGENet params: {NUDGENet.param_count()}")

    model = NUDGENet()
    img = torch.randn(2, 4, 224, 224)
    lx, ly, lz = model(img)
    print(f"Input: {img.shape}")
    print(f"Logits x: {lx.shape}, y: {ly.shape}, z: {lz.shape}")
    print(f"Predictions: x={lx.argmax(1).tolist()}, y={ly.argmax(1).tolist()}, z={lz.argmax(1).tolist()}")
