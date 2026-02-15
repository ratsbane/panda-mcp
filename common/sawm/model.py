"""
SAWMNet — Spatial Attention Workspace Model.

MobileNetV3-Small backbone + dual-head that predicts:
  1. (dx, dy) offset from gripper to target in robot-frame meters
  2. predicted crop_scale — an auxiliary distance signal

The crop_scale auxiliary head forces the network to learn an internal
distance model. During an approach, crop_scale should monotonically
decrease. If the predicted scale diverges from the observed scale,
the offset prediction is unreliable.

~1M params, runs ~15fps via ONNX Runtime on Pi CPU.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SAWMNet(nn.Module):
    """
    Predicts gripper-to-target offset (dx, dy) and expected crop_scale.

    Input:
        image: (B, 3, 224, 224) RGB crop
        crop_scale: (B, 1) float in [0, 1] — 0=tight, 1=wide

    Output:
        offset: (B, 2) — (dx, dy) in robot-frame meters
        pred_scale: (B, 1) — predicted crop_scale (auxiliary)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()

        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Keep only the feature extractor (no classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared trunk from visual features + crop_scale
        self.trunk = nn.Sequential(
            nn.Linear(576 + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Offset head: (dx, dy) in meters
        self.offset_head = nn.Linear(128, 2)

        # Scale head: predicted crop_scale (auxiliary supervision)
        self.scale_head = nn.Linear(128, 1)

        # Normalization stats (set from dataset before training)
        self.register_buffer('offset_mean', torch.zeros(2))
        self.register_buffer('offset_std', torch.ones(2))

    def forward(
        self,
        image: torch.Tensor,
        crop_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (B, 3, 224, 224)
            crop_scale: (B, 1)

        Returns:
            offset: (B, 2) normalized offset
            pred_scale: (B, 1) predicted crop_scale
        """
        x = self.features(image)          # (B, 576, 7, 7)
        x = self.pool(x)                  # (B, 576, 1, 1)
        x = x.flatten(1)                  # (B, 576)
        x = torch.cat([x, crop_scale], dim=1)  # (B, 577)
        h = self.trunk(x)                 # (B, 128)

        offset = self.offset_head(h)      # (B, 2)
        pred_scale = self.scale_head(h)   # (B, 1)

        return offset, pred_scale

    def predict(
        self,
        image: torch.Tensor,
        crop_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict offset in meters (denormalized) and predicted scale."""
        offset_norm, pred_scale = self.forward(image, crop_scale)
        offset_m = offset_norm * self.offset_std + self.offset_mean
        return offset_m, pred_scale

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Set output normalization stats from dataset."""
        self.offset_mean.copy_(mean)
        self.offset_std.copy_(std.clamp(min=1e-6))

    @staticmethod
    def param_count() -> str:
        """Quick parameter count without instantiating."""
        m = SAWMNet(pretrained=False)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"{total:,} total, {trainable:,} trainable"


class _SAWMNetDenormalized(nn.Module):
    """Wrapper that bakes denormalization into forward() for ONNX export."""

    def __init__(self, model: SAWMNet):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, crop_scale: torch.Tensor):
        offset_norm, pred_scale = self.model(image, crop_scale)
        offset_m = offset_norm * self.model.offset_std + self.model.offset_mean
        return offset_m, pred_scale


def export_onnx(
    model: SAWMNet,
    output_path: str,
    input_size: Tuple[int, int] = (224, 224),
):
    """Export SAWMNet to ONNX for Pi deployment. Output is in meters."""
    model.eval()
    wrapper = _SAWMNetDenormalized(model)
    wrapper.eval()

    dummy_image = torch.randn(1, 3, *input_size)
    dummy_scale = torch.tensor([[0.5]])

    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_scale),
        output_path,
        input_names=['image', 'crop_scale'],
        output_names=['offset', 'pred_scale'],
        dynamic_axes={
            'image': {0: 'batch'},
            'crop_scale': {0: 'batch'},
            'offset': {0: 'batch'},
            'pred_scale': {0: 'batch'},
        },
        opset_version=13,
    )
    print(f"SAWM model exported to {output_path}")


if __name__ == "__main__":
    print(f"SAWMNet params: {SAWMNet.param_count()}")

    model = SAWMNet(pretrained=False)
    img = torch.randn(2, 3, 224, 224)
    scale = torch.tensor([[0.8], [0.3]])
    offset, pred_s = model(img, scale)
    print(f"Input: image={img.shape}, scale={scale.shape}")
    print(f"Offset: {offset.shape} = {offset.detach()}")
    print(f"Pred scale: {pred_s.shape} = {pred_s.detach()}")
