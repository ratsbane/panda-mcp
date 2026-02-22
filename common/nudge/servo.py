"""
NUDGE servo loop -- applies discrete corrections during pick/place approach.

Loads ONNX model, constructs 4-channel input (RGB + mask) from camera frame
and target bbox, predicts correction classes, converts to meters, and returns
the correction to apply.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .discretize import class_to_continuous, class_label, DEFAULT_CONFIG, DiscretizeConfig

logger = logging.getLogger(__name__)


@dataclass
class NUDGEServoConfig:
    """Configuration for the NUDGE servo loop."""
    # Max correction per step (meters) -- safety clamp
    max_correction_m: float = 0.015  # 15mm

    # Gain: fraction of predicted correction to apply (0-1)
    gain: float = 0.7

    # Discretization config
    discretize: DiscretizeConfig = field(default_factory=DiscretizeConfig)


class NUDGEServoLoop:
    """Applies NUDGE corrections during approach using ONNX inference."""

    def __init__(
        self,
        model_path: str,
        config: NUDGEServoConfig = NUDGEServoConfig(),
    ):
        import onnxruntime as ort

        self.config = config
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._model_path = model_path
        self._active = False
        self._target_bbox: Optional[List[float]] = None
        self._step_count = 0
        self._corrections: list[dict] = []

        logger.info(f"NUDGE servo loaded: {model_path}")

    @property
    def active(self) -> bool:
        return self._active

    def start(self, target_bbox_pixels: List[float]):
        """Begin a servo-guided approach."""
        self._active = True
        self._target_bbox = list(target_bbox_pixels)
        self._step_count = 0
        self._corrections = []
        logger.info(f"NUDGE servo started, bbox={[round(b,1) for b in target_bbox_pixels]}")

    def stop(self):
        """End the servo approach."""
        self._active = False
        logger.info(
            f"NUDGE servo stopped after {self._step_count} steps, "
            f"{len(self._corrections)} corrections applied"
        )

    def predict(
        self,
        frame: np.ndarray,
        target_bbox_px: Optional[List[float]] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Run one prediction step.

        Args:
            frame: BGR camera frame (any size)
            target_bbox_px: Updated bbox [x1,y1,x2,y2] in camera pixels. Uses initial if None.

        Returns:
            (dx, dy, dz) correction in meters, or None if not active / prediction is "aligned"
        """
        if not self._active:
            return None

        bbox = target_bbox_px if target_bbox_px is not None else self._target_bbox
        self._step_count += 1

        # Prepare input: resize to 224x224, construct 4ch
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (224, 224))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = resized.astype(np.float32) / 255.0  # [0, 1]

        # Scale bbox to 224x224
        sx, sy = 224.0 / w, 224.0 / h
        bbox_224 = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]

        # Binary mask
        mask = np.zeros((224, 224), dtype=np.float32)
        x1 = max(0, int(bbox_224[0]))
        y1 = max(0, int(bbox_224[1]))
        x2 = min(224, int(bbox_224[2]))
        y2 = min(224, int(bbox_224[3]))
        mask[y1:y2, x1:x2] = 1.0

        # Stack: (1, 4, 224, 224)
        img_t = np.transpose(img, (2, 0, 1))  # (3, 224, 224)
        mask_t = mask[np.newaxis, :, :]        # (1, 224, 224)
        input_4ch = np.concatenate([img_t, mask_t], axis=0)  # (4, 224, 224)
        input_batch = input_4ch[np.newaxis, :, :, :].astype(np.float32)

        # ONNX inference
        t0 = time.time()
        outputs = self.session.run(None, {"image": input_batch})
        dt = time.time() - t0

        cls_x, cls_y, cls_z = int(outputs[0][0]), int(outputs[1][0]), int(outputs[2][0])

        # Convert classes to meters
        dx = class_to_continuous(cls_x, self.config.discretize)
        dy = class_to_continuous(cls_y, self.config.discretize)
        dz = class_to_continuous(cls_z, self.config.discretize)

        # Apply gain
        dx *= self.config.gain
        dy *= self.config.gain
        dz *= self.config.gain

        # Safety clamp
        max_c = self.config.max_correction_m
        dx = max(-max_c, min(max_c, dx))
        dy = max(-max_c, min(max_c, dy))
        dz = max(-max_c, min(max_c, dz))

        # Check if all aligned (class 3 = zero)
        if cls_x == 3 and cls_y == 3 and cls_z == 3:
            logger.info(f"NUDGE step {self._step_count}: ALIGNED ({dt*1000:.0f}ms)")
            return None  # No correction needed

        self._corrections.append({
            "step": self._step_count,
            "cls": [cls_x, cls_y, cls_z],
            "dx_m": round(dx, 5),
            "dy_m": round(dy, 5),
            "dz_m": round(dz, 5),
            "inference_ms": round(dt * 1000, 1),
        })

        logger.info(
            f"NUDGE step {self._step_count}: "
            f"x={class_label(cls_x)} y={class_label(cls_y)} z={class_label(cls_z)} "
            f"-> dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm dz={dz*1000:.1f}mm ({dt*1000:.0f}ms)"
        )

        return (dx, dy, dz)

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "model_path": self._model_path,
            "step_count": self._step_count,
            "corrections_applied": len(self._corrections),
            "last_correction": self._corrections[-1] if self._corrections else None,
            "config": {
                "max_correction_m": self.config.max_correction_m,
                "gain": self.config.gain,
            },
        }
