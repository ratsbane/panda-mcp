"""
NUDGE servo loop v2 -- applies continuous corrections during pick approach.

Loads ONNX model, constructs 4-channel input (RGB + mask) from camera frame
and target bbox, passes gripper position, predicts (dx, dy, dz) offset in mm,
and returns the correction to apply.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NUDGEServoConfig:
    """Configuration for the NUDGE servo loop."""
    # Max correction per step (meters) -- safety clamp
    max_correction_m: float = 0.015  # 15mm

    # Gain: fraction of predicted correction to apply (0-1)
    gain: float = 0.7


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

        # Check if model expects gripper_xyz input
        input_names = [inp.name for inp in self.session.get_inputs()]
        self._has_gripper_input = "gripper_xyz" in input_names

        # Check if model outputs mm values or class indices
        output_names = [out.name for out in self.session.get_outputs()]
        self._regression_mode = "dx_mm" in output_names

        logger.info(
            f"NUDGE servo loaded: {model_path} "
            f"(gripper_input={self._has_gripper_input}, "
            f"regression={self._regression_mode})"
        )

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
        gripper_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Run one prediction step.

        Args:
            frame: BGR camera frame (any size)
            target_bbox_px: Updated bbox [x1,y1,x2,y2] in camera pixels.
            gripper_xyz: Current gripper position (x, y, z) in meters.

        Returns:
            (dx, dy, dz) correction in meters, or None if not active
        """
        if not self._active:
            return None

        bbox = target_bbox_px if target_bbox_px is not None else self._target_bbox
        self._step_count += 1

        # Prepare input: resize to 224x224, construct 4ch
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (224, 224))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = resized.astype(np.float32) / 255.0

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
        img_t = np.transpose(img, (2, 0, 1))
        mask_t = mask[np.newaxis, :, :]
        input_4ch = np.concatenate([img_t, mask_t], axis=0)
        input_batch = input_4ch[np.newaxis, :, :, :].astype(np.float32)

        # Build ONNX inputs
        onnx_inputs = {"image": input_batch}
        if self._has_gripper_input:
            if gripper_xyz is not None:
                xyz_arr = np.array([[gripper_xyz[0], gripper_xyz[1], gripper_xyz[2]]],
                                   dtype=np.float32)
            else:
                xyz_arr = np.zeros((1, 3), dtype=np.float32)
            onnx_inputs["gripper_xyz"] = xyz_arr

        # ONNX inference
        t0 = time.time()
        outputs = self.session.run(None, onnx_inputs)
        dt = time.time() - t0

        if self._regression_mode:
            # Outputs are (dx_mm, dy_mm, dz_mm)
            dx_mm = float(outputs[0][0])
            dy_mm = float(outputs[1][0])
            dz_mm = float(outputs[2][0])

            # Convert mm to meters
            dx = dx_mm / 1000.0
            dy = dy_mm / 1000.0
            dz = dz_mm / 1000.0

            label_str = f"dx={dx_mm:.1f}mm dy={dy_mm:.1f}mm dz={dz_mm:.1f}mm"
        else:
            # Legacy: outputs are class indices
            from .discretize import class_to_continuous, class_label
            cls_x, cls_y, cls_z = int(outputs[0][0]), int(outputs[1][0]), int(outputs[2][0])
            dx = class_to_continuous(cls_x, axis="x")
            dy = class_to_continuous(cls_y, axis="y")
            dz = class_to_continuous(cls_z, axis="z")
            label_str = f"x={class_label(cls_x)} y={class_label(cls_y)} z={class_label(cls_z)}"

        # Apply gain
        dx *= self.config.gain
        dy *= self.config.gain
        dz *= self.config.gain

        # Safety clamp
        max_c = self.config.max_correction_m
        dx = max(-max_c, min(max_c, dx))
        dy = max(-max_c, min(max_c, dy))
        dz = max(-max_c, min(max_c, dz))

        # Check if correction is negligible
        if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dz) < 0.001:
            logger.info(f"NUDGE step {self._step_count}: ALIGNED ({dt*1000:.0f}ms)")
            return None

        self._corrections.append({
            "step": self._step_count,
            "dx_m": round(dx, 5),
            "dy_m": round(dy, 5),
            "dz_m": round(dz, 5),
            "inference_ms": round(dt * 1000, 1),
        })

        logger.info(
            f"NUDGE step {self._step_count}: {label_str} "
            f"-> correction dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm dz={dz*1000:.1f}mm "
            f"({dt*1000:.0f}ms)"
        )

        return (dx, dy, dz)

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "model_path": self._model_path,
            "regression_mode": self._regression_mode,
            "has_gripper_input": self._has_gripper_input,
            "step_count": self._step_count,
            "corrections_applied": len(self._corrections),
            "last_correction": self._corrections[-1] if self._corrections else None,
            "config": {
                "max_correction_m": self.config.max_correction_m,
                "gain": self.config.gain,
            },
        }
