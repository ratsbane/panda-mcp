"""
SAWM inference monitor — runs in background during pick_at().

Loads an ONNX model and predicts gripper-to-target offsets from
progressive crops. When the predicted error exceeds a distance-dependent
threshold, it signals a correction.

Threading pattern reused from franka_mcp/vla_client.py.
"""

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .cropper import get_cropper

logger = logging.getLogger(__name__)

# Threshold for triggering correction (meters)
# Far away: tolerate larger error; close: tighter
FAR_THRESHOLD = 0.020   # 2cm when >10cm away
NEAR_THRESHOLD = 0.005  # 5mm when <5cm away
FAR_DIST = 0.10
NEAR_DIST = 0.05


def _correction_threshold(distance_m: float) -> float:
    """Distance-dependent threshold for triggering corrections."""
    if distance_m >= FAR_DIST:
        return FAR_THRESHOLD
    elif distance_m <= NEAR_DIST:
        return NEAR_THRESHOLD
    else:
        t = (distance_m - NEAR_DIST) / (FAR_DIST - NEAR_DIST)
        return NEAR_THRESHOLD + t * (FAR_THRESHOLD - NEAR_THRESHOLD)


class SAWMMonitor:
    """
    Background thread that predicts gripper-to-target offsets during approach.

    Usage:
        monitor = SAWMMonitor("sawm.onnx")
        monitor.start(target_xy=(0.4, -0.1))
        ...
        correction = monitor.get_correction()  # non-blocking
        if correction:
            dx, dy = correction
            # adjust waypoint
        ...
        monitor.stop()
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAWM ONNX model: {e}")

        self._cropper = get_cropper()

        # State
        self._active = False
        self._target_xy: Optional[Tuple[float, float]] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Latest prediction
        self._correction: Optional[Tuple[float, float]] = None
        self._last_prediction: Optional[dict] = None
        self._prediction_count = 0

        # ImageNet normalization
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def active(self) -> bool:
        return self._active

    def start(self, target_xy: Tuple[float, float]):
        """Start monitoring for a pick approach."""
        if self._active:
            self.stop()

        self._target_xy = target_xy
        self._correction = None
        self._last_prediction = None
        self._prediction_count = 0
        self._stop_event.clear()
        self._active = True

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        logger.info(f"SAWM monitor started for target=({target_xy[0]:.3f}, {target_xy[1]:.3f})")

    def stop(self):
        """Stop monitoring."""
        self._active = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        logger.info(f"SAWM monitor stopped ({self._prediction_count} predictions)")

    def get_correction(self) -> Optional[Tuple[float, float]]:
        """
        Get correction if error exceeds threshold. Non-blocking.

        Returns:
            (dx, dy) in meters to correct, or None if no correction needed.
        """
        with self._lock:
            correction = self._correction
            self._correction = None  # consume it
            return correction

    def get_status(self) -> dict:
        """Get monitor status."""
        result = {
            "active": self._active,
            "predictions": self._prediction_count,
        }
        with self._lock:
            if self._last_prediction:
                result["last_prediction"] = self._last_prediction
            if self._target_xy:
                result["target_xy"] = list(self._target_xy)
        return result

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for ONNX inference."""
        # crop is 224x224 BGR uint8
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img[np.newaxis]  # add batch dim

    def _predict(self, crop: np.ndarray, scale: float) -> Tuple[float, float, float]:
        """
        Run ONNX inference.

        Returns:
            (dx, dy, pred_scale) — offset in meters + predicted crop_scale
        """
        image_input = self._preprocess(crop)
        scale_input = np.array([[scale]], dtype=np.float32)

        outputs = self._session.run(None, {
            "image": image_input,
            "crop_scale": scale_input,
        })

        dx, dy = outputs[0][0]
        pred_scale = float(outputs[1][0][0])
        return float(dx), float(dy), pred_scale

    def _monitor_loop(self):
        """Background thread: capture frames, predict offsets, signal corrections."""
        from camera_daemon.client import get_camera_client
        camera = get_camera_client()

        interval = 0.2  # ~5Hz prediction rate

        while self._active and not self._stop_event.is_set():
            try:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(interval)
                    continue

                # Get current gripper position (import here to avoid circular)
                from franka_mcp.controller import get_controller
                controller = get_controller()
                if not controller.connected:
                    time.sleep(interval)
                    continue

                state = controller.get_state()
                gripper_xy = (state.ee_position[0], state.ee_position[1])

                # Compute progressive crop
                crop, scale = self._cropper.compute_crop(
                    frame, self._target_xy, gripper_xy
                )

                # Predict offset + scale
                dx, dy, pred_scale = self._predict(crop, scale)
                self._prediction_count += 1

                error = np.sqrt(dx * dx + dy * dy)

                # Scale divergence: how far the predicted scale is from actual
                # High divergence = model is confused about distance = low confidence
                scale_divergence = abs(pred_scale - scale)
                confident = scale_divergence < 0.10  # trust if within 0.10

                # Distance from gripper to target
                dist_to_target = np.sqrt(
                    (self._target_xy[0] - gripper_xy[0]) ** 2 +
                    (self._target_xy[1] - gripper_xy[1]) ** 2
                )

                threshold = _correction_threshold(dist_to_target)

                pred_info = {
                    "dx_mm": round(dx * 1000, 1),
                    "dy_mm": round(dy * 1000, 1),
                    "error_mm": round(error * 1000, 1),
                    "threshold_mm": round(threshold * 1000, 1),
                    "distance_mm": round(dist_to_target * 1000, 1),
                    "scale": round(scale, 3),
                    "pred_scale": round(pred_scale, 3),
                    "scale_divergence": round(scale_divergence, 3),
                    "confident": confident,
                    "triggered": error > threshold and confident,
                }

                with self._lock:
                    self._last_prediction = pred_info
                    if error > threshold and confident:
                        self._correction = (dx, dy)
                        logger.info(
                            f"SAWM correction: dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm "
                            f"(err={error*1000:.1f}mm > thr={threshold*1000:.1f}mm, "
                            f"scale={scale:.3f} pred={pred_scale:.3f})"
                        )
                    elif error > threshold and not confident:
                        logger.debug(
                            f"SAWM skip: error {error*1000:.1f}mm but scale divergence "
                            f"{scale_divergence:.3f} too high (pred={pred_scale:.3f} vs actual={scale:.3f})"
                        )

            except Exception as e:
                logger.warning(f"SAWM monitor error: {e}")

            time.sleep(interval)
