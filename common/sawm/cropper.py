"""
Progressive crop computation for SAWM.

As the gripper approaches the target, the crop dynamically tightens:
- >15cm away: 80% of frame (wide context)
- 5-15cm: linear interpolation 40-80%
- <5cm: 20-40% (pixel-level precision)

Crop is centered on the midpoint between gripper and target pixel positions,
ensuring both are always visible.
"""

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Calibration file path
CALIBRATION_PATH = "/tmp/aruco_calibration.npz"


class ProgressiveCropper:
    """Computes progressive crops for SAWM data collection and inference."""

    # Crop fraction of frame at various distances
    WIDE_FRACTION = 0.80   # >15cm
    NARROW_FRACTION = 0.20  # <5cm
    FAR_DIST = 0.15         # meters
    NEAR_DIST = 0.05        # meters

    OUTPUT_SIZE = 224  # pixels

    def __init__(self, calibration_path: str = CALIBRATION_PATH):
        self._H_inv = None  # robot -> pixel (inverse homography)
        self._H = None       # pixel -> robot (homography)
        self._frame_shape = None  # (H, W) of camera frame
        self._load_calibration(calibration_path)

    def _load_calibration(self, path: str):
        """Load homography from ArUco calibration file."""
        try:
            data = np.load(path, allow_pickle=True)
            H = data["H"]  # pixel -> robot
            self._H = H
            self._H_inv = np.linalg.inv(H)  # robot -> pixel
            logger.info(f"SAWM cropper loaded calibration from {path}")
        except Exception as e:
            logger.warning(f"Could not load calibration from {path}: {e}")

    def robot_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert robot (x, y) to pixel coordinates via inverse homography."""
        if self._H_inv is None:
            raise RuntimeError("No calibration loaded")

        import cv2
        pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(pt, self._H_inv)
        return int(px[0, 0, 0]), int(px[0, 0, 1])

    def pixel_to_robot(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel to robot (x, y) via homography."""
        if self._H is None:
            raise RuntimeError("No calibration loaded")

        import cv2
        pt = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        robot = cv2.perspectiveTransform(pt, self._H)
        return float(robot[0, 0, 0]), float(robot[0, 0, 1])

    def compute_crop_scale(self, distance_m: float) -> float:
        """
        Compute crop scale from gripper-to-target distance.

        Returns:
            scale in [0, 1] — 0 = tightest crop, 1 = widest
        """
        if distance_m >= self.FAR_DIST:
            return 1.0
        elif distance_m <= self.NEAR_DIST:
            return 0.0
        else:
            # Linear interpolation
            t = (distance_m - self.NEAR_DIST) / (self.FAR_DIST - self.NEAR_DIST)
            return t

    def _crop_fraction(self, scale: float) -> float:
        """Convert scale [0,1] to fraction of frame to crop."""
        return self.NARROW_FRACTION + scale * (self.WIDE_FRACTION - self.NARROW_FRACTION)

    def compute_crop(
        self,
        frame: np.ndarray,
        target_robot_xy: Tuple[float, float],
        gripper_robot_xy: Tuple[float, float],
    ) -> Tuple[np.ndarray, float]:
        """
        Compute a progressive crop centered between gripper and target.

        Args:
            frame: Full camera frame (H, W, 3) BGR
            target_robot_xy: Target position in robot frame (meters)
            gripper_robot_xy: Gripper position in robot frame (meters)

        Returns:
            (crop_224x224, crop_scale) — the resized crop and its scale value
        """
        import cv2

        h, w = frame.shape[:2]
        self._frame_shape = (h, w)

        # Distance in robot frame
        dx = target_robot_xy[0] - gripper_robot_xy[0]
        dy = target_robot_xy[1] - gripper_robot_xy[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # Crop scale and fraction
        scale = self.compute_crop_scale(distance)
        fraction = self._crop_fraction(scale)

        # Convert both positions to pixels
        try:
            target_px = self.robot_to_pixel(*target_robot_xy)
            gripper_px = self.robot_to_pixel(*gripper_robot_xy)
        except RuntimeError:
            # No calibration — use center of frame as fallback
            logger.warning("No calibration for crop, using frame center")
            crop = cv2.resize(frame, (self.OUTPUT_SIZE, self.OUTPUT_SIZE))
            return crop, scale

        # Crop center: midpoint between gripper and target pixels
        cx = (target_px[0] + gripper_px[0]) // 2
        cy = (target_px[1] + gripper_px[1]) // 2

        # Crop size (square)
        crop_size = int(max(w, h) * fraction)

        # Ensure both gripper and target are inside the crop (with margin)
        margin = 20  # pixels
        min_crop = max(
            abs(target_px[0] - gripper_px[0]) + 2 * margin,
            abs(target_px[1] - gripper_px[1]) + 2 * margin,
        )
        crop_size = max(crop_size, min_crop)

        # Clamp crop to frame bounds
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)

        # Adjust if we hit a boundary
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)

        crop = frame[y1:y2, x1:x2]

        # Resize to model input size
        if crop.size == 0:
            logger.warning("Empty crop, using full frame")
            crop = frame

        crop = cv2.resize(crop, (self.OUTPUT_SIZE, self.OUTPUT_SIZE))

        return crop, scale

    def compute_crop_at_distance(
        self,
        frame: np.ndarray,
        target_robot_xy: Tuple[float, float],
        gripper_robot_xy: Tuple[float, float],
        override_distance: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Like compute_crop but also returns debug info.

        Returns:
            (crop_224x224, scale, debug_info)
        """
        import cv2

        h, w = frame.shape[:2]

        dx = target_robot_xy[0] - gripper_robot_xy[0]
        dy = target_robot_xy[1] - gripper_robot_xy[1]
        distance = override_distance or np.sqrt(dx * dx + dy * dy)

        crop, scale = self.compute_crop(frame, target_robot_xy, gripper_robot_xy)

        debug = {
            "distance_m": round(float(distance), 4),
            "scale": round(float(scale), 4),
            "fraction": round(float(self._crop_fraction(scale)), 4),
            "frame_size": (w, h),
        }

        try:
            debug["target_px"] = self.robot_to_pixel(*target_robot_xy)
            debug["gripper_px"] = self.robot_to_pixel(*gripper_robot_xy)
        except RuntimeError:
            pass

        return crop, scale, debug


# Singleton
_cropper: Optional[ProgressiveCropper] = None


def get_cropper() -> ProgressiveCropper:
    global _cropper
    if _cropper is None:
        _cropper = ProgressiveCropper()
    return _cropper
