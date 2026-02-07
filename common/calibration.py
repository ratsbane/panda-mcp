"""
Camera-to-robot calibration utilities.

Provides coordinate transformations between image pixels and robot workspace.
This requires calibration for accurate results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class CalibrationData:
    """
    Camera calibration data for pixel-to-world transformation.

    The simplest calibration uses a homography matrix for a planar workspace.
    For more accuracy, full camera intrinsics + extrinsics would be needed.
    """

    # Image dimensions
    image_width: int = 1280
    image_height: int = 720

    # Workspace bounds in robot coordinates (meters)
    # These define the rectangle visible in the camera
    workspace_x_min: float = 0.2
    workspace_x_max: float = 0.6
    workspace_y_min: float = -0.2
    workspace_y_max: float = 0.2
    workspace_z: float = 0.05  # Height of the workspace plane (table)

    # Homography matrix (3x3) for pixel to world transform
    # None means uncalibrated - will use simple linear interpolation
    homography: Optional[np.ndarray] = None

    # Calibration points: list of (pixel_x, pixel_y, robot_x, robot_y)
    calibration_points: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "workspace": {
                "x_min": self.workspace_x_min,
                "x_max": self.workspace_x_max,
                "y_min": self.workspace_y_min,
                "y_max": self.workspace_y_max,
                "z": self.workspace_z,
            },
            "homography": self.homography.tolist() if self.homography is not None else None,
            "calibration_points": self.calibration_points,
            "is_calibrated": self.homography is not None,
        }

    def save(self, path: str):
        """Save calibration to file."""
        data = self.to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_npz(cls, path: str, workspace_z: float = 0.013) -> "CalibrationData":
        """Load calibration from ArUco NPZ file (as saved by ArUco calibration script)."""
        data = np.load(path, allow_pickle=True)
        H = data["H"]
        pixel_pts = data["pixel_pts"]
        robot_pts = data["robot_pts"]
        table_z = float(data.get("table_z", workspace_z))

        cal = cls(workspace_z=table_z)
        cal.homography = H

        # Store calibration points
        for i in range(len(pixel_pts)):
            cal.calibration_points.append((
                float(pixel_pts[i][0]), float(pixel_pts[i][1]),
                float(robot_pts[i][0]), float(robot_pts[i][1]),
            ))

        return cal

    @classmethod
    def load(cls, path: str) -> "CalibrationData":
        """Load calibration from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        cal = cls(
            image_width=data["image_width"],
            image_height=data["image_height"],
            workspace_x_min=data["workspace"]["x_min"],
            workspace_x_max=data["workspace"]["x_max"],
            workspace_y_min=data["workspace"]["y_min"],
            workspace_y_max=data["workspace"]["y_max"],
            workspace_z=data["workspace"]["z"],
            calibration_points=data.get("calibration_points", []),
        )

        if data.get("homography"):
            cal.homography = np.array(data["homography"])

        return cal


class CoordinateTransformer:
    """
    Transforms between image pixel coordinates and robot workspace coordinates.
    """

    def __init__(self, calibration: Optional[CalibrationData] = None):
        self.calibration = calibration or CalibrationData()

    def pixel_to_robot(
        self,
        pixel_x: int,
        pixel_y: int,
    ) -> tuple[float, float, float]:
        """
        Convert image pixel coordinates to robot coordinates.

        Args:
            pixel_x: X coordinate in image (0 = left)
            pixel_y: Y coordinate in image (0 = top)

        Returns:
            (x, y, z) in robot base frame (meters)
        """
        cal = self.calibration

        if cal.homography is not None:
            # Use homography for accurate transformation
            point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point, cal.homography)
            robot_x, robot_y = transformed[0, 0]
        else:
            # Simple linear interpolation (uncalibrated estimate)
            # Assumes camera is roughly looking down at the workspace

            # Normalize pixel coordinates to 0-1
            norm_x = pixel_x / cal.image_width
            norm_y = pixel_y / cal.image_height

            # Map to workspace (note: image Y is typically inverted)
            # Also note: image X often maps to robot Y (left-right)
            # This mapping depends on camera orientation!

            # Assuming camera is behind robot looking forward:
            # - Image left-right (x) maps to robot left-right (y)
            # - Image top-bottom (y) maps to robot forward-back (x)

            robot_y = cal.workspace_y_max - norm_x * (cal.workspace_y_max - cal.workspace_y_min)
            robot_x = cal.workspace_x_min + norm_y * (cal.workspace_x_max - cal.workspace_x_min)

        return (robot_x, robot_y, cal.workspace_z)

    def robot_to_pixel(
        self,
        robot_x: float,
        robot_y: float,
    ) -> tuple[int, int]:
        """
        Convert robot coordinates to image pixel coordinates.

        Args:
            robot_x: X in robot frame (meters)
            robot_y: Y in robot frame (meters)

        Returns:
            (pixel_x, pixel_y) in image
        """
        cal = self.calibration

        if cal.homography is not None:
            # Invert homography
            inv_homography = np.linalg.inv(cal.homography)
            point = np.array([[robot_x, robot_y]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(point, inv_homography)
            pixel_x, pixel_y = transformed[0, 0]
        else:
            # Simple linear interpolation (inverse of pixel_to_robot)
            norm_y = (robot_x - cal.workspace_x_min) / (cal.workspace_x_max - cal.workspace_x_min)
            norm_x = (cal.workspace_y_max - robot_y) / (cal.workspace_y_max - cal.workspace_y_min)

            pixel_x = norm_x * cal.image_width
            pixel_y = norm_y * cal.image_height

        return (int(pixel_x), int(pixel_y))

    def add_calibration_point(
        self,
        pixel_x: int,
        pixel_y: int,
        robot_x: float,
        robot_y: float,
    ):
        """Add a calibration point (pixel coords + corresponding robot coords)."""
        self.calibration.calibration_points.append(
            (pixel_x, pixel_y, robot_x, robot_y)
        )

    def compute_homography(self) -> bool:
        """
        Compute homography matrix from calibration points.
        Requires at least 4 points.

        Returns:
            True if successful
        """
        points = self.calibration.calibration_points

        if len(points) < 4:
            return False

        src_points = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        dst_points = np.array([(p[2], p[3]) for p in points], dtype=np.float32)

        self.calibration.homography, _ = cv2.findHomography(src_points, dst_points)

        return self.calibration.homography is not None


# Global transformer instance
_transformer: Optional[CoordinateTransformer] = None
_calibration_file = "/home/doug/panda-mcp/calibration.json"


def get_transformer() -> CoordinateTransformer:
    """Get the global coordinate transformer."""
    global _transformer

    if _transformer is None:
        if os.path.exists(_calibration_file):
            cal = CalibrationData.load(_calibration_file)
            _transformer = CoordinateTransformer(cal)
        else:
            _transformer = CoordinateTransformer()

    return _transformer


def save_calibration():
    """Save current calibration to file."""
    global _transformer
    if _transformer:
        _transformer.calibration.save(_calibration_file)


# Need cv2 for homography
try:
    import cv2
except ImportError:
    # Provide stub if cv2 not available
    pass
