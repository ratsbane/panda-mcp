"""
3D camera-to-robot calibration for PhotoNeo depth camera.

Uses Kabsch (SVD-based) rigid body transform to map 3D points from
camera frame (mm) to robot frame (meters). Calibrated using ArUco
markers with known robot-frame positions.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path

CALIBRATION_PATH = "/tmp/depth_calibration.npz"


def compute_rigid_transform(
    camera_pts_mm: np.ndarray,
    robot_pts_m: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Compute optimal rigid transform (R, t) from camera frame to robot frame.

    Uses the Kabsch algorithm (SVD on cross-covariance matrix) to find
    the rotation and translation that minimizes RMSE between point pairs.

    Args:
        camera_pts_mm: Nx3 array of 3D points in camera frame (millimeters)
        robot_pts_m: Nx3 array of corresponding points in robot frame (meters)

    Returns:
        (T, rmse) where T is a 4x4 SE(3) transform matrix and rmse is in meters.
        To transform a point: p_robot = T @ [p_camera_mm/1000, 1]
    """
    assert len(camera_pts_mm) >= 3, "Need at least 3 point pairs"
    assert len(camera_pts_mm) == len(robot_pts_m)

    # Convert camera points from mm to meters
    src = camera_pts_mm.astype(np.float64) / 1000.0
    dst = robot_pts_m.astype(np.float64)

    # Compute centroids
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)

    # Center the points
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid

    # Cross-covariance matrix
    H = src_centered.T @ dst_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation, handling reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Compute translation
    t = dst_centroid - R @ src_centroid

    # Build 4x4 SE(3) matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Compute RMSE
    transformed = (R @ src.T).T + t
    errors = np.linalg.norm(transformed - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    return T, rmse


@dataclass
class DepthCalibrationData:
    """Stores a 3D rigid body transform from camera to robot frame."""

    transform: np.ndarray  # 4x4 SE(3) matrix
    camera_pts_mm: np.ndarray  # Nx3, calibration points in camera frame
    robot_pts_m: np.ndarray  # Nx3, calibration points in robot frame
    marker_ids: list[int]  # ArUco marker IDs used
    rmse_m: float  # RMSE in meters

    def save(self, path: str = CALIBRATION_PATH):
        np.savez(
            path,
            transform=self.transform,
            camera_pts_mm=self.camera_pts_mm,
            robot_pts_m=self.robot_pts_m,
            marker_ids=np.array(self.marker_ids),
            rmse_m=np.array(self.rmse_m),
        )

    @classmethod
    def load(cls, path: str = CALIBRATION_PATH) -> "DepthCalibrationData":
        data = np.load(path)
        return cls(
            transform=data["transform"],
            camera_pts_mm=data["camera_pts_mm"],
            robot_pts_m=data["robot_pts_m"],
            marker_ids=data["marker_ids"].tolist(),
            rmse_m=float(data["rmse_m"]),
        )


class DepthCoordinateTransformer:
    """Transforms 3D points between camera and robot frames."""

    def __init__(self, calibration: DepthCalibrationData):
        self.calibration = calibration
        self._T = calibration.transform
        self._T_inv = np.linalg.inv(calibration.transform)

    def camera_to_robot(self, x_mm: float, y_mm: float, z_mm: float) -> tuple[float, float, float]:
        """Transform a point from camera frame (mm) to robot frame (m)."""
        p_camera_m = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0, 1.0])
        p_robot = self._T @ p_camera_m
        return float(p_robot[0]), float(p_robot[1]), float(p_robot[2])

    def robot_to_camera(self, x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
        """Transform a point from robot frame (m) to camera frame (mm)."""
        p_robot = np.array([x_m, y_m, z_m, 1.0])
        p_camera_m = self._T_inv @ p_robot
        return float(p_camera_m[0] * 1000), float(p_camera_m[1] * 1000), float(p_camera_m[2] * 1000)


# Singleton
_depth_transformer: DepthCoordinateTransformer | None = None


def get_depth_transformer(path: str = CALIBRATION_PATH) -> DepthCoordinateTransformer | None:
    """Get cached depth transformer, loading from file if needed. Returns None if not calibrated."""
    global _depth_transformer
    if _depth_transformer is None:
        if Path(path).exists():
            cal = DepthCalibrationData.load(path)
            _depth_transformer = DepthCoordinateTransformer(cal)
    return _depth_transformer
