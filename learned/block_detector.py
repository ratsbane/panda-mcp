"""
Detect colored blocks in camera frames and compute robot-frame pick parameters.

Uses HSV color thresholding + ArUco homography for position,
and minAreaRect for grasp orientation (yaw).
"""

import json
import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV ranges for block colors
COLOR_RANGES = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),
        ((160, 100, 100), (180, 255, 255)),
    ],
    "green": [
        ((35, 80, 80), (85, 255, 255)),
    ],
    "blue": [
        ((80, 50, 50), (130, 255, 255)),
    ],
    "orange": [
        ((10, 100, 100), (25, 255, 255)),
    ],
}

MIN_BLOCK_AREA = 500  # pixels


@dataclass
class DetectedBlock:
    """A detected block with position and grasp parameters."""
    color: str
    pixel_x: int
    pixel_y: int
    pixel_area: float
    robot_x: float
    robot_y: float
    yaw: float  # gripper yaw in radians to align with block short axis
    in_workspace: bool
    bbox_w: int
    bbox_h: int


def load_homography(calibration_path: str = "calibration.json") -> np.ndarray:
    """Load the ArUco homography matrix from calibration file."""
    with open(calibration_path) as f:
        calib = json.load(f)
    return np.array(calib["homography"]), calib.get("workspace", {})


def pixel_to_robot(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Convert pixel coordinates to robot frame via homography."""
    p = np.array([px, py, 1.0])
    r = H @ p
    return r[0] / r[2], r[1] / r[2]


def _compute_grasp_yaw(contour: np.ndarray) -> float:
    """Compute gripper yaw to align with the block's short axis.

    Uses minAreaRect to find the block's orientation, then returns
    the yaw angle that aligns the gripper fingers across the short axis.

    Returns yaw in radians, clamped to [-pi/2, pi/2].
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # minAreaRect returns angle in [-90, 0) degrees
    # w and h are the sides of the rotated rectangle
    # We want the gripper to close across the SHORT axis
    if w > h:
        # Short axis is perpendicular to the width direction
        grasp_angle_deg = angle + 90
    else:
        # Short axis is along the angle direction
        grasp_angle_deg = angle

    # Convert to radians and normalize to [-pi/2, pi/2]
    yaw = math.radians(grasp_angle_deg)
    while yaw > math.pi / 2:
        yaw -= math.pi
    while yaw < -math.pi / 2:
        yaw += math.pi

    return yaw


def detect_blocks(
    frame: np.ndarray,
    H: np.ndarray,
    workspace: Optional[dict] = None,
    colors: Optional[list[str]] = None,
) -> list[DetectedBlock]:
    """Detect colored blocks in a camera frame.

    Args:
        frame: BGR camera frame
        H: homography matrix (pixel -> robot)
        workspace: dict with x_min, x_max, y_min, y_max bounds
        colors: list of color names to detect (default: all)

    Returns:
        List of DetectedBlock sorted by area (largest first)
    """
    if workspace is None:
        workspace = {"x_min": 0.2, "x_max": 0.6, "y_min": -0.2, "y_max": 0.2}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    target_colors = colors or list(COLOR_RANGES.keys())
    blocks = []

    for color_name in target_colors:
        if color_name not in COLOR_RANGES:
            continue

        ranges = COLOR_RANGES[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOCK_AREA:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            rx, ry = pixel_to_robot(cx, cy, H)
            in_ws = (
                workspace["x_min"] <= rx <= workspace["x_max"]
                and workspace["y_min"] <= ry <= workspace["y_max"]
            )

            yaw = _compute_grasp_yaw(cnt)

            blocks.append(DetectedBlock(
                color=color_name,
                pixel_x=cx,
                pixel_y=cy,
                pixel_area=area,
                robot_x=rx,
                robot_y=ry,
                yaw=yaw,
                in_workspace=in_ws,
                bbox_w=w,
                bbox_h=h,
            ))

    blocks.sort(key=lambda b: -b.pixel_area)
    return blocks


def find_block(
    frame: np.ndarray,
    H: np.ndarray,
    color: str,
    workspace: Optional[dict] = None,
) -> Optional[DetectedBlock]:
    """Find the largest block of a given color that's in the workspace.

    Returns None if no matching block found.
    """
    blocks = detect_blocks(frame, H, workspace, colors=[color])
    for block in blocks:
        if block.in_workspace:
            return block
    return None
