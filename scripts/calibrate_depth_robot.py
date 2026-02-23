#!/usr/bin/env python3
"""
Process depth calibration scans collected by the robot.

This script is the SECOND step of robot-based depth calibration:
1. Claude orchestrates: parks arm, captures background, moves to N positions, captures at each
2. This script processes the saved scans: background subtraction, gripper detection, transform computation

Input files (created by Claude during orchestration):
    /tmp/depth_cal_bg.npz          - Background scan (arm out of view)
    /tmp/depth_cal_positions.json  - List of {scan_path, robot_xyz} for each position

Usage:
    python scripts/calibrate_depth_robot.py
"""

import json
import sys
import numpy as np

sys.path.insert(0, ".")
from common.depth_calibration import compute_rigid_transform, DepthCalibrationData, CALIBRATION_PATH

POSITIONS_FILE = "/tmp/depth_cal_positions.json"
BG_SCAN_FILE = "/tmp/depth_cal_bg.npz"


def find_gripper_in_pointcloud(
    pointcloud: np.ndarray,
    bg_pointcloud: np.ndarray,
    min_diff_mm: float = 50.0,
) -> np.ndarray | None:
    """Find the gripper position in camera frame by background subtraction.

    Strategy: The arm is CLOSER to the camera than the table. Find pixels where
    depth decreased significantly (arm in front of table). Among those arm-surface
    points, the gripper tip is the one CLOSEST to the table = LARGEST camera Z
    (furthest from camera among arm points, but still closer than table).

    Returns:
        (camera_xyz_mm, n_arm_points) or None if not found
    """
    h, w = pointcloud.shape[:2]

    # Only use pixels valid in BOTH scans (avoid noise from coverage fluctuations)
    current_valid = np.all(pointcloud != 0, axis=2) & np.all(np.isfinite(pointcloud), axis=2)
    bg_valid = np.all(bg_pointcloud != 0, axis=2) & np.all(np.isfinite(bg_pointcloud), axis=2)
    both_valid = current_valid & bg_valid

    # Depth difference: positive = current closer to camera (arm appeared)
    depth_diff = np.zeros((h, w), dtype=np.float32)
    depth_diff[both_valid] = bg_pointcloud[both_valid, 2] - pointcloud[both_valid, 2]

    # Arm mask: current scan significantly closer than background
    arm_mask = depth_diff > min_diff_mm

    arm_points = pointcloud[arm_mask]
    n_arm = len(arm_points)

    if n_arm < 50:
        return None

    # Compute table surface depth (median Z of background valid points)
    bg_z_median = np.median(bg_pointcloud[bg_valid, 2])

    # Filter: arm points must be closer to camera than table (Z < table Z)
    # This removes any spurious points beyond the table
    arm_closer = arm_points[arm_points[:, 2] < bg_z_median]
    if len(arm_closer) < 50:
        return None

    # The gripper tip = arm point with LARGEST Z (closest to table, but still above it)
    # Take the bottom N points (largest Z = closest to table) and cluster
    sorted_idx = np.argsort(-arm_closer[:, 2])  # descending Z
    n_tip = min(100, max(30, len(arm_closer) // 50))  # ~2% of arm, capped at 100
    tip_points = arm_closer[sorted_idx[:n_tip]]

    # Remove outliers: keep points within 50mm of median in XY
    median_xy = np.median(tip_points[:, :2], axis=0)
    xy_dist = np.linalg.norm(tip_points[:, :2] - median_xy, axis=1)
    inliers = tip_points[xy_dist < 50]

    if len(inliers) < 10:
        # Fall back to all tip points
        gripper_pos = np.median(tip_points, axis=0)
    else:
        gripper_pos = np.median(inliers, axis=0)

    return gripper_pos, n_arm


def main():
    print("=== Process Depth Calibration Scans ===\n")

    # Load positions metadata
    try:
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {POSITIONS_FILE} not found.")
        print("Run the calibration collection first (via Claude MCP orchestration).")
        return 1

    print(f"Found {len(positions)} calibration positions")

    # Load background scan
    try:
        bg_data = np.load(BG_SCAN_FILE)
        bg_pointcloud = bg_data["pointcloud"]
        print(f"Background scan: {bg_pointcloud.shape[1]}x{bg_pointcloud.shape[0]}")
    except FileNotFoundError:
        print(f"ERROR: {BG_SCAN_FILE} not found.")
        return 1

    # Process each position
    camera_points = []
    robot_points = []

    for i, pos in enumerate(positions):
        scan_path = pos["scan_path"]
        robot_xyz = np.array(pos["robot_xyz"])

        print(f"\nPosition {i+1}/{len(positions)}: robot ({robot_xyz[0]:.4f}, {robot_xyz[1]:.4f}, {robot_xyz[2]:.4f})")

        try:
            scan_data = np.load(scan_path)
            pointcloud = scan_data["pointcloud"]
        except FileNotFoundError:
            print(f"  SKIPPED (scan file not found: {scan_path})")
            continue

        result = find_gripper_in_pointcloud(pointcloud, bg_pointcloud)
        if result is None:
            print(f"  SKIPPED (gripper not found)")
            continue

        gripper_cam, n_arm_points = result
        print(f"  Camera: ({gripper_cam[0]:.1f}, {gripper_cam[1]:.1f}, {gripper_cam[2]:.1f}) mm  "
              f"({n_arm_points} arm points)")

        camera_points.append(gripper_cam)
        robot_points.append(robot_xyz)

    print(f"\n{'='*50}")
    print(f"Valid correspondences: {len(camera_points)}")

    if len(camera_points) < 4:
        print("ERROR: Need at least 4 valid points!")
        return 1

    camera_pts = np.array(camera_points)
    robot_pts = np.array(robot_points)

    # Distance consistency check
    print("\nDistance consistency (rigid transform preserves distances):")
    ratios = []
    for i in range(len(camera_pts)):
        for j in range(i+1, len(camera_pts)):
            d_cam = np.linalg.norm(camera_pts[i] - camera_pts[j]) / 1000.0
            d_rob = np.linalg.norm(robot_pts[i] - robot_pts[j])
            if d_rob > 0.01:
                ratio = d_cam / d_rob
                ratios.append(ratio)
                if abs(ratio - 1.0) > 0.10:
                    print(f"  WARNING pts {i+1}-{j+1}: cam={d_cam*1000:.0f}mm robot={d_rob*1000:.0f}mm ratio={ratio:.3f}")
    if ratios:
        print(f"  Mean ratio: {np.mean(ratios):.4f} (should be ~1.0)")
        print(f"  Std ratio: {np.std(ratios):.4f} (should be small)")

    # Compute transform
    print("\nComputing rigid transform...")
    T, rmse = compute_rigid_transform(camera_pts, robot_pts)
    print(f"  RMSE: {rmse*1000:.2f} mm")

    # Per-point errors
    print("\nPer-point errors:")
    errors = []
    for i in range(len(camera_pts)):
        cam_m = camera_pts[i] / 1000.0
        transformed = T[:3, :3] @ cam_m + T[:3, 3]
        error_mm = np.linalg.norm(transformed - robot_pts[i]) * 1000
        errors.append(error_mm)
        print(f"  {i+1}. robot ({robot_pts[i][0]:.3f}, {robot_pts[i][1]:.3f}, {robot_pts[i][2]:.3f}) "
              f"-> {error_mm:.1f} mm")
    print(f"  Mean: {np.mean(errors):.1f} mm, Max: {np.max(errors):.1f} mm")

    # Save
    cal = DepthCalibrationData(
        transform=T,
        camera_pts_mm=camera_pts,
        robot_pts_m=robot_pts,
        marker_ids=list(range(len(camera_pts))),
        rmse_m=rmse,
    )
    cal.save(CALIBRATION_PATH)
    print(f"\nCalibration saved to {CALIBRATION_PATH}")

    # Quick verification: transform each calibration point
    print("\nVerification:")
    from common.depth_calibration import DepthCoordinateTransformer
    transformer = DepthCoordinateTransformer(cal)
    for i in range(len(camera_pts)):
        rx, ry, rz = transformer.camera_to_robot(*camera_pts[i])
        expected = robot_pts[i]
        print(f"  {i+1}. camera ({camera_pts[i][0]:.0f}, {camera_pts[i][1]:.0f}, {camera_pts[i][2]:.0f}) mm "
              f"-> robot ({rx:.4f}, {ry:.4f}, {rz:.4f}) "
              f"[expected ({expected[0]:.4f}, {expected[1]:.4f}, {expected[2]:.4f})]")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
