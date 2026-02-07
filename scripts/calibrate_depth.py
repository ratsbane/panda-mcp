#!/usr/bin/env python3
"""
Calibrate PhotoNeo depth camera to Franka robot frame.

Detects ArUco markers (4X4_50, IDs 0-3) in the PhotoNeo texture image,
looks up their 3D positions from the pointcloud, and computes a rigid
body transform (Kabsch/SVD) to the robot frame.

Usage:
    python scripts/calibrate_depth.py

Requires: depth-mcp dependencies (numpy), opencv-python, and SSH access to tuppy.
"""

import asyncio
import importlib.util
import sys
import numpy as np

sys.path.insert(0, ".")

# Import PhoxiClient directly to avoid depth_mcp/__init__.py pulling in mcp server deps
_spec = importlib.util.spec_from_file_location("phoxi_client", "depth_mcp/phoxi_client.py")
_phoxi_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phoxi_mod)
PhoxiClient = _phoxi_mod.PhoxiClient

from common.depth_calibration import compute_rigid_transform, DepthCalibrationData, CALIBRATION_PATH

# Known marker positions in robot frame (meters)
# IDs 0-3 from 2D ArUco calibration, IDs 4-5 from teaching mode
MARKER_ROBOT_COORDS = {
    0: np.array([-0.004, -0.309, 0.013]),
    1: np.array([0.121,  0.222, 0.013]),
    2: np.array([0.441, -0.359, 0.013]),
    3: np.array([0.524,  0.128, 0.013]),
    4: np.array([0.191, -0.191, 0.014]),
    5: np.array([0.238,  0.041, 0.015]),
}


def detect_aruco_markers(texture_uint16: np.ndarray) -> dict[int, tuple[int, int]]:
    """Detect ArUco markers in PhotoNeo texture image.

    Args:
        texture_uint16: PhotoNeo texture as uint16 array

    Returns:
        Dict mapping marker ID to (center_x, center_y) pixel coordinates
    """
    import cv2

    # Convert uint16 to uint8 for ArUco detection
    # PhotoNeo texture is typically 16-bit grayscale
    texture_8 = (texture_uint16 // 256).astype(np.uint8)

    # If the image looks too dark, try normalize
    if texture_8.max() < 30:
        print("  Texture too dark with //256, trying normalize...")
        texture_8 = cv2.normalize(texture_uint16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, rejected = detector.detectMarkers(texture_8)

    markers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Compute center of marker
            c = corners[i][0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            markers[int(marker_id)] = (cx, cy)

    return markers


async def main():
    print("=== PhotoNeo Depth Camera Calibration ===\n")

    # Connect and capture
    client = PhoxiClient()
    print("Connecting to tuppy...")
    result = await client.connect()
    if not result.get("connected"):
        print(f"Failed to connect: {result.get('error')}")
        return 1

    print(f"Connected to {result['hostname']}")
    print("Capturing scan...")
    scan = await client.capture()
    if not scan.get("success"):
        print(f"Capture failed: {scan.get('error')}")
        return 1

    print(f"Scan: {scan['width']}x{scan['height']}, {scan['coverage']*100:.1f}% coverage\n")

    # Detect ArUco markers
    print("Detecting ArUco markers (4X4_50)...")
    markers = detect_aruco_markers(client.texture)

    if not markers:
        print("ERROR: No ArUco markers detected!")
        print("Make sure markers are visible and well-lit.")
        return 1

    print(f"Found {len(markers)} markers: {sorted(markers.keys())}")
    for mid, (cx, cy) in sorted(markers.items()):
        print(f"  ID {mid}: pixel ({cx}, {cy})")

    # Look up 3D positions from pointcloud
    print("\nLooking up 3D positions from pointcloud...")
    camera_pts = []
    robot_pts = []
    used_ids = []

    for mid in sorted(markers.keys()):
        if mid not in MARKER_ROBOT_COORDS:
            print(f"  ID {mid}: SKIPPED (no known robot coords)")
            continue

        cx, cy = markers[mid]
        # Use median patch for robustness
        patch_result = client.get_depth_patch(cx, cy, radius=5)
        if not patch_result.get("valid"):
            print(f"  ID {mid}: SKIPPED (no depth data at pixel)")
            continue

        pos = patch_result["position_mm"]
        camera_pt = np.array([pos["x"], pos["y"], pos["z"]])
        robot_pt = MARKER_ROBOT_COORDS[mid]

        camera_pts.append(camera_pt)
        robot_pts.append(robot_pt)
        used_ids.append(mid)

        print(f"  ID {mid}: camera ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) mm "
              f"-> robot ({robot_pt[0]:.3f}, {robot_pt[1]:.3f}, {robot_pt[2]:.3f}) m")

    if len(camera_pts) < 3:
        print(f"\nERROR: Need at least 3 valid markers, got {len(camera_pts)}")
        return 1

    # Compute rigid transform
    camera_pts = np.array(camera_pts)
    robot_pts = np.array(robot_pts)

    print(f"\nComputing rigid transform from {len(camera_pts)} point pairs...")
    T, rmse = compute_rigid_transform(camera_pts, robot_pts)

    print(f"RMSE: {rmse*1000:.2f} mm")

    # Print per-marker errors
    print("\nPer-marker errors:")
    for i, mid in enumerate(used_ids):
        cam_m = camera_pts[i] / 1000.0
        transformed = (T[:3, :3] @ cam_m) + T[:3, 3]
        error = np.linalg.norm(transformed - robot_pts[i]) * 1000
        print(f"  ID {mid}: {error:.2f} mm")

    # Save
    cal = DepthCalibrationData(
        transform=T,
        camera_pts_mm=camera_pts,
        robot_pts_m=robot_pts,
        marker_ids=used_ids,
        rmse_m=rmse,
    )
    cal.save(CALIBRATION_PATH)
    print(f"\nCalibration saved to {CALIBRATION_PATH}")

    # Verify with a round-trip
    print("\nVerification (camera -> robot -> camera):")
    from common.depth_calibration import DepthCoordinateTransformer
    transformer = DepthCoordinateTransformer(cal)
    for i, mid in enumerate(used_ids):
        rx, ry, rz = transformer.camera_to_robot(*camera_pts[i])
        cx, cy, cz = transformer.robot_to_camera(rx, ry, rz)
        orig = camera_pts[i]
        print(f"  ID {mid}: ({orig[0]:.1f}, {orig[1]:.1f}, {orig[2]:.1f}) "
              f"-> robot ({rx:.4f}, {ry:.4f}, {rz:.4f}) "
              f"-> back ({cx:.1f}, {cy:.1f}, {cz:.1f}) mm")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
