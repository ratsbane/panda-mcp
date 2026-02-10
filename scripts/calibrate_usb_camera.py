#!/usr/bin/env python3
"""
Calibrate USB camera to robot frame using ArUco markers.

Detects ArUco markers (4X4_50) in the USB camera image, matches them to
known robot positions, and computes a homography with quality metrics.

Uses RANSAC when >4 markers are available for robustness.

Usage:
    python scripts/calibrate_usb_camera.py

Requires: camera daemon running, opencv-python
"""

import sys
import numpy as np

sys.path.insert(0, ".")

import cv2
from camera_daemon.client import CameraClient
from common.calibration import CalibrationData, CoordinateTransformer

# Known marker positions in robot frame (meters)
# IDs 0-3: original ArUco markers taped to desk
# IDs 4-5: from teaching mode (depth calibration)
MARKER_ROBOT_COORDS = {
    0: (-0.004, -0.309),
    1: (0.121, 0.222),
    2: (0.441, -0.359),
    3: (0.524, 0.128),
    4: (0.191, -0.191),
    5: (0.238, 0.041),
}

# Output paths
NPZ_PATH = "/tmp/aruco_calibration.npz"
JSON_PATH = "/home/doug/panda-mcp/calibration.json"
INTRINSICS_PATH = "/tmp/camera_intrinsics.npz"
TABLE_Z = 0.013  # table height in meters


def detect_aruco_markers(frame: np.ndarray) -> dict[int, tuple[int, int]]:
    """Detect ArUco markers in a BGR frame.

    Returns dict mapping marker ID to (center_x, center_y) pixel coordinates.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)

    markers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            cx = float(np.mean(c[:, 0]))
            cy = float(np.mean(c[:, 1]))
            markers[int(marker_id)] = (cx, cy)

    return markers


def main():
    print("=== USB Camera Calibration ===\n")

    # Connect to camera daemon
    print("Connecting to camera daemon...")
    client = CameraClient()
    if not client.connect():
        print("ERROR: Could not connect to camera daemon.")
        print("Make sure camera-daemon is running (systemctl status camera-daemon)")
        return 1

    info = client.get_info()
    print(f"Camera: {info.width}x{info.height} via {info.endpoint}")

    # Capture frame
    print("Capturing frame...")
    frame = client.get_frame()
    if frame is None:
        print("ERROR: Could not capture frame")
        return 1

    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    # Detect ArUco markers
    print("\nDetecting ArUco markers (4X4_50)...")
    markers = detect_aruco_markers(frame)

    if not markers:
        print("ERROR: No ArUco markers detected!")
        print("Make sure markers are visible and well-lit.")
        return 1

    print(f"Found {len(markers)} markers: {sorted(markers.keys())}")

    # Match to known positions
    pixel_pts = []
    robot_pts = []
    used_ids = []

    for mid in sorted(markers.keys()):
        if mid not in MARKER_ROBOT_COORDS:
            print(f"  ID {mid}: pixel ({markers[mid][0]:.0f}, {markers[mid][1]:.0f}) - SKIPPED (no robot coords)")
            continue

        px, py = markers[mid]
        rx, ry = MARKER_ROBOT_COORDS[mid]
        pixel_pts.append((px, py))
        robot_pts.append((rx, ry))
        used_ids.append(mid)
        print(f"  ID {mid}: pixel ({px:.0f}, {py:.0f}) -> robot ({rx:.3f}, {ry:.3f})")

    if len(pixel_pts) < 4:
        print(f"\nERROR: Need at least 4 matched markers, got {len(pixel_pts)}")
        print("Detected but unmatched IDs:", [m for m in markers if m not in MARKER_ROBOT_COORDS])
        return 1

    # Load camera intrinsics if available
    import os
    camera_matrix = None
    dist_coeffs = None
    if os.path.exists(INTRINSICS_PATH):
        intrinsics = np.load(INTRINSICS_PATH)
        camera_matrix = intrinsics["camera_matrix"]
        dist_coeffs = intrinsics["dist_coeffs"]
        rms = float(intrinsics.get("rms_error", 0))
        print(f"\nLoaded camera intrinsics from {INTRINSICS_PATH} (RMS={rms:.3f}px)")
        print(f"  Will apply lens undistortion before homography computation.")
    else:
        print(f"\nNo camera intrinsics found at {INTRINSICS_PATH}")
        print(f"  Run calibrate_camera_intrinsics.py with a checkerboard for better accuracy.")

    # Build calibration
    print(f"\nComputing homography from {len(pixel_pts)} point pairs...")
    cal = CalibrationData(
        image_width=w,
        image_height=h,
        workspace_z=TABLE_Z,
    )
    cal.camera_matrix = camera_matrix
    cal.dist_coeffs = dist_coeffs

    for (px, py), (rx, ry) in zip(pixel_pts, robot_pts):
        cal.calibration_points.append((px, py, rx, ry))

    transformer = CoordinateTransformer(cal)
    if not transformer.compute_homography():
        print("ERROR: Homography computation failed!")
        return 1

    # Report quality
    print(f"\n=== Calibration Quality ===")
    print(f"  Points used: {len(pixel_pts)}")
    if len(pixel_pts) > 4:
        print(f"  Method: RANSAC (robust to outliers)")
    else:
        print(f"  Method: exact fit (4 points, no redundancy)")
    print(f"  Mean reprojection error: {cal.mean_error_mm:.2f} mm")
    print(f"  Max reprojection error:  {cal.max_error_mm:.2f} mm")
    print(f"  Condition number: {cal.condition_number:.1f}")

    if cal.reprojection_errors:
        print(f"\n  Per-point errors:")
        for i, (mid, err) in enumerate(zip(used_ids, cal.reprojection_errors)):
            print(f"    ID {mid}: {err:.2f} mm")

    if cal.condition_number and cal.condition_number > 1000:
        print(f"\n  WARNING: High condition number ({cal.condition_number:.0f})")
        print(f"  This means pixel errors are amplified ~{cal.condition_number:.0f}x in worst case.")
        print(f"  Consider: more calibration points, lens undistortion, or use depth camera instead.")

    # Cross-validation (leave-one-out) if we have enough points
    if len(pixel_pts) >= 5:
        print(f"\n=== Leave-One-Out Cross-Validation ===")
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(robot_pts, dtype=np.float32)
        loo_errors = []

        for i in range(len(pixel_pts)):
            # Remove point i, recompute homography
            src_loo = np.delete(src, i, axis=0)
            dst_loo = np.delete(dst, i, axis=0)
            H_loo, _ = cv2.findHomography(src_loo, dst_loo)
            if H_loo is None:
                continue

            # Predict held-out point
            pt = src[i:i+1].reshape(-1, 1, 2)
            pred = cv2.perspectiveTransform(pt, H_loo)
            pred_x, pred_y = pred[0, 0]
            gt_x, gt_y = dst[i]
            err = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) * 1000
            loo_errors.append(err)
            print(f"  ID {used_ids[i]}: held out -> {err:.1f} mm error")

        if loo_errors:
            print(f"\n  LOO mean error: {np.mean(loo_errors):.1f} mm")
            print(f"  LOO max error:  {np.max(loo_errors):.1f} mm")
            print(f"  (This estimates real-world accuracy better than reprojection error)")

    # Save NPZ (compatible with existing code)
    pixel_arr = np.array(pixel_pts, dtype=np.float32)
    robot_arr = np.array(robot_pts, dtype=np.float32)
    H = cal.homography
    H_inv = np.linalg.inv(H)

    save_kwargs = dict(
        H=H, H_inv=H_inv,
        pixel_pts=pixel_arr, robot_pts=robot_arr,
        table_z=TABLE_Z, marker_ids=used_ids,
    )
    if camera_matrix is not None:
        save_kwargs["camera_matrix"] = camera_matrix
        save_kwargs["dist_coeffs"] = dist_coeffs
    np.savez(NPZ_PATH, **save_kwargs)
    print(f"\nSaved NPZ to {NPZ_PATH}")

    # Save JSON (with quality metrics)
    cal.save(JSON_PATH)
    print(f"Saved JSON to {JSON_PATH}")

    # Save annotated image for review
    annotated = frame.copy()
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if corners:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    for mid, (px, py) in markers.items():
        if mid in MARKER_ROBOT_COORDS:
            rx, ry = MARKER_ROBOT_COORDS[mid]
            cv2.putText(annotated, f"ID{mid} ({rx:.2f},{ry:.2f})",
                       (int(px) + 10, int(py) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("/tmp/calibration_annotated.jpg", annotated)
    print(f"Saved annotated image to /tmp/calibration_annotated.jpg")

    client.disconnect()
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
