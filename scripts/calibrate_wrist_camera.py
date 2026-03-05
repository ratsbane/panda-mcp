#!/usr/bin/env python3
"""
Calibrate D405 wrist camera to Franka end-effector frame (hand-eye calibration).

For a wrist-mounted camera, we need T_ee_camera:
    p_robot = T_base_ee(q) @ T_ee_camera @ p_camera

This script:
1. Loads calibration samples (NPZ files with EE pose + D405 image/depth)
2. Detects ArUco markers (4X4_50) in each sample's color image
3. Gets 3D camera-frame positions from aligned depth
4. Transforms known marker positions from robot frame to EE frame
5. Solves Kabsch/SVD for T_ee_camera
6. Saves calibration to calibration/wrist_calibration.npz

Data collection is done interactively via MCP tools. Each sample NPZ contains:
    - color: (H,W,3) uint8 BGR image
    - depth: (H,W) uint16 raw depth
    - depth_scale: float (meters per count)
    - intrinsics: [ppx, ppy, fx, fy]
    - ee_position: [x, y, z] in meters
    - ee_rpy: [roll, pitch, yaw] in radians

Usage:
    # After collecting samples via MCP:
    python scripts/calibrate_wrist_camera.py

    # Specify sample directory:
    python scripts/calibrate_wrist_camera.py --samples-dir calibration/wrist_samples

    # Save a sample from MCP data (called by collection helper):
    python scripts/calibrate_wrist_camera.py --save-sample \\
        --scan-path /tmp/realsense_scan.npz \\
        --ee-pos 0.4,0.0,0.3 --ee-rpy -3.14,0.0,0.0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Known marker positions in robot frame (meters) — same as other calibration scripts
MARKER_ROBOT_COORDS = {
    0: np.array([-0.004, -0.309, 0.013]),
    1: np.array([0.121,  0.222, 0.013]),
    2: np.array([0.441, -0.359, 0.013]),
    3: np.array([0.524,  0.128, 0.013]),
    4: np.array([0.191, -0.191, 0.014]),
    5: np.array([0.238,  0.041, 0.015]),
}

SAMPLES_DIR = Path("calibration/wrist_samples")
OUTPUT_PATH = Path("calibration/wrist_calibration.npz")


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles (ZYX convention) to 3x3 rotation matrix.

    Same convention as panda-py / franka_mcp controller.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return Rz @ Ry @ Rx


def ee_pose_to_matrix(position, rpy):
    """Build 4x4 SE(3) matrix from EE position + Euler angles.

    Args:
        position: [x, y, z] in meters
        rpy: [roll, pitch, yaw] in radians

    Returns:
        4x4 numpy array (T_base_ee)
    """
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(*rpy)
    T[:3, 3] = position
    return T


def detect_aruco_markers(image):
    """Detect ArUco markers (4X4_50) in an image.

    Args:
        image: BGR or grayscale numpy array

    Returns:
        Dict mapping marker ID to (center_x, center_y) pixel coordinates
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Enhance contrast for IR images
    gray = cv2.equalizeHist(gray)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Relax parameters for challenging conditions (IR, low contrast)
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.05

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(gray)

    markers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            markers[int(marker_id)] = (cx, cy)

    return markers


def deproject_pixel(px, py, depth_m, intrinsics):
    """Deproject pixel + depth to 3D camera-frame point.

    Args:
        px, py: pixel coordinates
        depth_m: depth in meters
        intrinsics: dict or array [ppx, ppy, fx, fy]

    Returns:
        [x, y, z] in camera frame (meters)
    """
    if isinstance(intrinsics, dict):
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    else:
        ppx, ppy, fx, fy = intrinsics
    x = (px - ppx) * depth_m / fx
    y = (py - ppy) * depth_m / fy
    return np.array([x, y, depth_m])


def get_depth_at_pixel(depth_frame, depth_scale, px, py, radius=5):
    """Get depth in meters at a pixel with median filtering."""
    h, w = depth_frame.shape
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    patch = depth_frame[y0:y1, x0:x1]
    valid = patch[(patch > 0) & (patch < 65535)]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) * depth_scale


def save_sample(scan_path, ee_pos, ee_rpy, output_dir=None):
    """Save a calibration sample from a realsense scan + EE pose."""
    out_dir = Path(output_dir) if output_dir else SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load scan
    scan = np.load(scan_path)
    existing = sorted(out_dir.glob("sample_*.npz"))
    idx = len(existing)

    sample_path = out_dir / f"sample_{idx:03d}.npz"
    data = {
        "ee_position": np.array(ee_pos),
        "ee_rpy": np.array(ee_rpy),
    }
    # Copy relevant fields from scan
    for key in ["color", "depth", "intrinsics"]:
        if key in scan:
            data[key] = scan[key]
    if "depth_scale" in scan:
        data["depth_scale"] = scan["depth_scale"]
    else:
        data["depth_scale"] = np.array(0.0001)  # D405 default

    np.savez_compressed(str(sample_path), **data)
    print(f"Saved sample {idx} to {sample_path}")
    return str(sample_path)


def compute_calibration(samples_dir=None):
    """Compute T_ee_camera from saved calibration samples."""
    sdir = Path(samples_dir) if samples_dir else SAMPLES_DIR
    sample_files = sorted(sdir.glob("sample_*.npz"))

    if not sample_files:
        print(f"No samples found in {sdir}")
        return None

    print(f"=== Wrist Camera Hand-Eye Calibration ===")
    print(f"Found {len(sample_files)} samples in {sdir}\n")

    all_camera_pts = []  # 3D points in camera frame (meters)
    all_ee_pts = []      # same points in EE frame (meters)
    all_labels = []      # (sample_idx, marker_id)

    for si, sf in enumerate(sample_files):
        sample = np.load(str(sf))
        color = sample["color"]
        depth = sample["depth"]
        depth_scale = float(sample["depth_scale"])
        intrinsics = sample["intrinsics"]
        ee_pos = sample["ee_position"]
        ee_rpy = sample["ee_rpy"]

        # Build T_base_ee and its inverse
        T_base_ee = ee_pose_to_matrix(ee_pos, ee_rpy)
        T_ee_base = np.linalg.inv(T_base_ee)

        print(f"Sample {si}: EE at ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

        # Detect ArUco markers
        markers = detect_aruco_markers(color)
        if not markers:
            print(f"  No ArUco markers detected — skipping")
            continue

        print(f"  Found markers: {sorted(markers.keys())}")

        for mid, (cx, cy) in sorted(markers.items()):
            if mid not in MARKER_ROBOT_COORDS:
                print(f"  ID {mid}: skipped (no known robot coords)")
                continue

            # Get 3D position in camera frame
            depth_m = get_depth_at_pixel(depth, depth_scale, cx, cy, radius=5)
            if depth_m is None or depth_m < 0.01 or depth_m > 1.0:
                print(f"  ID {mid}: skipped (bad depth: {depth_m})")
                continue

            camera_pt = deproject_pixel(cx, cy, depth_m, intrinsics)

            # Transform known robot-frame position to EE frame
            robot_pt = MARKER_ROBOT_COORDS[mid]
            robot_pt_h = np.array([robot_pt[0], robot_pt[1], robot_pt[2], 1.0])
            ee_pt = (T_ee_base @ robot_pt_h)[:3]

            all_camera_pts.append(camera_pt)
            all_ee_pts.append(ee_pt)
            all_labels.append((si, mid))

            print(f"  ID {mid}: px ({cx},{cy}), depth {depth_m*1000:.1f}mm, "
                  f"cam ({camera_pt[0]:.4f}, {camera_pt[1]:.4f}, {camera_pt[2]:.4f}), "
                  f"ee ({ee_pt[0]:.4f}, {ee_pt[1]:.4f}, {ee_pt[2]:.4f})")

    n = len(all_camera_pts)
    if n < 3:
        print(f"\nERROR: Need at least 3 point pairs, got {n}")
        return None

    print(f"\n{n} point correspondences from {len(sample_files)} samples")

    # Kabsch algorithm: find T_ee_camera such that T_ee_camera @ p_cam ≈ p_ee
    src = np.array(all_camera_pts, dtype=np.float64)  # already in meters
    dst = np.array(all_ee_pts, dtype=np.float64)

    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    t = dst_centroid - R @ src_centroid

    T_ee_camera = np.eye(4)
    T_ee_camera[:3, :3] = R
    T_ee_camera[:3, 3] = t

    # Compute errors
    transformed = (R @ src.T).T + t
    errors = np.linalg.norm(transformed - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    print(f"\nT_ee_camera (camera → EE frame):")
    print(T_ee_camera)
    print(f"\nRMSE: {rmse*1000:.2f} mm")

    print("\nPer-point errors:")
    for i, (si, mid) in enumerate(all_labels):
        print(f"  Sample {si}, ID {mid}: {errors[i]*1000:.2f} mm")

    # Leave-one-out cross-validation
    if n >= 4:
        loo_errors = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            s = src[mask]
            d_ = dst[mask]
            sc = s.mean(axis=0)
            dc = d_.mean(axis=0)
            Hloo = (s - sc).T @ (d_ - dc)
            Ul, Sl, Vtl = np.linalg.svd(Hloo)
            dl = np.linalg.det(Vtl.T @ Ul.T)
            Rl = Vtl.T @ np.diag([1, 1, np.sign(dl)]) @ Ul.T
            tl = dc - Rl @ sc
            pred = Rl @ src[i] + tl
            loo_errors.append(np.linalg.norm(pred - dst[i]))
        loo_errors = np.array(loo_errors)
        print(f"\nLOO cross-validation: mean={loo_errors.mean()*1000:.1f}mm, "
              f"max={loo_errors.max()*1000:.1f}mm")

    # Verification: transform each sample's marker positions to robot frame
    print("\nVerification (camera → EE → robot):")
    for si, sf in enumerate(sample_files):
        sample = np.load(str(sf))
        ee_pos = sample["ee_position"]
        ee_rpy = sample["ee_rpy"]
        T_base_ee = ee_pose_to_matrix(ee_pos, ee_rpy)

        pts_in_sample = [(i, mid) for i, (s, mid) in enumerate(all_labels) if s == si]
        for pi, mid in pts_in_sample:
            cam_pt = all_camera_pts[pi]
            cam_h = np.array([*cam_pt, 1.0])
            ee_pred = T_ee_camera @ cam_h
            robot_pred = T_base_ee @ ee_pred
            robot_actual = MARKER_ROBOT_COORDS[mid]
            err = np.linalg.norm(robot_pred[:3] - robot_actual) * 1000
            print(f"  Sample {si}, ID {mid}: predicted robot "
                  f"({robot_pred[0]:.4f}, {robot_pred[1]:.4f}, {robot_pred[2]:.4f}) "
                  f"actual ({robot_actual[0]:.3f}, {robot_actual[1]:.3f}, {robot_actual[2]:.3f}) "
                  f"error {err:.1f}mm")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(OUTPUT_PATH),
        T_ee_camera=T_ee_camera,
        camera_pts_m=src,
        ee_pts_m=dst,
        labels=np.array(all_labels),
        rmse_m=np.array(rmse),
        n_samples=np.array(len(sample_files)),
        n_points=np.array(n),
    )
    print(f"\nCalibration saved to {OUTPUT_PATH}")

    return T_ee_camera


def main():
    parser = argparse.ArgumentParser(description="Wrist camera hand-eye calibration")
    parser.add_argument("--samples-dir", default=str(SAMPLES_DIR),
                        help="Directory with calibration sample NPZ files")
    parser.add_argument("--save-sample", action="store_true",
                        help="Save a single calibration sample")
    parser.add_argument("--scan-path", default="/tmp/realsense_scan.npz",
                        help="Path to realsense scan NPZ (for --save-sample)")
    parser.add_argument("--ee-pos", type=str,
                        help="EE position as x,y,z (for --save-sample)")
    parser.add_argument("--ee-rpy", type=str,
                        help="EE orientation as roll,pitch,yaw in radians (for --save-sample)")
    args = parser.parse_args()

    if args.save_sample:
        if not args.ee_pos or not args.ee_rpy:
            print("ERROR: --ee-pos and --ee-rpy required with --save-sample")
            return 1
        ee_pos = [float(x) for x in args.ee_pos.split(",")]
        ee_rpy = [float(x) for x in args.ee_rpy.split(",")]
        save_sample(args.scan_path, ee_pos, ee_rpy, args.samples_dir)
        return 0

    compute_calibration(args.samples_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
