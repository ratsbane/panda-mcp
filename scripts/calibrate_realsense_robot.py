#!/usr/bin/env python3
"""
Calibrate RealSense D435 to robot frame.

Two calibration sources, combinable:

1. ArUco markers (primary, reliable): Detects markers in the D435 color image
   and matches to known robot-frame positions. Works well for table-plane objects.
   Requires markers taped to desk with known positions.

2. Robot arm positions (supplementary, experimental): Uses background subtraction
   to detect the arm in depth, finds the bottommost contour point as gripper tip.
   Less reliable — the D435 can't resolve thin gripper fingers from the arm body.
   Only useful for positions where detection is unambiguous.

Workflow:
    # ArUco-only (recommended):
    python scripts/calibrate_realsense_robot.py --aruco-scan /tmp/rs_cal_bg.npz

    # Robot arm positions (experimental, needs data collection via MCP):
    python scripts/calibrate_realsense_robot.py \
        --bg /tmp/rs_cal_bg.npz \
        --positions /tmp/rs_cal_positions.json

    # Combined (ArUco + robot arm):
    python scripts/calibrate_realsense_robot.py \
        --aruco-scan /tmp/rs_cal_bg.npz \
        --bg /tmp/rs_cal_bg.npz \
        --positions /tmp/rs_cal_positions.json

Data collection for robot arm mode (orchestrated by Claude via MCP):
1. Park arm out of D435 view → capture + save_scan → /tmp/rs_cal_bg.npz
2. Move gripper to N positions → capture + save_scan at each
3. Build /tmp/rs_cal_positions.json: [{"scan_path": "...", "robot_xyz": [x,y,z]}, ...]
4. Run this script
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_OUTPUT = "calibration/realsense_calibration.npz"

# Known ArUco marker positions in robot frame (meters)
MARKER_ROBOT_COORDS = {
    0: np.array([-0.004, -0.309, 0.013]),
    1: np.array([0.121,  0.222, 0.013]),
    2: np.array([0.441, -0.359, 0.013]),
    3: np.array([0.524,  0.128, 0.013]),
    4: np.array([0.191, -0.191, 0.014]),
    5: np.array([0.238,  0.041, 0.015]),
}


def load_scan(path: str) -> dict:
    """Load a RealSense scan NPZ."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def deproject_pixel(px, py, depth_m, intrinsics):
    """Deproject pixel + depth to 3D camera-frame point (meters)."""
    ppx, ppy, fx, fy = intrinsics
    x = (px - ppx) * depth_m / fx
    y = (py - ppy) * depth_m / fy
    return np.array([x, y, depth_m])


def detect_aruco_points(scan: dict) -> tuple[list, list, list]:
    """Detect ArUco markers and return (camera_pts_m, robot_pts_m, labels)."""
    color = scan["color"]
    depth = scan["depth"]
    depth_scale = float(scan["depth_scale"])
    intrinsics = scan["intrinsics"]

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    camera_pts, robot_pts, labels = [], [], []
    if ids is None:
        return camera_pts, robot_pts, labels

    ppx, ppy, fx, fy = intrinsics
    h, w = depth.shape

    for i, mid in enumerate(ids.flatten()):
        mid = int(mid)
        if mid not in MARKER_ROBOT_COORDS:
            continue

        c = corners[i][0]
        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))

        # Median depth in 5-pixel radius
        y0, y1 = max(0, cy - 5), min(h, cy + 6)
        x0, x1 = max(0, cx - 5), min(w, cx + 6)
        patch = depth[y0:y1, x0:x1]
        valid = patch[(patch > 0) & (patch < 65535)]
        if len(valid) == 0:
            continue

        depth_m = float(np.median(valid)) * depth_scale
        cam_pt = deproject_pixel(cx, cy, depth_m, intrinsics)

        camera_pts.append(cam_pt)
        robot_pts.append(MARKER_ROBOT_COORDS[mid])
        labels.append(f"aruco_{mid}")

        print(f"  ArUco {mid}: px ({cx},{cy}) depth {depth_m*1000:.0f}mm  "
              f"cam ({cam_pt[0]:.4f}, {cam_pt[1]:.4f}, {cam_pt[2]:.4f})")

    return camera_pts, robot_pts, labels


def detect_arm_gripper(scan: dict, bg_scan: dict,
                       min_diff_mm: float = 30.0,
                       max_depth_m: float = 0.9) -> np.ndarray | None:
    """Detect gripper position via background subtraction + contour analysis.

    Finds the bottommost point of the largest arm contour. This is an
    approximation — it finds the lowest visible arm point, not necessarily
    the gripper tip.

    Returns camera-frame 3D point (meters) or None.
    """
    depth = scan["depth"]
    bg_depth = bg_scan["depth"]
    depth_scale = float(scan.get("depth_scale", bg_scan["depth_scale"]))
    intrinsics = scan.get("intrinsics", bg_scan["intrinsics"])

    bg_mm = bg_depth.astype(np.float32) * depth_scale * 1000
    pos_mm = depth.astype(np.float32) * depth_scale * 1000

    both_valid = ((bg_depth > 0) & (bg_depth < 65535) &
                  (depth > 0) & (depth < 65535))
    diff = np.zeros_like(bg_mm)
    diff[both_valid] = bg_mm[both_valid] - pos_mm[both_valid]

    # Arm mask: pixels significantly closer than background
    arm_mask = (diff > min_diff_mm).astype(np.uint8) * 255

    # Morphological close to merge arm parts into one contour
    kernel = np.ones((15, 15), np.uint8)
    arm_closed = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        arm_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour = the arm
    arm_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(arm_cnt) < 1000:
        return None

    # Bottommost point of contour = lowest in scene = gripper area
    px, py = tuple(arm_cnt[arm_cnt[:, :, 1].argmax()][0])

    # Get depth at that pixel
    h, w = depth.shape
    radius = 8
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[(patch > 0) & (patch < 65535)]
    if len(valid) == 0:
        return None

    depth_m = float(np.median(valid)) * depth_scale

    # Filter: skip distant points (unreliable detection)
    if depth_m > max_depth_m:
        return None

    return deproject_pixel(px, py, depth_m, intrinsics)


def compute_rigid_transform(camera_pts_m, robot_pts_m):
    """Kabsch/SVD rigid transform. Returns (T_4x4, rmse_m)."""
    assert len(camera_pts_m) >= 3
    src = np.array(camera_pts_m, dtype=np.float64)
    dst = np.array(robot_pts_m, dtype=np.float64)

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    H = (src - src_c).T @ (dst - dst_c)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    t = dst_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    transformed = (R @ src.T).T + t
    errors = np.linalg.norm(transformed - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    return T, rmse


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate RealSense D435 to robot frame")
    parser.add_argument("--aruco-scan",
                        help="NPZ scan with ArUco markers visible (arm out of view)")
    parser.add_argument("--bg",
                        help="Background NPZ scan (for robot arm detection)")
    parser.add_argument("--positions",
                        help="Robot arm positions JSON (for robot arm detection)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output calibration NPZ path")
    parser.add_argument("--max-depth", type=float, default=0.9,
                        help="Max depth for arm detection (m, default: 0.9)")
    args = parser.parse_args()

    if not args.aruco_scan and not args.positions:
        print("ERROR: Provide --aruco-scan and/or --positions")
        return 1

    print("=== RealSense D435 Calibration ===\n")

    camera_pts = []
    robot_pts = []
    labels = []

    # --- ArUco markers ---
    if args.aruco_scan:
        print(f"ArUco markers from {args.aruco_scan}:")
        scan = load_scan(args.aruco_scan)
        a_cam, a_rob, a_labels = detect_aruco_points(scan)
        camera_pts.extend(a_cam)
        robot_pts.extend(a_rob)
        labels.extend(a_labels)
        print(f"  Found {len(a_cam)} markers\n")

    # --- Robot arm positions ---
    if args.positions and args.bg:
        print(f"Robot arm positions from {args.positions}:")
        bg_scan = load_scan(args.bg)
        with open(args.positions) as f:
            positions = json.load(f)
        print(f"  {len(positions)} positions to process")

        for i, pos in enumerate(positions):
            scan = load_scan(pos["scan_path"])
            robot_xyz = np.array(pos["robot_xyz"])

            cam_pt = detect_arm_gripper(
                scan, bg_scan, max_depth_m=args.max_depth)

            if cam_pt is None:
                print(f"  Pos {i+1}: SKIP (detection failed)")
                continue

            camera_pts.append(cam_pt)
            robot_pts.append(robot_xyz)
            labels.append(f"arm_{i+1}")
            print(f"  Pos {i+1}: cam ({cam_pt[0]:.4f}, {cam_pt[1]:.4f}, "
                  f"{cam_pt[2]:.4f})  robot ({robot_xyz[0]:.3f}, "
                  f"{robot_xyz[1]:.3f}, {robot_xyz[2]:.3f})")
        print()

    camera_pts = np.array(camera_pts)
    robot_pts = np.array(robot_pts)
    n = len(camera_pts)

    print(f"Total points: {n}")
    if n < 3:
        print(f"ERROR: Need at least 3 points, got {n}")
        return 1

    # Distance consistency
    ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            d_cam = np.linalg.norm(camera_pts[i] - camera_pts[j])
            d_rob = np.linalg.norm(robot_pts[i] - robot_pts[j])
            if d_rob > 0.01:
                ratios.append(d_cam / d_rob)
    if ratios:
        print(f"Distance ratios: mean={np.mean(ratios):.4f}, "
              f"std={np.std(ratios):.4f} (ideal: mean=1.0, std=0.0)")

    # Compute transform
    T, rmse = compute_rigid_transform(camera_pts, robot_pts)
    print(f"\nRMSE: {rmse*1000:.1f}mm")

    R, t = T[:3, :3], T[:3, 3]
    print("\nPer-point errors:")
    for i in range(n):
        pred = R @ camera_pts[i] + t
        err = np.linalg.norm(pred - robot_pts[i]) * 1000
        print(f"  {labels[i]:10s}: {err:.1f}mm")

    # LOO cross-validation
    if n >= 4:
        loo = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            T_loo, _ = compute_rigid_transform(
                camera_pts[mask], robot_pts[mask])
            pred = T_loo[:3, :3] @ camera_pts[i] + T_loo[:3, 3]
            loo.append(np.linalg.norm(pred - robot_pts[i]) * 1000)
        loo = np.array(loo)
        print(f"\nLOO cross-validation: "
              f"mean={loo.mean():.1f}mm, max={loo.max():.1f}mm")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    intrinsics_data = load_scan(
        args.aruco_scan or args.bg).get("intrinsics")
    depth_scale_data = float(load_scan(
        args.aruco_scan or args.bg).get("depth_scale", 0.001))

    np.savez_compressed(
        str(output),
        transform=T,
        camera_pts_mm=camera_pts * 1000,
        robot_pts_m=robot_pts,
        marker_ids=np.arange(n),
        rmse_m=np.array(rmse),
        intrinsics=intrinsics_data,
        depth_scale=np.array(depth_scale_data),
    )
    print(f"\nCalibration saved to {output}")
    print(f"\nTransform (camera_m → robot_m):")
    print(T)

    return 0


if __name__ == "__main__":
    sys.exit(main())
