#!/usr/bin/env python3
"""
Calibrate USB camera intrinsics using a checkerboard pattern.

Captures multiple images of a checkerboard from different angles,
then computes camera matrix and distortion coefficients.

Usage:
    python scripts/calibrate_camera_intrinsics.py [--rows 6] [--cols 9] [--square-size 0.025]

Press SPACE to capture an image, 'q' to finish and compute calibration.
Need at least 10 images from varied angles for good calibration.
"""

import argparse
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from camera_daemon.client import CameraClient

OUTPUT_PATH = "/tmp/camera_intrinsics.npz"


def main():
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corner rows")
    parser.add_argument("--cols", type=int, default=9, help="Checkerboard inner corner cols")
    parser.add_argument("--square-size", type=float, default=0.025, help="Square size in meters")
    parser.add_argument("--auto", action="store_true", help="Auto-capture (no keyboard)")
    parser.add_argument("--count", type=int, default=20, help="Number of images to capture (auto mode)")
    args = parser.parse_args()

    board_size = (args.cols, args.rows)
    square_size = args.square_size

    print(f"=== Camera Intrinsic Calibration ===")
    print(f"Checkerboard: {args.cols}x{args.rows} inner corners, {square_size*1000:.0f}mm squares\n")

    # Connect to camera
    print("Connecting to camera daemon...")
    client = CameraClient()
    if not client.connect():
        print("ERROR: Could not connect to camera daemon")
        return 1

    info = client.get_info()
    print(f"Camera: {info.width}x{info.height}")

    # Prepare object points (3D checkerboard corners in world frame)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []  # 3D points
    img_points = []  # 2D pixel points
    img_size = None

    if args.auto:
        print(f"\nAuto-capture mode: capturing {args.count} frames with checkerboard detection...")
        captured = 0
        attempts = 0
        max_attempts = args.count * 10

        while captured < args.count and attempts < max_attempts:
            frame = client.get_frame()
            if frame is None:
                continue

            attempts += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]

            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(refined)
                captured += 1
                print(f"  Captured {captured}/{args.count} (attempt {attempts})")
                time.sleep(0.5)  # Brief pause between captures

        print(f"\nCaptured {captured} valid images out of {attempts} attempts")
    else:
        print(f"\nManual capture mode:")
        print(f"  Hold checkerboard in front of camera at various angles.")
        print(f"  Press SPACE when checkerboard is detected (green overlay).")
        print(f"  Press 'q' when done (need at least 10 images).")
        print(f"  (Requires display - if headless, use --auto)\n")

        while True:
            frame = client.get_frame()
            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]

            found, corners = cv2.findChessboardCorners(gray, board_size, None)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, board_size, corners, found)
                cv2.putText(display, "DETECTED - Press SPACE to capture",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No checkerboard detected",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Captured: {len(img_points)} images",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord(' ') and found:
                refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_points.append(objp)
                img_points.append(refined)
                print(f"  Captured image {len(img_points)}")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    if len(img_points) < 5:
        print(f"\nERROR: Need at least 5 images, got {len(img_points)}")
        return 1

    # Compute calibration
    print(f"\nComputing calibration from {len(img_points)} images...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    print(f"\nCalibration RMS reprojection error: {ret:.4f} pixels")
    print(f"\nCamera matrix:")
    print(f"  fx = {camera_matrix[0, 0]:.1f}")
    print(f"  fy = {camera_matrix[1, 1]:.1f}")
    print(f"  cx = {camera_matrix[0, 2]:.1f}")
    print(f"  cy = {camera_matrix[1, 2]:.1f}")
    print(f"\nDistortion coefficients: {dist_coeffs.ravel()}")

    k1, k2 = dist_coeffs.ravel()[:2]
    print(f"\n  k1 = {k1:.6f} (radial)")
    print(f"  k2 = {k2:.6f} (radial)")
    if abs(k1) > 0.01:
        print(f"  Significant radial distortion detected!")

    # Save
    np.savez(OUTPUT_PATH,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=img_size,
             rms_error=ret,
             num_images=len(img_points))

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\nTo use with USB camera calibration:")
    print(f"  The calibrate_usb_camera.py script will auto-detect this file")
    print(f"  and apply lens undistortion before computing the homography.")

    client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
