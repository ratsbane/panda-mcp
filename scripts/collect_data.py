#!/usr/bin/env python3
"""
Data collection script for visuomotor policy training.

Run this to collect (image, robot_state, camera_pose) samples.
Supports multiple collection modes:
- random: Move to random positions in workspace
- grid: Systematic grid of positions
- manual: User-controlled via keyboard

Usage:
    python scripts/collect_data.py --mode random --samples 1000
    python scripts/collect_data.py --mode grid --grid-resolution 10
    python scripts/collect_data.py --mode manual
"""

import sys
import os
import time
import argparse
import random
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_collection import (
    DataCollector,
    create_robot_state_from_status,
    generate_aruco_markers,
)


class RobotInterface:
    """Interface to robot via direct panda-py calls."""

    def __init__(self, robot_ip: str = "192.168.0.253"):
        self.robot_ip = robot_ip
        self.robot = None
        self.gripper = None

    def connect(self):
        """Connect to robot."""
        import panda_py
        self.robot = panda_py.Panda(self.robot_ip)
        self.gripper = panda_py.libfranka.Gripper(self.robot_ip)
        print(f"Connected to robot at {self.robot_ip}")

    def get_status(self) -> dict:
        """Get current robot state in MCP-like format."""
        if self.robot is None:
            raise RuntimeError("Not connected")

        q = self.robot.q  # Joint positions
        pose = self.robot.get_pose()  # 4x4 transformation matrix

        # Extract position from pose matrix
        position = pose[:3, 3]

        # Extract rotation (simplified - just use euler angles)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_matrix(pose[:3, :3])
        euler = rot.as_euler('xyz')

        # Get gripper width
        gripper_state = self.gripper.read_once()
        gripper_width = gripper_state.width

        return {
            "joint_positions_rad": list(q),
            "end_effector": {
                "position_m": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "orientation_rad": {
                    "roll": float(euler[0]),
                    "pitch": float(euler[1]),
                    "yaw": float(euler[2]),
                },
            },
            "gripper_width_m": float(gripper_width),
        }

    def move_to(self, x: float, y: float, z: float,
                roll: float = np.pi, pitch: float = 0, yaw: float = 0) -> bool:
        """Move end effector to position."""
        if self.robot is None:
            return False

        try:
            # Build pose matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_euler('xyz', [roll, pitch, yaw])
            pose = np.eye(4)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = [x, y, z]

            self.robot.move_to_pose(pose)
            return True
        except Exception as e:
            print(f"Move failed: {e}")
            return False

    def move_gripper(self, width: float) -> bool:
        """Move gripper to width."""
        if self.gripper is None:
            return False
        try:
            self.gripper.move(width, 0.1)
            return True
        except:
            return False


class CameraInterface:
    """Interface to camera via OpenCV."""

    def __init__(self, device: int = 0):
        self.device = device
        self.cap = None

    def connect(self):
        """Connect to camera."""
        import cv2
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warmup
        for _ in range(10):
            self.cap.read()

        print(f"Camera connected: {self.device}")

    def capture(self) -> np.ndarray:
        """Capture a frame."""
        if self.cap is None:
            raise RuntimeError("Camera not connected")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Capture failed")

        return frame

    def disconnect(self):
        if self.cap:
            self.cap.release()


def random_position_in_workspace():
    """Generate a random position within the robot's workspace."""
    x = random.uniform(0.3, 0.6)   # Forward reach
    y = random.uniform(-0.3, 0.3)  # Left/right
    z = random.uniform(0.1, 0.5)   # Height

    return x, y, z


def grid_positions(resolution: int = 5):
    """Generate a grid of positions in the workspace."""
    positions = []

    x_range = np.linspace(0.3, 0.6, resolution)
    y_range = np.linspace(-0.3, 0.3, resolution)
    z_range = np.linspace(0.1, 0.5, resolution)

    for x in x_range:
        for y in y_range:
            for z in z_range:
                positions.append((x, y, z))

    return positions


def collect_random(robot: RobotInterface, camera: CameraInterface,
                   collector: DataCollector, num_samples: int,
                   delay: float = 0.5):
    """Collect samples at random positions."""
    print(f"Collecting {num_samples} random samples...")

    for i in range(num_samples):
        # Move to random position
        x, y, z = random_position_in_workspace()
        print(f"Sample {i+1}/{num_samples}: Moving to ({x:.3f}, {y:.3f}, {z:.3f})")

        if not robot.move_to(x, y, z):
            print("  Move failed, skipping")
            continue

        # Wait for robot to settle
        time.sleep(delay)

        # Capture image and state
        image = camera.capture()
        status = robot.get_status()
        robot_state = create_robot_state_from_status(status)

        # Record sample
        sample = collector.record_sample(image, robot_state)

        if sample.camera_pose:
            print(f"  Camera pose confidence: {sample.camera_pose.confidence:.2f}")

        # Also vary gripper width occasionally
        if random.random() < 0.3:
            width = random.uniform(0.02, 0.08)
            robot.move_gripper(width)
            time.sleep(0.2)

            # Record another sample with different gripper
            image = camera.capture()
            status = robot.get_status()
            robot_state = create_robot_state_from_status(status)
            collector.record_sample(image, robot_state)

    print(f"Collection complete: {collector.sample_count} samples")


def collect_grid(robot: RobotInterface, camera: CameraInterface,
                 collector: DataCollector, resolution: int,
                 delay: float = 0.5):
    """Collect samples at grid positions."""
    positions = grid_positions(resolution)
    print(f"Collecting {len(positions)} grid samples...")

    for i, (x, y, z) in enumerate(positions):
        print(f"Sample {i+1}/{len(positions)}: ({x:.3f}, {y:.3f}, {z:.3f})")

        if not robot.move_to(x, y, z):
            print("  Move failed, skipping")
            continue

        time.sleep(delay)

        image = camera.capture()
        status = robot.get_status()
        robot_state = create_robot_state_from_status(status)
        collector.record_sample(image, robot_state)

    print(f"Collection complete: {collector.sample_count} samples")


def collect_manual(robot: RobotInterface, camera: CameraInterface,
                   collector: DataCollector):
    """Manual collection - record current position on keypress."""
    import cv2

    print("Manual collection mode")
    print("Press 's' to save sample, 'q' to quit")
    print("Move the robot manually or via other interface")

    cv2.namedWindow("Collection", cv2.WINDOW_NORMAL)

    while True:
        image = camera.capture()
        status = robot.get_status()

        # Show live view with info
        display = image.copy()
        ee = status["end_effector"]["position_m"]
        text = f"Pos: ({ee['x']:.3f}, {ee['y']:.3f}, {ee['z']:.3f})"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Samples: {collector.sample_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Collection", display)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('s'):
            robot_state = create_robot_state_from_status(status)
            sample = collector.record_sample(image, robot_state)
            print(f"Saved sample {sample.sample_id}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Collection complete: {collector.sample_count} samples")


def main():
    parser = argparse.ArgumentParser(description="Collect visuomotor training data")

    parser.add_argument("--mode", choices=["random", "grid", "manual"],
                        default="random", help="Collection mode")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples (random mode)")
    parser.add_argument("--grid-resolution", type=int, default=5,
                        help="Grid resolution per axis (grid mode)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between samples (seconds)")
    parser.add_argument("--output-dir", type=str, default="./training_data",
                        help="Output directory")
    parser.add_argument("--session-name", type=str, default=None,
                        help="Session name (default: timestamp)")
    parser.add_argument("--robot-ip", type=str, default="192.168.0.253",
                        help="Robot IP address")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index")
    parser.add_argument("--generate-markers", action="store_true",
                        help="Generate ArUco markers and exit")
    parser.add_argument("--note", type=str, default="",
                        help="Note to add to session metadata")

    args = parser.parse_args()

    if args.generate_markers:
        generate_aruco_markers()
        return

    # Initialize
    print("Initializing...")

    robot = RobotInterface(args.robot_ip)
    camera = CameraInterface(args.camera)
    collector = DataCollector(args.output_dir, args.session_name)

    # Add note if provided
    if args.note:
        collector.add_note(args.note)

    try:
        robot.connect()
        camera.connect()

        # Run collection
        if args.mode == "random":
            collect_random(robot, camera, collector, args.samples, args.delay)
        elif args.mode == "grid":
            collect_grid(robot, camera, collector, args.grid_resolution, args.delay)
        elif args.mode == "manual":
            collect_manual(robot, camera, collector)

    finally:
        # Save session
        collector.save_session()
        camera.disconnect()

    print(f"\nData saved to: {collector.session_dir}")


if __name__ == "__main__":
    main()
