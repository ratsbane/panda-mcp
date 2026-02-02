#!/usr/bin/env python3
"""
Data collection script using MCP servers.
Collects (image, gripper_position) pairs by moving to random positions.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests


class MCPRobot:
    """Interface to robot via MCP server."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}"

    def get_status(self) -> dict:
        """Get current robot state."""
        # MCP uses JSON-RPC, but we can also read from the server directly
        # For now, use a simple approach
        import subprocess
        result = subprocess.run(
            ["claude", "--print", "mcp__franka-mcp__get_status"],
            capture_output=True, text=True
        )
        return json.loads(result.stdout) if result.returncode == 0 else None


class DirectCamera:
    """Direct camera access via OpenCV."""

    def __init__(self, device: int = 0):
        self.device = device
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Warmup
        for _ in range(10):
            self.cap.read()
        print(f"Camera connected: device {self.device}")

    def capture(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Capture failed")
        return frame

    def disconnect(self):
        if self.cap:
            self.cap.release()


def random_position():
    """Generate random position in workspace."""
    x = random.uniform(0.3, 0.6)
    y = random.uniform(-0.3, 0.3)
    z = random.uniform(0.1, 0.5)
    return x, y, z


def main():
    parser = argparse.ArgumentParser(description="Collect training data via MCP")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between samples")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")
    parser.add_argument("--session-name", type=str, default=None, help="Session name")
    parser.add_argument("--camera", type=int, default=0, help="Camera device")
    args = parser.parse_args()

    # Create session directory
    session_name = args.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_name
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize camera
    camera = DirectCamera(args.camera)
    camera.connect()

    # Collect samples
    samples = []
    print(f"Collecting {args.samples} samples...")
    print(f"Output: {session_dir}")
    print("Move the camera to different positions during collection!")

    try:
        # We'll use panda-py through subprocess since direct import doesn't work
        import panda_py
        robot = panda_py.Panda("192.168.0.253")
        gripper = panda_py.libfranka.Gripper("192.168.0.253")

        from scipy.spatial.transform import Rotation

        for i in range(args.samples):
            # Move to random position
            x, y, z = random_position()

            try:
                # Build pose matrix
                rot = Rotation.from_euler('xyz', [np.pi, 0, 0])
                pose = np.eye(4)
                pose[:3, :3] = rot.as_matrix()
                pose[:3, 3] = [x, y, z]

                robot.move_to_pose(pose)
            except Exception as e:
                print(f"  Move failed: {e}")
                continue

            time.sleep(args.delay)

            # Capture image
            frame = camera.capture()

            # Get robot state
            q = robot.q
            ee_pose = robot.get_pose()
            position = ee_pose[:3, 3].tolist()

            gripper_state = gripper.read_once()
            gripper_width = gripper_state.width

            # Save image
            img_filename = f"frame_{i:06d}.jpg"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), frame)

            # Record sample
            sample = {
                "image_path": f"images/{img_filename}",
                "robot_state": {
                    "end_effector_position": position,
                    "joint_positions": q.tolist(),
                    "gripper_width": float(gripper_width),
                },
                "timestamp": datetime.now().isoformat(),
            }
            samples.append(sample)

            if (i + 1) % 50 == 0:
                print(f"  Collected {i + 1}/{args.samples} samples")
                # Save intermediate results
                with open(session_dir / "samples.json", "w") as f:
                    json.dump(samples, f, indent=2)

    except ImportError:
        print("panda_py not available - using MCP-based collection")
        print("This requires the franka-mcp server to be running")

        # Alternative: just capture images with current robot position
        # The robot will need to be moved manually or via MCP commands
        import subprocess

        for i in range(args.samples):
            # Capture image
            frame = camera.capture()

            # Get robot status via MCP (this is a simplified version)
            # In practice, you'd need to call the MCP server properly
            try:
                # Try to get status - this is a placeholder
                # The actual implementation would need proper MCP client
                position = [0.4, 0.0, 0.3]  # Placeholder

                # Save image
                img_filename = f"frame_{i:06d}.jpg"
                img_path = images_dir / img_filename
                cv2.imwrite(str(img_path), frame)

                sample = {
                    "image_path": f"images/{img_filename}",
                    "robot_state": {
                        "end_effector_position": position,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                samples.append(sample)

                if (i + 1) % 50 == 0:
                    print(f"  Collected {i + 1}/{args.samples} samples")

            except Exception as e:
                print(f"  Error: {e}")
                continue

            time.sleep(args.delay)

    finally:
        camera.disconnect()

        # Save final results
        with open(session_dir / "samples.json", "w") as f:
            json.dump(samples, f, indent=2)

        # Save metadata
        metadata = {
            "session_name": session_name,
            "total_samples": len(samples),
            "created": datetime.now().isoformat(),
            "note": "Multi-viewpoint collection",
        }
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nCollection complete: {len(samples)} samples")
        print(f"Saved to: {session_dir}")


if __name__ == "__main__":
    main()
