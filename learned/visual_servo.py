#!/usr/bin/env python3
"""
Visual servo controller for pick tasks.

Developed through direct experimentation. This is v1 - uses simple color-based
detection and a proportional controller. Expected to need iteration.

Usage:
    python -m learned.visual_servo --target red --action pick

Architecture:
    1. Camera captures frame at ~30fps
    2. Color detector finds target block centroid in pixel coords
    3. Proportional controller computes robot-frame correction
    4. Robot moves incrementally toward target
    5. Once aligned, descend and grasp

The pixel-to-robot mapping is the key learned component. Initially it's a
simple linear transform with hand-tuned gains. This will be replaced by a
trained model as data accumulates.
"""

import os
import sys
import json
import time
import logging
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ServoConfig:
    """Tunable parameters - expect these to change through experimentation."""

    # Camera
    camera_device: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Color detection HSV ranges (will need tuning per object)
    color_ranges: dict = field(default_factory=lambda: {
        "red": {
            # Red wraps around in HSV, so we need two ranges
            "lower1": [0, 100, 100],
            "upper1": [10, 255, 255],
            "lower2": [160, 100, 100],
            "upper2": [180, 255, 255],
        },
        "green": {
            "lower1": [35, 80, 80],
            "upper1": [85, 255, 255],
        },
        "blue": {
            "lower1": [100, 100, 80],
            "upper1": [130, 255, 255],
        },
    })

    # Servo control
    approach_height: float = 0.12       # Height to start servo from (meters)
    grasp_height: float = 0.020         # Height to grasp at (meters)
    servo_gain: float = 0.3             # Fraction of offset to correct each step
    convergence_px: int = 20            # Pixel distance threshold for "aligned"
    max_iterations: int = 30            # Safety limit on servo iterations
    servo_rate_hz: float = 5.0          # Target update rate

    # Pixel-to-robot mapping at servo height
    # These are rough initial values - MUST BE TUNED through experimentation
    # Units: meters per pixel, measured at approach_height
    px_to_robot_x: float = 0.00035      # pixel_dy -> robot_dx (down in image = +X)
    px_to_robot_y: float = -0.00035     # pixel_dx -> robot_dy (right in image = -Y)

    # Robot workspace safety bounds
    x_min: float = 0.25
    x_max: float = 0.58
    y_min: float = -0.18
    y_max: float = 0.20
    z_min: float = 0.013
    z_max: float = 0.35

    # Gripper
    gripper_open_width: float = 0.08
    grasp_width: float = 0.03
    grasp_force: float = 40.0

    # The gripper's approximate pixel position when centered in frame
    # at approach_height. This is the "reference point" - we move until
    # the target is at this pixel position.
    # NEEDS CALIBRATION: move gripper to known position, photograph, note pixel coords
    gripper_ref_px: tuple = (1050, 350)

    # Bias corrections (meters) applied after pixel_to_robot mapping
    # Positive y_bias shifts the gripper toward camera-left (robot +Y)
    y_bias: float = 0.0


def load_config(path: Optional[str] = None) -> ServoConfig:
    """Load config from JSON file, falling back to defaults."""
    if path is None:
        path = Path(__file__).parent / "servo_config.json"
    if Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        config = ServoConfig()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
    return ServoConfig()


def save_config(config: ServoConfig, path: Optional[str] = None):
    """Save current config for persistence across sessions."""
    if path is None:
        path = Path(__file__).parent / "servo_config.json"
    data = {}
    for k, v in config.__dict__.items():
        data[k] = v
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Color Detection
# ---------------------------------------------------------------------------

class ColorDetector:
    """Finds colored objects in camera frames using HSV thresholding."""

    def __init__(self, config: ServoConfig):
        self.config = config

    def detect(self, frame: np.ndarray, color: str) -> Optional[tuple]:
        """
        Find the centroid of the largest blob of the specified color.

        Returns (cx, cy, area) in pixel coords, or None if not found.
        """
        if color not in self.config.color_ranges:
            logger.error(f"Unknown color: {color}")
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = self.config.color_ranges[color]

        # Build mask (some colors like red need two HSV ranges)
        lower1 = np.array(ranges["lower1"])
        upper1 = np.array(ranges["upper1"])
        mask = cv2.inRange(hsv, lower1, upper1)

        if "lower2" in ranges:
            lower2 = np.array(ranges["lower2"])
            upper2 = np.array(ranges["upper2"])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour above minimum area
        min_area = 500
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.contourArea(largest)

        return (cx, cy, area)

    def detect_all(self, frame: np.ndarray, color: str) -> list:
        """Find all blobs of a color. Returns list of (cx, cy, area)."""
        if color not in self.config.color_ranges:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = self.config.color_ranges[color]

        lower1 = np.array(ranges["lower1"])
        upper1 = np.array(ranges["upper1"])
        mask = cv2.inRange(hsv, lower1, upper1)

        if "lower2" in ranges:
            lower2 = np.array(ranges["lower2"])
            upper2 = np.array(ranges["upper2"])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    results.append((cx, cy, area))

        return sorted(results, key=lambda x: x[2], reverse=True)


# ---------------------------------------------------------------------------
# Servo Controller
# ---------------------------------------------------------------------------

class VisualServoController:
    """
    Proportional visual servo controller.

    Core idea: move the robot so the target appears at the gripper's
    reference pixel position in the camera image.
    """

    def __init__(self, config: ServoConfig):
        self.config = config
        self.detector = ColorDetector(config)
        self.camera = None
        self._cam_client = None
        self._use_daemon = False
        self.robot = None
        self.gripper = None
        self._controller = None
        self.log_entries = []

    def connect_camera(self):
        """Connect to camera via ZMQ daemon (shared access) or direct OpenCV."""
        try:
            from camera_daemon.client import CameraClient
            self._cam_client = CameraClient()
            if self._cam_client.connect():
                self._use_daemon = True
                logger.info(f"Camera connected via daemon at {self._cam_client.active_endpoint}")
                return
            else:
                logger.warning("Camera daemon not available, trying direct...")
                self._cam_client = None
        except ImportError:
            logger.warning("camera_daemon not importable, trying direct...")
            self._cam_client = None

        self._use_daemon = False
        self.camera = cv2.VideoCapture(self.config.camera_device)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        for _ in range(5):
            self.camera.read()
        logger.info("Camera connected directly")

    def connect_robot(self, robot_ip: str = "192.168.0.253"):
        """Connect to Franka Panda via panda-py."""
        import panda_py
        self.robot = panda_py.Panda(robot_ip)
        self.gripper = panda_py.libfranka.Gripper(robot_ip)
        logger.info(f"Robot connected at {robot_ip}")

    def use_existing_robot(self, robot, gripper):
        """Use an already-connected robot and gripper (e.g., from MCP server)."""
        self.robot = robot
        self.gripper = gripper
        logger.info("Using existing robot/gripper connection")

    def use_controller(self, controller):
        """Use a FrankaController instance for all robot operations.
        This is the preferred method when running inside the MCP server,
        as it reuses the existing connection and all the IK/safety logic."""
        self._controller = controller
        self.robot = controller._robot
        self.gripper = controller._gripper
        logger.info("Using FrankaController for robot operations")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera (daemon or direct)."""
        if self._use_daemon and self._cam_client:
            return self._cam_client.get_frame()
        if self.camera is not None:
            ret, frame = self.camera.read()
            return frame if ret else None
        return None

    def get_robot_position(self) -> tuple:
        """Get current end-effector position (x, y, z)."""
        if hasattr(self, '_controller') and self._controller:
            state = self._controller.get_state()
            return (state.ee_position[0], state.ee_position[1], state.ee_position[2])
        state = self.robot.get_state()
        pose = state.O_T_EE  # 4x4 homogeneous transform, column-major
        return (pose[12], pose[13], pose[14])

    def move_to(self, x: float, y: float, z: float):
        """
        Move end effector to target position.
        Uses FrankaController.move_cartesian_ik when available (preferred),
        falls back to direct panda-py calls.
        Clamps to workspace bounds.
        """
        x = max(self.config.x_min, min(self.config.x_max, x))
        y = max(self.config.y_min, min(self.config.y_max, y))
        z = max(self.config.z_min, min(self.config.z_max, z))

        if hasattr(self, '_controller') and self._controller:
            result = self._controller.move_cartesian_ik(
                x=x, y=y, z=z, confirmed=True,
            )
            return result.get("success", False)

        # Fallback: direct panda-py (requires own connection)
        import panda_py
        state = self.robot.get_state()
        q_current = state.q
        position = np.array([x, y, z])
        # Pointing straight down quaternion: w=0, x=1, y=0, z=0 (180 deg around X)
        orientation = np.array([0.0, 1.0, 0.0, 0.0])
        sol = panda_py.ik(position, orientation, q_current, q_current[6])
        if sol is None:
            logger.warning(f"IK failed for ({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        self.robot.move_to_joint_position(sol, speed_factor=0.15)
        return True

    def pixel_offset_to_robot(self, dx_px: float, dy_px: float) -> tuple:
        """
        Convert pixel offset to robot-frame displacement.

        This is the core mapping that needs to be learned/tuned.
        Currently uses a simple linear model.

        dx_px: positive = target is to the RIGHT of gripper ref in image
        dy_px: positive = target is BELOW gripper ref in image
        """
        # Image right -> robot -Y (toward robot's right)
        # Image down -> robot +X (further from base)
        dx_robot = dy_px * self.config.px_to_robot_x
        dy_robot = dx_px * self.config.px_to_robot_y
        return (dx_robot, dy_robot)

    def log(self, msg: str, data: dict = None):
        """Log an observation for later analysis."""
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "elapsed": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
            "msg": msg,
        }
        if data:
            entry.update(data)
        self.log_entries.append(entry)
        logger.info(f"[servo] {msg} {data or ''}")

    def save_episode(self, success: bool, target_color: str):
        """Save episode data for later analysis and training."""
        episode_dir = Path(__file__).parent.parent / "data" / "servo_episodes"
        episode_dir.mkdir(parents=True, exist_ok=True)

        episode_id = len(list(episode_dir.glob("episode_*.json")))
        episode_path = episode_dir / f"episode_{episode_id:04d}.json"

        episode = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_color": target_color,
            "success": success,
            "config": {
                "approach_height": self.config.approach_height,
                "grasp_height": self.config.grasp_height,
                "servo_gain": self.config.servo_gain,
                "px_to_robot_x": self.config.px_to_robot_x,
                "px_to_robot_y": self.config.px_to_robot_y,
                "gripper_ref_px": self.config.gripper_ref_px,
            },
            "log": self.log_entries,
        }

        with open(episode_path, "w") as f:
            json.dump(episode, f, indent=2)
        logger.info(f"Episode saved to {episode_path}")

    # -------------------------------------------------------------------
    # Main servo routines
    # -------------------------------------------------------------------

    def find_target(self, color: str) -> Optional[tuple]:
        """
        Capture a frame and find the target block.
        Returns (cx, cy, area) or None.
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.detector.detect(frame, color)

    def pixel_to_robot(self, px: int, py: int) -> tuple:
        """
        Convert pixel coordinates to approximate robot (x, y) coordinates.

        Uses calibration points to build an affine mapping.
        Currently uses a single known point + estimated scale factors.
        Will improve as more calibration data is collected.
        """
        # Known calibration point (from gamepad demonstration 2026-02-19):
        # Red block at pixel (795, 534) = robot (0.489, 0.016)
        cal_px, cal_py = 795, 534
        cal_rx, cal_ry = 0.489, 0.016

        # Scale factors (estimated from camera geometry, NEEDS TUNING)
        # pixel_y -> robot_x: going down in image = further from base = +X
        # pixel_x -> robot_y: going right in image = toward robot's right = depends
        scale_py_to_rx = self.config.px_to_robot_x   # m per pixel in Y
        scale_px_to_ry = self.config.px_to_robot_y    # m per pixel in X

        robot_x = cal_rx + (py - cal_py) * scale_py_to_rx
        robot_y = cal_ry + (px - cal_px) * scale_px_to_ry + self.config.y_bias
        return (robot_x, robot_y)

    def move_above_target(self, color: str) -> Optional[tuple]:
        """
        Detect target and move gripper above it at approach_height.
        Uses the pixel-to-robot mapping for coarse positioning.

        Returns (robot_x, robot_y) if successful, None if target not found.
        """
        # First, detect with arm out of the way (or nearby)
        detection = self.find_target(color)
        if detection is None:
            self.log("target_not_found", {"color": color})
            return None

        cx, cy, area = detection
        robot_x, robot_y = self.pixel_to_robot(cx, cy)

        self.log("target_detected", {
            "pixel": (cx, cy),
            "area": area,
            "estimated_robot": (round(robot_x, 4), round(robot_y, 4)),
        })

        # Move above the estimated position
        self.move_to(robot_x, robot_y, self.config.approach_height)
        return (robot_x, robot_y)

    def pick(self, color: str) -> bool:
        """
        Full pick sequence using lateral approach:
        1. Open gripper
        2. Detect target and estimate its robot position
        3. Move to offset position at TABLE height (beside the block)
        4. Slide laterally so block enters gripper gap
        5. Close gripper
        6. Lift

        The lateral approach avoids the descent-alignment problem where
        gripper fingers catch on block edges during top-down approach.

        Returns True if grasp succeeded.
        """
        self._start_time = time.time()
        self.log_entries = []
        self.log("pick_start", {"color": color})

        # Step 1: Open gripper
        self.log("gripper_open")
        if hasattr(self, '_controller') and self._controller:
            self._controller.gripper_move(self.config.gripper_open_width)
        else:
            self.gripper.move(self.config.gripper_open_width, 0.1)

        # Step 2: Move arm out of the way and detect target
        rx, ry, rz = self.get_robot_position()
        self.log("moving_clear", {"to_z": 0.25})
        self.move_to(0.30, 0.0, 0.25)
        time.sleep(0.5)

        detection = self.find_target(color)
        if detection is None:
            self.log("target_not_found")
            self.save_episode(False, color)
            return False

        cx, cy, area = detection
        target_rx, target_ry = self.pixel_to_robot(cx, cy)
        self.log("target_detected", {
            "pixel": (cx, cy),
            "area": area,
            "robot_estimate": (round(target_rx, 4), round(target_ry, 4)),
        })

        # Step 3: Position horizontally above the block, THEN descend
        # All horizontal movement happens at approach height (above everything)
        # Only descend once we're directly above the target
        self.log("move_above_target", {
            "target_rx": round(target_rx, 4),
            "target_ry": round(target_ry, 4),
            "height": self.config.approach_height,
        })

        self.move_to(target_rx, target_ry, self.config.approach_height)
        time.sleep(0.3)

        # Step 4: Descend vertically to grasp height in steps
        z = self.config.approach_height
        step = 0.02
        self.log("descending", {"from_z": round(z, 3), "to_z": self.config.grasp_height})

        while z > self.config.grasp_height + step:
            z -= step
            self.move_to(target_rx, target_ry, z)
            time.sleep(0.05)
        self.move_to(target_rx, target_ry, self.config.grasp_height)
        self.log("at_grasp_height", {"z": self.config.grasp_height})
        time.sleep(0.2)

        # Step 5: Grasp
        self.log("grasping", {"width": self.config.grasp_width, "force": self.config.grasp_force})
        if hasattr(self, '_controller') and self._controller:
            result = self._controller.gripper_grasp(
                width=self.config.grasp_width,
                force=self.config.grasp_force,
            )
            success = result.get("success", False)
            width = result.get("width", 0)
        else:
            success = self.gripper.grasp(
                self.config.grasp_width,
                speed=0.1,
                force=self.config.grasp_force,
                epsilon_inner=0.01,
                epsilon_outer=0.08,
            )
            state = self.gripper.read_once()
            width = state.width

        self.log("grasp_result", {
            "libfranka_success": success,
            "gripper_width": round(width, 4),
        })
        got_object = width > 0.005

        # Step 6: Lift
        self.log("lifting")
        self.move_to(target_rx, target_ry, self.config.approach_height)
        time.sleep(0.3)

        # Verify object is still held
        if hasattr(self, '_controller') and self._controller:
            st = self._controller.get_state()
            width2 = st.gripper_width
        else:
            state2 = self.gripper.read_once()
            width2 = state2.width
        held = width2 > 0.005

        # Take a photo to verify
        frame = self.capture_frame()
        if frame is not None:
            result_path = Path(__file__).parent.parent / "data" / "servo_episodes"
            result_path.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(result_path / "last_pick_result.jpg"), frame)

        self.log("pick_result", {
            "got_object": got_object,
            "still_held": held,
            "final_width": round(width2, 4),
        })

        self.save_episode(held, color)
        return held

    def disconnect(self):
        """Clean up."""
        if self._cam_client:
            self._cam_client.disconnect()
        if self.camera is not None:
            self.camera.release()
        logger.info("Disconnected")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def calibrate_gripper_ref(controller: VisualServoController, color: str):
    """
    Helper to calibrate gripper_ref_px.

    Move the gripper so it's DIRECTLY OVER a colored block at approach height,
    then call this. It records where the block appears in the image -
    that becomes the reference pixel position.
    """
    detection = controller.find_target(color)
    if detection is None:
        print(f"Can't find {color} target!")
        return

    cx, cy, area = detection
    print(f"Target detected at pixel ({cx}, {cy}), area={area}")
    print(f"Setting gripper_ref_px = ({cx}, {cy})")
    controller.config.gripper_ref_px = (cx, cy)
    save_config(controller.config)
    print("Config saved.")


def calibrate_pixel_scale(controller: VisualServoController, color: str):
    """
    Helper to calibrate px_to_robot_x and px_to_robot_y.

    Moves the gripper by known amounts and measures the pixel displacement
    of a visible target. This establishes the pixel-to-robot scale factors.
    """
    print("Calibrating pixel-to-robot scale...")
    print("Make sure a colored block is visible and the gripper is at approach height.")

    # Record initial position and target pixel position
    rx0, ry0, rz0 = controller.get_robot_position()
    det0 = controller.find_target(color)
    if det0 is None:
        print("Can't find target!")
        return

    # Move robot +3cm in X, re-detect
    print("Moving +3cm in X...")
    controller.move_to(rx0 + 0.03, ry0, rz0)
    time.sleep(0.5)
    det1 = controller.find_target(color)
    if det1 is None:
        print("Lost target after X move!")
        controller.move_to(rx0, ry0, rz0)
        return

    # Move back, then +3cm in Y
    controller.move_to(rx0, ry0, rz0)
    time.sleep(0.5)
    print("Moving +3cm in Y...")
    controller.move_to(rx0, ry0 + 0.03, rz0)
    time.sleep(0.5)
    det2 = controller.find_target(color)
    if det2 is None:
        print("Lost target after Y move!")
        controller.move_to(rx0, ry0, rz0)
        return

    controller.move_to(rx0, ry0, rz0)

    # Compute scale factors
    # When robot moves +X, target appears to shift in image (due to parallax/perspective)
    # But actually, the TARGET doesn't move - the CAMERA-ROBOT relationship changes
    # Wait - this is wrong. The target is fixed on the table. Moving the robot changes
    # where the gripper is, not where the target appears.
    #
    # What we actually want: if the target is at pixel (px, py) and we want it at
    # the reference pixel (ref_x, ref_y), how much robot movement is needed?
    #
    # Better calibration: move the TARGET (or use multiple known positions)
    # For now, skip this and tune manually.

    print("NOTE: Automatic calibration of pixel scale requires more thought.")
    print(f"Detection at start: {det0}")
    print(f"Detection after +3cm X: {det1}")
    print(f"Detection after +3cm Y: {det2}")
    print("The target is FIXED - moving the robot doesn't move the target in the image.")
    print("Pixel scale should be calibrated by moving gripper to two known pixel positions")
    print("and measuring the robot displacement between them.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Visual servo pick controller")
    parser.add_argument("--target", default="red", help="Target color (red/green/blue)")
    parser.add_argument("--action", default="pick", choices=["pick", "servo", "detect", "simulate", "calibrate_ref", "calibrate_scale"])
    parser.add_argument("--robot-ip", default="192.168.0.253")
    parser.add_argument("--no-robot", action="store_true", help="Camera-only mode (no robot)")
    parser.add_argument("--show", action="store_true", help="Display camera feed with overlay")
    args = parser.parse_args()

    config = load_config()
    controller = VisualServoController(config)

    # Connect camera
    controller.connect_camera()

    # Connect robot (unless camera-only mode)
    if not args.no_robot:
        controller.connect_robot(args.robot_ip)

    try:
        if args.action == "detect":
            # Just detect and print - useful for tuning color ranges
            print(f"Detecting {args.target}...")
            for i in range(10):
                det = controller.find_target(args.target)
                if det:
                    cx, cy, area = det
                    print(f"  [{i}] pixel=({cx}, {cy}) area={area}")
                else:
                    print(f"  [{i}] not found")
                time.sleep(0.5)

        elif args.action == "simulate":
            # Simulate pick sequence - vision only, report planned moves
            print(f"Simulating lateral pick for {args.target} (no robot moves)...")
            print()

            # Detect all targets
            frame = controller.capture_frame()
            if frame is None:
                print("Failed to capture frame!")
                return
            all_det = controller.detector.detect_all(frame, args.target)
            print(f"Detected {len(all_det)} {args.target} blobs:")
            for i, (cx, cy, area) in enumerate(all_det):
                rx, ry = controller.pixel_to_robot(cx, cy)
                print(f"  [{i}] pixel=({cx},{cy}) area={area} -> robot=({rx:.4f}, {ry:.4f})")

            # Use largest blob
            det = controller.find_target(args.target)
            if det is None:
                print("No target found!")
                return

            cx, cy, area = det
            target_rx, target_ry = controller.pixel_to_robot(cx, cy)
            print(f"\nTarget: pixel ({cx},{cy}) -> robot ({target_rx:.4f}, {target_ry:.4f})")

            # Plan lateral approach
            offset_y = 0.08
            approach_ry = target_ry + offset_y
            print(f"\nPlanned sequence:")
            print(f"  1. Open gripper")
            print(f"  2. Move clear: (0.30, 0.00, 0.25)")
            print(f"  3. Move above offset: ({target_rx:.3f}, {approach_ry:.3f}, {config.approach_height})")
            print(f"  4. Descend to grasp height: z={config.grasp_height}")
            print(f"  5. Slide from y={approach_ry:.3f} to y={target_ry:.3f} (8 steps of 1cm)")
            print(f"  6. Grasp at width={config.grasp_width}")
            print(f"  7. Lift to z={config.approach_height}")

        elif args.action == "servo":
            # Servo only (no grasp)
            aligned = controller.servo_to_target(args.target)
            print(f"Servo result: {'aligned' if aligned else 'failed'}")

        elif args.action == "pick":
            success = controller.pick(args.target)
            print(f"Pick result: {'success' if success else 'failed'}")

        elif args.action == "calibrate_ref":
            calibrate_gripper_ref(controller, args.target)

        elif args.action == "calibrate_scale":
            calibrate_pixel_scale(controller, args.target)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
