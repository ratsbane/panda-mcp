"""
Franka Panda arm controller wrapper.

Provides a safe, high-level interface to panda-py/libfranka.
"""

import os
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional
from dataclasses import dataclass
import numpy as np

# Timeout for blocking panda-py calls (seconds)
MOTION_TIMEOUT_S = 15.0
GRIPPER_TIMEOUT_S = 10.0
STATE_TIMEOUT_S = 5.0

# Check for mock mode (no hardware)
MOCK_MODE = os.environ.get("FRANKA_MOCK", "0") == "1"

if not MOCK_MODE:
    try:
        import panda_py
        from panda_py import Panda
        from panda_py.libfranka import Gripper, RobotState as LibfrankaRobotState
        PANDA_PY_AVAILABLE = True
    except ImportError:
        PANDA_PY_AVAILABLE = False
        logging.warning("panda-py not available, running in mock mode")
        MOCK_MODE = True
else:
    PANDA_PY_AVAILABLE = False

from common.safety import SafetyValidator, get_safety_config

logger = logging.getLogger(__name__)


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    panda-py returns quaternion as [x, y, z, w] (scalar last).
    """
    x, y, z, w = q

    # Roll (rotation around x-axis)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (rotation around z-axis)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion.

    Returns quaternion as [x, y, z, w] (scalar last) for panda-py.
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])



# Good reference pose for tabletop picking (straight down, elbow up)
# This is close to the home pose but with slight forward reach
_REFERENCE_JOINTS = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])

# Panda joint limits
_JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])

# Joint centers (midpoint of limits)
_JOINT_CENTERS = (_JOINT_LIMITS[:, 0] + _JOINT_LIMITS[:, 1]) / 2.0
_JOINT_RANGES = _JOINT_LIMITS[:, 1] - _JOINT_LIMITS[:, 0]


def _solve_ik(
    position: np.ndarray,
    orientation_quat: np.ndarray,
    current_joints: np.ndarray,
    max_single_joint_change: float = 2.5,
    frozen_joints: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Solve analytical IK for a target Cartesian pose.

    Uses panda_py.ik() with current joints as seed to find the nearest
    valid solution. Tries the current q7 first, then nearby values,
    to minimize unnecessary joint movement.

    Args:
        position: Target [x, y, z] in meters
        orientation_quat: Target quaternion [x, y, z, w]
        current_joints: Current 7 joint angles (used as seed)
        max_single_joint_change: Max change for any single joint (radians)
        frozen_joints: Dict mapping joint index to desired value (soft penalty)

    Returns:
        Best joint solution (7,) or None if no valid solution found.
    """
    if not PANDA_PY_AVAILABLE:
        return None

    x, y, z = position
    margin = 0.10  # ~5.7 degrees from joint limits

    # Build q7 candidates: current q7 first, then nearby, then fixed
    current_q7 = float(current_joints[6])
    q7_candidates = [
        current_q7,
        current_q7 + 0.2, current_q7 - 0.2,
        current_q7 + 0.5, current_q7 - 0.5,
        0.785, 0.0, -0.785, 1.57,
    ]
    # Deduplicate (keep order) and clamp to joint 7 limits
    seen = set()
    unique_q7 = []
    for q7 in q7_candidates:
        q7 = max(_JOINT_LIMITS[6, 0] + margin, min(_JOINT_LIMITS[6, 1] - margin, q7))
        key = round(q7, 2)
        if key not in seen:
            seen.add(key)
            unique_q7.append(q7)

    best_solution = None
    best_cost = float('inf')

    for q7 in unique_q7:
        try:
            # panda_py.ik returns the solution closest to current_joints
            sol = panda_py.ik(position, orientation_quat, current_joints, q7)
        except Exception as e:
            logger.debug(f"IK failed for q7={q7:.3f}: {e}")
            continue

        if sol is None:
            continue

        sol = np.array(sol).flatten()
        if sol.shape != (7,):
            continue

        # Hard reject: joint limits
        in_limits = True
        for i in range(7):
            if sol[i] < _JOINT_LIMITS[i, 0] + margin or sol[i] > _JOINT_LIMITS[i, 1] - margin:
                in_limits = False
                break
        if not in_limits:
            continue

        # Hard reject: max single joint change
        joint_changes = np.abs(sol - current_joints)
        if np.any(joint_changes > max_single_joint_change):
            continue

        # Verify with FK: position error < 3mm
        try:
            fk_pose = panda_py.fk(sol)
            pos_error = np.linalg.norm(fk_pose[:3, 3] - position)
            if pos_error > 0.003:
                continue
        except Exception:
            continue

        # Cost: minimize total joint travel (the only thing that matters)
        travel = np.sum((sol - current_joints) ** 2)

        # Small penalty for being near joint limits
        normalized_pos = (sol - _JOINT_CENTERS) / (_JOINT_RANGES / 2)
        limit_penalty = 0.1 * np.sum(normalized_pos ** 4)  # quartic: only bites near limits

        cost = travel + limit_penalty
        if cost < best_cost:
            best_cost = cost
            best_solution = sol

    if best_solution is not None:
        max_change = np.max(np.abs(best_solution - current_joints))
        logger.info(
            f"IK solution: travel={best_cost:.3f}, max_change={max_change:.3f} rad"
        )
    else:
        logger.warning(f"No valid IK solution for target ({x:.3f}, {y:.3f}, {z:.3f})")

    return best_solution


@dataclass
class RobotState:
    """Current state of the robot."""
    joint_positions: list[float]  # 7 joints in radians
    ee_position: tuple[float, float, float]  # x, y, z in meters
    ee_orientation: tuple[float, float, float]  # roll, pitch, yaw in radians
    gripper_width: float  # meters
    is_moving: bool
    has_error: bool
    error_message: Optional[str]

    def to_dict(self) -> dict:
        return {
            "joint_positions_rad": self.joint_positions,
            "end_effector": {
                "position_m": {
                    "x": self.ee_position[0],
                    "y": self.ee_position[1],
                    "z": self.ee_position[2],
                },
                "orientation_rad": {
                    "roll": self.ee_orientation[0],
                    "pitch": self.ee_orientation[1],
                    "yaw": self.ee_orientation[2],
                },
            },
            "gripper_width_m": self.gripper_width,
            "is_moving": self.is_moving,
            "has_error": self.has_error,
            "error_message": self.error_message,
        }


class MockRobot:
    """Mock robot for testing without hardware."""

    def __init__(self):
        # Start in a reasonable home position
        self._joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self._ee_pos = np.array([0.4, 0.0, 0.3])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        self._has_error = False
        self._error_msg = None

    def get_state(self):
        """Mock state with q attribute."""
        class MockState:
            def __init__(self, joints):
                self.q = joints
                self.current_errors = None
        return MockState(self._joints)

    def get_position(self):
        return self._ee_pos.copy()

    def get_orientation(self):
        return self._ee_quat.copy()

    def move_to_joint_position(self, positions, speed_factor=0.2):
        self._joints = np.array(positions)
        logger.info(f"Mock joint move to {positions}")
        return True

    def move_to_pose(self, positions, orientations, speed_factor=0.2):
        # Handle sequences - move to final position
        if positions:
            self._ee_pos = np.array(positions[-1])
            self._ee_quat = np.array(orientations[-1])
            logger.info(f"Mock pose sequence: {len(positions)} waypoints, final={positions[-1]}")
        return True

    def get_robot(self):
        return self

    def recover_from_errors(self):
        self._has_error = False
        self._error_msg = None


class MockGripper:
    """Mock gripper for testing."""

    def __init__(self):
        self._width = 0.04

    def read_once(self):
        class MockGripperState:
            def __init__(self, width):
                self.width = width
        return MockGripperState(self._width)

    def move(self, width: float, speed: float = 0.1):
        self._width = np.clip(width, 0.0, 0.08)
        logger.info(f"Mock gripper move to {self._width}")
        return True

    def grasp(self, width: float, speed: float, force: float, epsilon_inner: float = 0.005, epsilon_outer: float = 0.005):
        self._width = width
        logger.info(f"Mock gripper grasp: width={width}, force={force}")
        return True

    def stop(self):
        return True


class FrankaController:
    """
    High-level controller for Franka Panda arm.

    Wraps panda-py with safety validation and cleaner interface.
    """

    ROBOT_IP = "192.168.0.253"  # Panda IP (direct connection via Pi eth0)

    def __init__(self, robot_ip: Optional[str] = None):
        self.robot_ip = robot_ip or self.ROBOT_IP
        self.validator = SafetyValidator()
        self._robot: Optional[Panda] = None
        self._gripper: Optional[Gripper] = None
        self._connected = False
        self._mock_mode = MOCK_MODE

    def connect(self) -> dict:
        """Connect to the robot."""
        if self._mock_mode:
            logger.info("Running in MOCK mode - no hardware connection")
            self._robot = MockRobot()
            self._gripper = MockGripper()
            self._connected = True
            return {"connected": True, "mock": True}

        try:
            self._robot = Panda(self.robot_ip)

            # Try to recover from any existing errors (may fail if robot not in right mode)
            try:
                self._robot.recover()
            except Exception as recovery_error:
                logger.warning(f"Auto-recovery skipped: {recovery_error}")

            self._gripper = Gripper(self.robot_ip)
            self._connected = True

            logger.info(f"Connected to Franka at {self.robot_ip}")
            return {"connected": True, "mock": False}

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return {"connected": False, "error": str(e)}

    def disconnect(self):
        """Disconnect from robot."""
        self._robot = None
        self._gripper = None
        self._connected = False

    def _call_with_timeout(self, fn, *args, timeout: float = STATE_TIMEOUT_S, label: str = "call", recover: bool = False, **kwargs):
        """
        Call a blocking panda-py/libfranka function with a timeout.

        Many panda-py calls can hang indefinitely if communication drops or
        the arm enters a bad state. This wraps them with a timeout.

        IMPORTANT: We must NOT use `with ThreadPoolExecutor` because the
        context manager calls shutdown(wait=True) on exit, which blocks
        until the zombie thread finishes — completely defeating the timeout.
        Instead we create the executor, submit, and shutdown(wait=False).
        """
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(f"{label} timed out after {timeout}s")
            if recover:
                try:
                    self._robot.recover()
                except Exception:
                    pass
            raise TimeoutError(f"{label} timed out after {timeout}s")
        finally:
            executor.shutdown(wait=False)

    def _move_joints_with_timeout(
        self, solution, speed_factor: float = 0.15, timeout: float = MOTION_TIMEOUT_S,
    ) -> bool:
        """Execute move_to_joint_position with a timeout and recovery."""
        return self._call_with_timeout(
            self._robot.move_to_joint_position, solution,
            speed_factor=speed_factor,
            timeout=timeout, label="Joint motion", recover=True,
        )

    @property
    def connected(self) -> bool:
        return self._connected

    def get_state(self) -> RobotState:
        """Get current robot state."""
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        state = self._call_with_timeout(
            self._robot.get_state, timeout=STATE_TIMEOUT_S, label="get_state")
        position = self._robot.get_position()
        orientation = self._robot.get_orientation()

        # Convert quaternion to Euler
        roll, pitch, yaw = quaternion_to_euler(orientation)

        # Check for errors
        has_error = False
        error_msg = None
        if hasattr(state, 'current_errors') and state.current_errors:
            has_error = True
            error_msg = str(state.current_errors)

        # Read gripper width with timeout (this is a common hang point)
        gripper_w = 0.04
        if self._gripper:
            try:
                gripper_state = self._call_with_timeout(
                    self._gripper.read_once, timeout=STATE_TIMEOUT_S, label="gripper_read")
                gripper_w = gripper_state.width
            except (TimeoutError, Exception) as e:
                logger.warning(f"Gripper read failed: {e}, using cached/default width")

        return RobotState(
            joint_positions=list(state.q),
            ee_position=(float(position[0]), float(position[1]), float(position[2])),
            ee_orientation=(roll, pitch, yaw),
            gripper_width=gripper_w,
            is_moving=False,  # panda-py moves are synchronous
            has_error=has_error,
            error_message=error_msg,
        )

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        confirmed: bool = False,
    ) -> dict:
        """
        Move end effector to Cartesian position.

        Args:
            x, y, z: Target position in meters
            roll, pitch, yaw: Target orientation in radians (optional, keeps current if not specified)
            confirmed: If True, skip confirmation for large moves

        Returns:
            dict with success status and any warnings/errors
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        config = get_safety_config()

        # Get current state for validation
        current_state = self.get_state()
        current_pos = current_state.ee_position

        # Use current orientation if not specified
        if roll is None:
            roll = current_state.ee_orientation[0]
        if pitch is None:
            pitch = current_state.ee_orientation[1]
        if yaw is None:
            yaw = current_state.ee_orientation[2]

        # Validate
        validation = self.validator.validate_cartesian_target(
            x, y, z, current_position=current_pos
        )

        if not validation["valid"]:
            return {
                "success": False,
                "errors": validation["errors"],
            }

        # Handle clamping
        if validation["clamped_position"]:
            x, y, z = validation["clamped_position"]

        # Check confirmation requirement
        if validation["requires_confirmation"] and not confirmed:
            distance = np.sqrt(
                (x - current_pos[0])**2 +
                (y - current_pos[1])**2 +
                (z - current_pos[2])**2
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "message": f"Large move ({distance:.3f}m) requires confirmation. "
                          f"Call again with confirmed=true to execute.",
                "target": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
                "warnings": validation["warnings"],
            }

        # Dry run check
        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry run - no movement executed",
                "would_move_to": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
                "warnings": validation["warnings"],
            }

        # Execute movement
        try:
            # Convert Euler to quaternion for panda-py
            quat = euler_to_quaternion(roll, pitch, yaw)

            # panda-py expects lists of waypoints
            positions = [np.array([x, y, z])]
            orientations = [quat]

            success = self._robot.move_to_pose(
                positions,
                orientations,
                speed_factor=0.1  # Conservative speed
            )

            if success:
                return {
                    "success": True,
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
                    "warnings": validation["warnings"],
                }
            else:
                return {
                    "success": False,
                    "error": "Motion command returned false",
                }

        except Exception as e:
            logger.error(f"Move failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def move_cartesian_ik(
        self,
        x: float,
        y: float,
        z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        confirmed: bool = False,
        frozen_joints: Optional[dict] = None,
    ) -> dict:
        """
        Move end effector to Cartesian position using analytical IK.

        Same interface as move_cartesian but uses IK + move_to_joint_position,
        which can reliably reach table height (z~0.013) unlike the Cartesian planner.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        config = get_safety_config()
        current_state = self.get_state()
        current_pos = current_state.ee_position

        if roll is None:
            roll = current_state.ee_orientation[0]
        if pitch is None:
            pitch = current_state.ee_orientation[1]
        if yaw is None:
            yaw = current_state.ee_orientation[2]

        # Validate target
        validation = self.validator.validate_cartesian_target(
            x, y, z, current_position=current_pos
        )

        if not validation["valid"]:
            return {"success": False, "errors": validation["errors"]}

        if validation["clamped_position"]:
            x, y, z = validation["clamped_position"]

        if validation["requires_confirmation"] and not confirmed:
            distance = np.sqrt(
                (x - current_pos[0])**2 +
                (y - current_pos[1])**2 +
                (z - current_pos[2])**2
            )
            return {
                "success": False,
                "requires_confirmation": True,
                "message": f"Large move ({distance:.3f}m) requires confirmation.",
                "target": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
                "warnings": validation["warnings"],
            }

        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry run - no movement executed",
                "would_move_to": {"x": x, "y": y, "z": z},
                "warnings": validation["warnings"],
            }

        # Solve IK
        target_quat = euler_to_quaternion(roll, pitch, yaw)
        target_pos = np.array([x, y, z])
        current_joints = np.array(current_state.joint_positions)

        solution = _solve_ik(target_pos, target_quat, current_joints,
                             frozen_joints=frozen_joints)

        if solution is None:
            return {
                "success": False,
                "error": "No valid IK solution found for target pose",
                "target": {"x": x, "y": y, "z": z, "roll": roll, "pitch": pitch, "yaw": yaw},
            }

        # Verify solution via FK: check position and orientation errors
        if PANDA_PY_AVAILABLE:
            fk_pose = panda_py.fk(solution)
            fk_pos = fk_pose[:3, 3]
            pos_error = np.linalg.norm(fk_pos - target_pos)

            # Orientation error via rotation matrix difference
            R_target = quaternion_to_rotation_matrix(target_quat)
            R_fk = fk_pose[:3, :3]
            R_err = R_target.T @ R_fk
            angle_error = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

            if pos_error > 0.003:
                return {
                    "success": False,
                    "error": f"IK solution position error too large: {pos_error*1000:.1f}mm",
                }
            if angle_error > np.radians(5):
                return {
                    "success": False,
                    "error": f"IK solution orientation error too large: {np.degrees(angle_error):.1f}deg",
                }

        # Execute via joint motion
        try:
            success = self._move_joints_with_timeout(solution, speed_factor=0.15)

            new_state = self.get_state()
            return {
                "success": True,
                "method": "ik",
                "position": {"x": x, "y": y, "z": z},
                "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
                "actual_position": {
                    "x": round(new_state.ee_position[0], 4),
                    "y": round(new_state.ee_position[1], 4),
                    "z": round(new_state.ee_position[2], 4),
                },
                "warnings": validation["warnings"],
            }

        except Exception as e:
            logger.error(f"IK move failed: {e}")
            return {"success": False, "error": str(e)}

    def move_relative(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        confirmed: bool = False,
    ) -> dict:
        """Move end effector relative to current position."""
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        state = self.get_state()
        x = state.ee_position[0] + dx
        y = state.ee_position[1] + dy
        z = state.ee_position[2] + dz

        return self.move_cartesian_ik(
            x, y, z,
            roll=state.ee_orientation[0],
            pitch=state.ee_orientation[1],
            yaw=state.ee_orientation[2],
            confirmed=confirmed,
        )

    def move_joints(
        self,
        joints: list[float],
        confirmed: bool = False,
    ) -> dict:
        """Move to joint configuration."""
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        validation = self.validator.validate_joint_target(joints)

        if not validation["valid"]:
            return {
                "success": False,
                "errors": validation["errors"],
            }

        config = get_safety_config()
        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry run - no movement executed",
                "would_move_to": {"joints": joints},
            }

        try:
            success = self._move_joints_with_timeout(
                np.array(joints), speed_factor=0.2,
            )

            if success:
                return {
                    "success": True,
                    "joints": joints,
                }
            else:
                return {
                    "success": False,
                    "error": "Joint motion command returned false",
                }

        except Exception as e:
            logger.error(f"Joint move failed: {e}")
            return {"success": False, "error": str(e)}

    def move_cartesian_sequence(
        self,
        waypoints: list[dict],
        speed_factor: float = 0.1,
    ) -> dict:
        """
        Execute a sequence of Cartesian waypoints as a smooth trajectory.

        Each waypoint is a dict with x, y, z (required) and optional roll, pitch, yaw.
        All waypoints are passed to panda-py at once for continuous motion.

        Args:
            waypoints: List of waypoint dicts with position and optional orientation
            speed_factor: Motion speed (0.0 to 1.0)

        Returns:
            dict with success status
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        if not waypoints:
            return {"success": False, "error": "No waypoints provided"}

        config = get_safety_config()
        current_state = self.get_state()

        # Get current orientation as default
        current_roll, current_pitch, current_yaw = current_state.ee_orientation

        # Validate all waypoints and build position/orientation lists
        positions = []
        orientations = []
        all_warnings = []

        for i, wp in enumerate(waypoints):
            x = wp.get("x")
            y = wp.get("y")
            z = wp.get("z")

            if x is None or y is None or z is None:
                return {"success": False, "error": f"Waypoint {i} missing x, y, or z"}

            # Validate position
            validation = self.validator.validate_cartesian_target(x, y, z)

            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Waypoint {i} invalid",
                    "errors": validation["errors"],
                }

            # Apply clamping if needed
            if validation["clamped_position"]:
                x, y, z = validation["clamped_position"]

            all_warnings.extend(validation.get("warnings", []))

            # Get orientation (use current if not specified)
            roll = wp.get("roll", current_roll)
            pitch = wp.get("pitch", current_pitch)
            yaw = wp.get("yaw", current_yaw)

            positions.append(np.array([x, y, z]))
            orientations.append(euler_to_quaternion(roll, pitch, yaw))

        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry run - no movement executed",
                "waypoint_count": len(waypoints),
                "warnings": all_warnings,
            }

        try:
            # panda-py's move_to_pose may return None on success
            self._robot.move_to_pose(
                positions,
                orientations,
                speed_factor=speed_factor,
            )

            return {
                "success": True,
                "waypoints_executed": len(waypoints),
                "warnings": all_warnings if all_warnings else None,
            }

        except Exception as e:
            logger.error(f"Sequence move failed: {e}")
            return {"success": False, "error": str(e)}

    def move_joint_sequence(
        self,
        configurations: list[list[float]],
        speed_factor: float = 0.2,
    ) -> dict:
        """
        Execute a sequence of joint configurations as smooth motion.

        Args:
            configurations: List of 7-element joint angle arrays (radians)
            speed_factor: Motion speed (0.0 to 1.0)

        Returns:
            dict with success status
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        if not configurations:
            return {"success": False, "error": "No configurations provided"}

        config = get_safety_config()

        # Validate all configurations
        for i, joints in enumerate(configurations):
            if len(joints) != 7:
                return {"success": False, "error": f"Configuration {i} must have 7 joints"}

            validation = self.validator.validate_joint_target(joints)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": f"Configuration {i} invalid",
                    "errors": validation["errors"],
                }

        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message": "Dry run - no movement executed",
                "configuration_count": len(configurations),
            }

        try:
            # Execute each configuration in sequence
            # Note: panda-py's move_to_joint_position is per-waypoint,
            # so we execute them in a tight loop for smoothest motion
            for i, joints in enumerate(configurations):
                success = self._move_joints_with_timeout(
                    np.array(joints), speed_factor=speed_factor,
                )
                if not success:
                    return {
                        "success": False,
                        "error": f"Motion failed at configuration {i}",
                        "completed": i,
                    }

            return {
                "success": True,
                "configurations_executed": len(configurations),
            }

        except Exception as e:
            logger.error(f"Joint sequence failed: {e}")
            return {"success": False, "error": str(e)}

    def gripper_move(self, width: float, speed: float = 0.1) -> dict:
        """Move gripper to specified width."""
        if not self._gripper:
            return {"success": False, "error": "Gripper not available"}

        validation = self.validator.validate_gripper_command(width)
        if not validation["valid"]:
            return {"success": False, "errors": validation["errors"]}

        config = get_safety_config()
        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "would_set_width": width,
            }

        try:
            success = self._call_with_timeout(
                self._gripper.move, width, speed,
                timeout=GRIPPER_TIMEOUT_S, label="gripper_move")
            return {"success": success, "width": width}
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gripper_grasp(
        self,
        width: float,
        force: float = 20.0,
        speed: float = 0.1,
    ) -> dict:
        """Grasp with specified parameters."""
        if not self._gripper:
            return {"success": False, "error": "Gripper not available"}

        validation = self.validator.validate_gripper_command(width, force)
        if not validation["valid"]:
            return {"success": False, "errors": validation["errors"]}

        config = get_safety_config()
        if config.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "would_grasp": {"width": width, "force": force},
            }

        try:
            success = self._call_with_timeout(
                self._gripper.grasp, width, speed, force,
                timeout=GRIPPER_TIMEOUT_S, label="gripper_grasp")
            return {"success": success, "width": width, "force": force}
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop(self) -> dict:
        """Stop current motion."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        try:
            if self._gripper:
                self._gripper.stop()
            return {"success": True, "message": "Motion stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def recover(self) -> dict:
        """Recover from error state."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        try:
            self._robot.recover()
            return {"success": True, "message": "Recovered from errors"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def pick_at(
        self,
        x: float,
        y: float,
        z: float = 0.013,
        grasp_width: float = 0.03,
        grasp_force: float = 70.0,
        x_offset: float = 0.0,
        approach_height: float = 0.15,
        yaw: float = 0.0,
    ) -> dict:
        """
        Pick an object at the given robot coordinates.

        Executes: open gripper -> approach from above -> lower -> grasp -> lift.

        Args:
            x, y: Target position in robot frame (meters)
            z: Grasp height (default: table height 0.013)
            grasp_width: Expected object width for grasp (meters)
            grasp_force: Grasp force in Newtons
            x_offset: Systematic X offset to compensate homography error
            approach_height: Height to approach/retreat from
            yaw: Wrist rotation in radians (0=default, pi/2=rotated 90 degrees)
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        steps = []
        target_x = x + x_offset

        # Top-down picking orientation (roll=pi, pitch=0, yaw configurable)
        import math
        pick_roll = math.pi
        pick_pitch = 0.0
        pick_yaw = yaw

        # SAWM data collection: start approach if collector is active
        sawm_active = hasattr(self, '_sawm_collector') and self._sawm_collector is not None
        if sawm_active:
            self._sawm_collector.start_approach(target_robot_xy=(x, y))  # use original position

        # SAWM monitor: start background inference if monitor is loaded
        sawm_monitor = hasattr(self, '_sawm_monitor') and self._sawm_monitor is not None
        if sawm_monitor:
            self._sawm_monitor.start(target_xy=(x, y))  # use original position, not offset

        # Step 1: Open gripper
        result = self.gripper_move(0.08)
        steps.append({"action": "open_gripper", "result": result})

        # Step 2: Move above target (IK with straight-down orientation)
        result = self.move_cartesian_ik(
            target_x, y, approach_height,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
        steps.append({"action": "approach_above", "method": "ik", "result": result})

        # SAWM: record frame at approach height
        if sawm_active:
            self._sawm_record_frame()

        # Step 3: Lower in increments (avoid large joint changes that trigger reflex)
        state = self.get_state()
        current_z = state.ee_position[2]
        step_z = 0.04  # 4cm increments
        while current_z - step_z > z:
            intermediate_z = current_z - step_z
            result = self.move_cartesian_ik(
                target_x, y, intermediate_z,
                roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
            steps.append({"action": "lower_step", "method": "ik", "target_z": round(intermediate_z, 4), "result": result})
            state = self.get_state()
            current_z = state.ee_position[2]

            # SAWM: record frame at each Z step
            if sawm_active:
                self._sawm_record_frame()

            # SAWM monitor: check for correction (clamped to ±15mm)
            if sawm_monitor:
                correction = self._sawm_monitor.get_correction()
                if correction:
                    cdx, cdy = correction
                    MAX_CORRECTION = 0.015  # 15mm max per step
                    cdx = max(-MAX_CORRECTION, min(MAX_CORRECTION, cdx))
                    cdy = max(-MAX_CORRECTION, min(MAX_CORRECTION, cdy))
                    target_x += cdx
                    y += cdy
                    steps.append({
                        "action": "sawm_correction",
                        "dx_mm": round(cdx * 1000, 1),
                        "dy_mm": round(cdy * 1000, 1),
                    })
                    logger.info(f"SAWM correction applied: dx={cdx*1000:.1f}mm dy={cdy*1000:.1f}mm")

        # Final lower to grasp height
        result = self.move_cartesian_ik(
            target_x, y, z,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
        steps.append({"action": "lower_to_grasp", "method": "ik", "result": result})
        state = self.get_state()
        actual_z = state.ee_position[2]
        steps.append({"action": "check_z", "target_z": z, "actual_z": round(actual_z, 4)})

        # SAWM: record frame at grasp height (closest frame)
        if sawm_active:
            self._sawm_record_frame()

        # Step 4: Grasp
        result = self.gripper_grasp(width=grasp_width, force=grasp_force)
        steps.append({"action": "grasp", "result": result})
        # Check actual gripper width
        state = self.get_state()
        actual_grip = state.gripper_width
        # Grasped if gripper stopped between fully-closed and target+tolerance
        # If gripper < 1mm, it closed fully (nothing grasped)
        # If gripper > target + 5mm, object was wider than expected
        min_grip = 0.001  # 1mm - below this means fully closed / empty
        grasped = actual_grip > min_grip and actual_grip < grasp_width + 0.005
        steps.append({"action": "check_grasp", "gripper_width": round(actual_grip, 4), "grasped": grasped})

        # SAWM: end approach with grasp result
        if sawm_active:
            ee = self.get_state().ee_position
            self._sawm_collector.end_approach(
                success=grasped,
                final_gripper_xy=(ee[0], ee[1]),
            )

        # SAWM monitor: stop
        if sawm_monitor:
            self._sawm_monitor.stop()

        # Step 5: Lift (IK with straight-down orientation)
        state = self.get_state()
        result = self.move_cartesian_ik(
            state.ee_position[0], state.ee_position[1],
            state.ee_position[2] + approach_height,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
        steps.append({"action": "lift", "method": "ik", "result": result})
        state = self.get_state()

        return {
            "success": grasped,
            "gripper_width": round(actual_grip, 4),
            "final_position": {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
            },
            "steps": steps,
        }

    def _sawm_record_frame(self):
        """Capture a camera frame and record it for SAWM data collection."""
        try:
            from camera_daemon.client import get_camera_client
            client = get_camera_client()
            frame = client.get_frame()
            if frame is None:
                logger.warning("SAWM: no camera frame available")
                return

            state = self.get_state()
            ee = state.ee_position
            joints = np.array(state.joint_positions)

            self._sawm_collector.record_frame(
                frame=frame,
                gripper_robot_xy=(ee[0], ee[1]),
                gripper_z=ee[2],
                joints=joints,
            )
        except Exception as e:
            logger.warning(f"SAWM frame recording failed: {e}")

    def place_at(
        self,
        x: float,
        y: float,
        z: float = 0.08,
        approach_height: float = 0.15,
        yaw: float = 0.0,
    ) -> dict:
        """
        Place a held object at the given robot coordinates.

        Executes: move above target -> lower -> release -> retreat up.

        Args:
            x, y: Target position in robot frame (meters)
            z: Place height (default: 0.08 for gentle placement)
            approach_height: Height to approach/retreat from
            yaw: Wrist rotation in radians (0=default, pi/2=rotated 90 degrees)
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        steps = []

        # Top-down orientation for placing (yaw configurable)
        import math
        place_roll = math.pi
        place_pitch = 0.0
        place_yaw = yaw

        # Step 1: Move above target (IK with straight-down orientation)
        result = self.move_cartesian_ik(
            x, y, approach_height,
            roll=place_roll, pitch=place_pitch, yaw=place_yaw, confirmed=True)
        steps.append({"action": "move_above", "method": "ik", "result": result})

        # Step 2: Lower to place height (IK for reliable low-Z reach)
        result = self.move_cartesian_ik(
            x, y, z,
            roll=place_roll, pitch=place_pitch, yaw=place_yaw, confirmed=True)
        steps.append({"action": "lower_to_place", "method": "ik", "result": result})

        # Step 3: Open gripper to release
        result = self.gripper_move(0.08)
        steps.append({"action": "release", "result": result})

        # Step 4: Retreat up (IK with straight-down orientation)
        state = self.get_state()
        result = self.move_cartesian_ik(
            state.ee_position[0], state.ee_position[1],
            state.ee_position[2] + approach_height,
            roll=place_roll, pitch=place_pitch, yaw=place_yaw, confirmed=True)
        steps.append({"action": "retreat", "method": "ik", "result": result})
        state = self.get_state()

        return {
            "success": True,
            "final_position": {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
            },
            "steps": steps,
        }

    def start_recording(self, language_instruction: str, fps: int = 30) -> dict:
        """Start trajectory recording for current episode."""
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        from common.trajectory_recorder import get_recorder
        recorder = get_recorder(fps=fps)
        return recorder.start_episode(language_instruction, self)

    def stop_recording(self, success: bool) -> dict:
        """Stop recording and save episode."""
        from common.trajectory_recorder import get_recorder
        recorder = get_recorder()
        return recorder.stop_episode(success)

    def get_recording_status(self) -> dict:
        """Check if recording is active + stats."""
        from common.trajectory_recorder import get_recorder
        recorder = get_recorder()
        return recorder.get_status()

    def list_episodes(self) -> dict:
        """List recorded episodes."""
        from common.trajectory_recorder import get_recorder
        recorder = get_recorder()
        return recorder.list_episodes()

    # --- Gamepad jog mode ---

    def start_jog(self, record: bool = False, remote: bool = False) -> dict:
        """Start gamepad jog mode. Polls gamepad and executes small moves.

        Args:
            record: If True, capture frames during jogging for SAWM training.
                    Frames are recorded every 0.5s or 10mm of movement.
                    On successful grasp (A button), the approach is saved
                    and a new one starts automatically.
            remote: If True, start a WebSocket server for browser-based control
                    instead of using a local USB gamepad.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if hasattr(self, '_jog_active') and self._jog_active:
            return {"success": False, "error": "Jog already active"}

        if remote:
            try:
                from .web_gamepad import WebGamepad
            except (ImportError, ModuleNotFoundError):
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location(
                    "web_gamepad", __file__.replace("controller.py", "web_gamepad.py"))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                WebGamepad = _mod.WebGamepad
            self._gamepad = WebGamepad()
        else:
            try:
                from .gamepad import GamepadHandler
            except (ImportError, ModuleNotFoundError):
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location(
                    "gamepad", __file__.replace("controller.py", "gamepad.py"))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                GamepadHandler = _mod.GamepadHandler
            self._gamepad = GamepadHandler()
        result = self._gamepad.start()
        if not result["success"]:
            return result

        self._jog_active = True
        self._jog_recording = record
        self._jog_frozen_joints = None
        self._jog_last_pitch = 0.0
        self._jog_last_yaw = 0.0

        self._jog_thread = threading.Thread(target=self._jog_loop, daemon=True)
        self._jog_thread.start()

        response = {
            "success": True,
            "message": "Gamepad jog mode enabled" + (" (RECORDING)" if record else ""),
            "recording": record,
            "remote": remote,
            "controller": result.get("controller", ""),
            "controls": {
                "left_stick": "X-Y motion",
                "right_stick_y": "Z motion (up/down)",
                "dpad_up_down": "Pitch (tilt gripper forward/back)",
                "dpad_left_right": "Yaw (rotate gripper)",
                "A": "Grasp (30mm, 70N)" + (" — ends approach, labels data" if record else ""),
                "B": "Open gripper" + (" — starts new approach" if record else ""),
                "X": "Cycle speed (slow/medium/fast)",
                "Y": "Reset pitch/yaw to zero (straight down)",
                "LB_hold": "Fine mode (1mm steps, 0.6° angle steps)",
                "Back": "Stop jog mode",
            },
            "speed": "medium (5mm/step)",
        }
        if remote:
            response["websocket_port"] = result.get("port", 8766)
            response["message"] += f" — WebSocket on port {result.get('port', 8766)}"
        return response

    def _jog_loop(self):
        """Background thread: read gamepad, execute moves."""
        logger.info("Jog loop started")
        move_interval = 0.05  # ~20Hz max move rate

        # Recording state
        recording = getattr(self, '_jog_recording', False)
        collector = None
        camera = None
        last_record_time = 0.0
        last_record_xy = (0.0, 0.0)
        record_interval = 0.5  # seconds between frame captures
        record_dist = 0.01    # 10mm minimum movement between captures

        if recording:
            try:
                from common.sawm.servo_collector import get_servo_collector
                from camera_daemon.client import get_camera_client
                collector = get_servo_collector()
                camera = get_camera_client()
                camera.connect()
                # Start first approach
                current = self.get_state()
                collector.start_approach(
                    target_hint_xy=(current.ee_position[0], current.ee_position[1])
                )
                logger.info("Jog recording: started first approach")
            except Exception as e:
                logger.error(f"Failed to initialize jog recording: {e}")
                recording = False

        while self._jog_active:
            try:
                state = self._gamepad.get_state()

                # Handle button events first
                for event in state.events:
                    if event.action == "stop_jog":
                        logger.info("Jog stopped via gamepad Back button")
                        self._jog_active = False
                        break
                    elif event.action == "grasp":
                        try:
                            self.gripper_grasp(width=0.03, force=70, speed=0.1)
                        except Exception as e:
                            logger.warning(f"Grasp failed: {e}")
                        # Check if grasp succeeded (for recording)
                        # Do this even if gripper_grasp threw — end the approach either way
                        if recording and collector and collector.active:
                            time.sleep(0.1)  # Let gripper settle
                            rs = self.get_state()
                            gw = rs.gripper_width
                            # Holding something: not fully closed (<2mm) and not fully open (>70mm)
                            grasped = gw > 0.002 and gw < 0.070
                            gxy = (rs.ee_position[0], rs.ee_position[1])
                            result = collector.end_approach(
                                success=grasped, final_gripper_xy=gxy
                            )
                            logger.info(
                                f"Jog recording: approach ended — "
                                f"{'SUCCESS' if grasped else 'FAIL'} "
                                f"(gripper_width={gw:.4f}), "
                                f"{result.get('frames', 0)} frames"
                            )
                    elif event.action == "open_gripper":
                        try:
                            self.gripper_move(0.08)
                            # Start a new approach if recording
                            if recording and collector and not collector.active:
                                current = self.get_state()
                                collector.start_approach(
                                    target_hint_xy=(current.ee_position[0], current.ee_position[1])
                                )
                                last_record_time = 0.0
                                last_record_xy = (0.0, 0.0)
                                logger.info("Jog recording: started new approach")
                        except Exception as e:
                            logger.warning(f"Gripper open failed: {e}")
                    elif event.action == "home":
                        try:
                            logger.info("Jog: returning to home position")
                            self._move_joints_with_timeout(
                                [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
                                speed_factor=0.2,
                            )
                            if hasattr(self, '_jog_frozen_joints'):
                                self._jog_frozen_joints = None
                        except Exception as e:
                            logger.warning(f"Home failed: {e}")

                if not self._jog_active:
                    break

                # Execute move if sticks or D-pad are displaced
                has_motion = state.dx != 0 or state.dy != 0 or state.dz != 0
                # D-pad: detect changes in browser's accumulated pitch/yaw
                pitch_delta = state.pitch - self._jog_last_pitch
                yaw_delta = state.yaw - self._jog_last_yaw
                has_orientation = abs(pitch_delta) > 0.001 or abs(yaw_delta) > 0.001

                if has_motion or has_orientation:
                    try:
                        # Auto-recover from error state before moving
                        current = self.get_state()
                        if current.has_error:
                            logger.warning(f"Jog: auto-recovering from {current.error_message}")
                            try:
                                self._robot.recover()
                                time.sleep(0.2)
                            except Exception as re:
                                logger.error(f"Jog: recovery failed: {re}")
                                time.sleep(1.0)
                                continue
                        target_x = current.ee_position[0] + state.dx
                        target_y = current.ee_position[1] + state.dy
                        target_z = current.ee_position[2] + state.dz
                        # Always use current arm orientation + any d-pad delta
                        # This prevents drift and avoids fighting accumulated error
                        target_pitch = current.ee_orientation[1] + pitch_delta
                        target_yaw = current.ee_orientation[2] + yaw_delta

                        # Solve IK directly with current joints as seed
                        # (avoids double get_state() and threshold mismatch in move_cartesian_ik)
                        target_quat = euler_to_quaternion(
                            current.ee_orientation[0], target_pitch, target_yaw
                        )
                        target_pos = np.array([target_x, target_y, target_z])
                        current_joints = np.array(current.joint_positions)

                        solution = _solve_ik(
                            target_pos, target_quat, current_joints,
                            frozen_joints=getattr(self, '_jog_frozen_joints', None),
                        )

                        if solution is not None:
                            self._move_joints_with_timeout(
                                solution, speed_factor=0.15, timeout=5.0,
                            )
                            self._jog_ik_fail_count = 0
                        else:
                            self._jog_ik_fail_count = getattr(self, '_jog_ik_fail_count', 0) + 1
                            logger.warning(
                                f"Jog IK failed: no solution for "
                                f"target=({target_x:.3f},{target_y:.3f},{target_z:.3f}) "
                                f"pitch={target_pitch:.3f} yaw={target_yaw:.3f}"
                            )
                        self._jog_last_pitch = state.pitch
                        self._jog_last_yaw = state.yaw
                    except Exception as e:
                        logger.warning(f"Jog move failed: {e}")
                        try:
                            self._robot.recover()
                        except Exception:
                            pass
                else:
                    # No motion requested — clear IK blocked indicator
                    self._jog_ik_fail_count = 0

                # Push status to WebSocket clients (if remote gamepad)
                if hasattr(self._gamepad, 'update_status'):
                    try:
                        current_st = self.get_state()
                        self._gamepad.update_status(
                            position={
                                "x": round(current_st.ee_position[0], 4),
                                "y": round(current_st.ee_position[1], 4),
                                "z": round(current_st.ee_position[2], 4),
                            },
                            gripper_width=current_st.gripper_width,
                            speed=state.speed_name,
                            joints=[round(q, 4) for q in current_st.joint_positions],
                            joint_limits=_JOINT_LIMITS.tolist(),
                            orientation={
                                "roll": round(current_st.ee_orientation[0], 4),
                                "pitch": round(current_st.ee_orientation[1], 4),
                                "yaw": round(current_st.ee_orientation[2], 4),
                            },
                            ik_blocked=getattr(self, '_jog_ik_fail_count', 0) > 0,
                        )
                    except Exception:
                        pass

                # Record frame if enough time/distance has passed
                if recording and collector and collector.active and camera:
                    now = time.time()
                    current = self.get_state()
                    gx, gy = current.ee_position[0], current.ee_position[1]
                    dist = ((gx - last_record_xy[0])**2 + (gy - last_record_xy[1])**2) ** 0.5
                    elapsed = now - last_record_time

                    if elapsed > record_interval or dist > record_dist:
                        try:
                            frame = camera.get_frame()
                            if frame is not None:
                                gz = current.ee_position[2]
                                pitch_val = state.pitch
                                # Use current gripper position as target estimate
                                # (human is guiding toward the target)
                                collector.record_frame(
                                    frame=frame,
                                    gripper_robot_xy=(gx, gy),
                                    target_estimate_xy=(gx, gy),
                                    gripper_z=gz,
                                    pitch=pitch_val,
                                )
                                last_record_time = now
                                last_record_xy = (gx, gy)
                        except Exception as e:
                            logger.warning(f"Jog record frame failed: {e}")

                time.sleep(move_interval)

            except Exception as e:
                logger.error(f"Jog loop error: {e}")
                time.sleep(0.5)

        # End any active approach on jog stop
        if recording and collector and collector.active:
            collector.end_approach(success=False, final_gripper_xy=(0, 0))
            logger.info("Jog recording: ended active approach (jog stopped)")

        # Cleanup
        if hasattr(self, '_gamepad') and self._gamepad:
            self._gamepad.stop()
        logger.info("Jog loop ended")

    def stop_jog(self) -> dict:
        """Stop gamepad jog mode."""
        if not hasattr(self, '_jog_active') or not self._jog_active:
            return {"success": False, "error": "Jog not active"}

        self._jog_active = False
        if hasattr(self, '_jog_thread') and self._jog_thread:
            self._jog_thread.join(timeout=3.0)

        # Get final position
        if self._connected:
            state = self.get_state()
            return {
                "success": True,
                "message": "Jog mode stopped",
                "final_position": {
                    "x": round(state.ee_position[0], 4),
                    "y": round(state.ee_position[1], 4),
                    "z": round(state.ee_position[2], 4),
                },
                "gripper_width": round(state.gripper_width, 4),
            }

        return {"success": True, "message": "Jog mode stopped"}

    def get_jog_status(self) -> dict:
        """Get current jog mode status."""
        active = hasattr(self, '_jog_active') and self._jog_active

        result = {"active": active}

        if active and hasattr(self, '_gamepad'):
            state = self._gamepad.get_state()
            result["speed"] = state.speed_name
            result["step_size_mm"] = state.step_size * 1000
            result["fine_mode"] = state.fine_mode
            result["controller"] = state.controller_name
            result["stick"] = {
                "dx": round(state.dx * 1000, 1),
                "dy": round(state.dy * 1000, 1),
                "dz": round(state.dz * 1000, 1),
            }
            result["orientation"] = {
                "pitch_rad": round(state.pitch, 3),
                "pitch_deg": round(state.pitch * 57.2958, 1),
                "yaw_rad": round(state.yaw, 3),
                "yaw_deg": round(state.yaw * 57.2958, 1),
            }

        if self._connected:
            s = self.get_state()
            result["position"] = {
                "x": round(s.ee_position[0], 4),
                "y": round(s.ee_position[1], 4),
                "z": round(s.ee_position[2], 4),
            }
            result["gripper_width"] = round(s.gripper_width, 4)

        return result

    # --- VLA autonomous control ---

    def start_vla(self, server_url: str, task: str) -> dict:
        """Start VLA autonomous control loop."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if hasattr(self, '_vla_client') and self._vla_client is not None:
            status = self._vla_client.get_status()
            if status.get("active"):
                return {"success": False, "error": "VLA already active"}

        # Stop jog if active (mutual exclusion)
        if hasattr(self, '_jog_active') and self._jog_active:
            self.stop_jog()

        from .vla_client import VLAClient
        self._vla_client = VLAClient(self, server_url, task)
        result = self._vla_client.start()

        if result.get("success"):
            result["controls"] = {
                "stop": "Call vla_disable to stop",
                "status": "Call vla_status to check progress",
            }

        return result

    def stop_vla(self) -> dict:
        """Stop VLA autonomous control."""
        if not hasattr(self, '_vla_client') or self._vla_client is None:
            return {"success": False, "error": "VLA not active"}

        result = self._vla_client.stop()

        # Include final position
        if self._connected:
            state = self.get_state()
            result["final_position"] = {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
            }
            result["gripper_width"] = round(state.gripper_width, 4)

        self._vla_client = None
        return result

    def get_vla_status(self) -> dict:
        """Get current VLA control status."""
        if not hasattr(self, '_vla_client') or self._vla_client is None:
            result = {"active": False}
        else:
            result = self._vla_client.get_status()

        if self._connected:
            state = self.get_state()
            result["position"] = {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
            }
            result["gripper_width"] = round(state.gripper_width, 4)

        return result

    def teaching_mode(self, active: bool) -> dict:
        """
        Enable/disable teaching mode (gravity compensation).

        When active, the arm goes compliant - motors compensate for gravity
        but don't resist external forces. The user can physically move the arm.
        This bypasses joint wall checks, so it can be used to recover from
        positions at or beyond joint limits.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if self._mock_mode:
            return {"success": True, "teaching_mode": active, "mock": True}

        try:
            self._robot.teaching_mode(active)
            if active:
                return {
                    "success": True,
                    "teaching_mode": True,
                    "message": "Teaching mode ON - arm is compliant. "
                              "Physically move joints as needed, then disable teaching mode.",
                }
            else:
                return {
                    "success": True,
                    "teaching_mode": False,
                    "message": "Teaching mode OFF - arm is under position control again.",
                }
        except Exception as e:
            logger.error(f"Teaching mode failed: {e}")
            return {"success": False, "error": str(e)}

    # --- Skill episode logging ---

    def start_skill_episode(self, task: str) -> dict:
        """Start logging skill calls for VLM training data collection."""
        from common.skill_logger import SkillLogger
        if hasattr(self, '_skill_logger') and self._skill_logger and self._skill_logger.active:
            return {"success": False, "error": "Skill episode already active"}
        self._skill_logger = SkillLogger()
        return self._skill_logger.start_episode(task)

    def stop_skill_episode(self, success: bool) -> dict:
        """Stop skill episode logging, capture final 'done' frame."""
        if not hasattr(self, '_skill_logger') or not self._skill_logger or not self._skill_logger.active:
            return {"success": False, "error": "No active skill episode"}
        result = self._skill_logger.end_episode(success)
        return result

    def get_skill_episode_status(self) -> dict:
        """Get current skill episode logging status."""
        if not hasattr(self, '_skill_logger') or not self._skill_logger:
            return {"active": False}
        return self._skill_logger.get_status()

    def list_skill_episodes(self) -> dict:
        """List all collected skill episodes."""
        from common.skill_logger import SkillLogger
        if not hasattr(self, '_skill_logger') or not self._skill_logger:
            self._skill_logger = SkillLogger()
        return self._skill_logger.list_episodes()

    # --- SAWM data collection ---

    def sawm_enable(self, model_path: str) -> dict:
        """Enable SAWM inference monitor for trajectory correction during picks."""
        try:
            from common.sawm.monitor import SAWMMonitor
            self._sawm_monitor = SAWMMonitor(model_path)
            return {
                "success": True,
                "message": f"SAWM monitor loaded from {model_path}. Will correct pick trajectories.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sawm_disable(self) -> dict:
        """Disable SAWM inference monitor."""
        if not hasattr(self, '_sawm_monitor') or self._sawm_monitor is None:
            return {"success": False, "error": "SAWM monitor not active"}

        if self._sawm_monitor.active:
            self._sawm_monitor.stop()

        status = self._sawm_monitor.get_status()
        self._sawm_monitor = None
        return {
            "success": True,
            "message": "SAWM monitor disabled.",
            "final_status": status,
        }

    def sawm_status(self) -> dict:
        """Get SAWM monitor status."""
        if not hasattr(self, '_sawm_monitor') or self._sawm_monitor is None:
            return {"active": False, "monitor_loaded": False}
        result = self._sawm_monitor.get_status()
        result["monitor_loaded"] = True
        return result

    def sawm_collect_enable(self) -> dict:
        """Enable SAWM data collection during pick_at() calls."""
        from common.sawm.data_collector import get_collector
        self._sawm_collector = get_collector()
        return {
            "success": True,
            "message": "SAWM data collection enabled. Frames will be recorded during pick_at().",
            "stats": self._sawm_collector.get_stats(),
        }

    def sawm_collect_disable(self) -> dict:
        """Disable SAWM data collection."""
        if not hasattr(self, '_sawm_collector') or self._sawm_collector is None:
            return {"success": False, "error": "SAWM collection not active"}

        # End any in-progress approach
        if self._sawm_collector.active:
            self._sawm_collector.end_approach(success=False, final_gripper_xy=(0, 0))

        stats = self._sawm_collector.get_stats()
        self._sawm_collector = None
        return {
            "success": True,
            "message": "SAWM data collection disabled.",
            "stats": stats,
        }

    def sawm_collect_stats(self) -> dict:
        """Get SAWM data collection statistics."""
        from common.sawm.data_collector import get_collector
        collector = self._sawm_collector if hasattr(self, '_sawm_collector') and self._sawm_collector else get_collector()
        return collector.get_stats()

    # --- Visual servo pick ---

    def servo_pick_at(
        self,
        x: float,
        y: float,
        grasp_width: float = 0.03,
        grasp_force: float = 70.0,
        grasp_z: float = 0.013,
        approach_height: float = 0.15,
        servo_z: float = 0.05,
        servo_pitch: float = 0.3,
        gain: float = 0.5,
        max_step: float = 0.03,
        convergence_threshold: float = 0.015,
        max_iterations: int = 20,
        model_path: Optional[str] = None,
        collect_data: bool = True,
    ) -> dict:
        """
        Pick an object using visual servo approach.

        Two-phase pick: (1) coarse visual servo with tilted gripper visible
        to camera, (2) untilt + vertical descent + grasp.

        In fallback mode (no model), moves toward hint in fractional steps
        and still generates training data.

        Args:
            x, y: Rough position hint in robot frame (meters)
            grasp_width: Expected object width (meters)
            grasp_force: Grasp force (N)
            grasp_z: Grasp height (meters)
            approach_height: Lift height after grasp (meters)
            servo_z: Height during servo (meters)
            servo_pitch: Forward tilt during servo (radians)
            gain: Fraction of predicted offset to move each step
            max_step: Maximum step size (meters)
            convergence_threshold: Stop when offset below this (meters)
            max_iterations: Maximum servo iterations
            model_path: Path to SAWM ONNX model (None = fallback mode)
            collect_data: Whether to record frames for training
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        from common.sawm.servo import VisualServoLoop, ServoConfig

        config = ServoConfig(
            servo_z=servo_z,
            servo_pitch=servo_pitch,
            gain=gain,
            max_step=max_step,
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
        )

        # Use controller-level model_path if set
        effective_model = model_path
        if effective_model is None and hasattr(self, '_servo_model_path'):
            effective_model = self._servo_model_path

        servo = VisualServoLoop(
            controller=self,
            config=config,
            model_path=effective_model,
        )

        return servo.execute(
            target_x_hint=x,
            target_y_hint=y,
            grasp_width=grasp_width,
            grasp_force=grasp_force,
            grasp_z=grasp_z,
            approach_height=approach_height,
            collect_data=collect_data,
        )

    # --- Plan executor ---

    def execute_plan(self, steps: list[dict]) -> dict:
        """
        Execute a sequence of skill commands back-to-back with no inter-step
        latency. Claude plans the full sequence, robot executes it all at once.

        Supported skills:
            pick  — pick_at(x, y, z?, grasp_width?, grasp_force?, approach_height?)
            place — place_at(x, y, z?, approach_height?)
            move  — move_cartesian_ik(x, y, z)
            open_gripper — gripper_move(width?)
            grasp — gripper_grasp(width?, force?)
            home  — move to safe home position above table
            wait  — pause for seconds
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if not steps:
            return {"success": False, "error": "No steps provided"}

        results = []
        t_plan_start = time.time()

        # Check if skill logging is active
        _logging = hasattr(self, '_skill_logger') and self._skill_logger and self._skill_logger.active

        for i, step in enumerate(steps):
            skill = step.get("skill")
            t_step = time.time()
            logger.info(f"Plan step {i+1}/{len(steps)}: {skill} {step}")

            # Log skill BEFORE execution (captures pre-action frame)
            step_params = {k: v for k, v in step.items() if k != "skill"}
            if _logging:
                self._skill_logger.log_skill(skill, step_params)

            try:
                if skill == "pick":
                    r = self.pick_at(
                        x=step["x"],
                        y=step["y"],
                        z=step.get("z", 0.013),
                        grasp_width=step.get("grasp_width", 0.03),
                        grasp_force=step.get("grasp_force", 70.0),
                        x_offset=step.get("x_offset", 0.0),
                        approach_height=step.get("approach_height", 0.15),
                        yaw=step.get("yaw", 0.0),
                    )

                elif skill == "place":
                    r = self.place_at(
                        x=step["x"],
                        y=step["y"],
                        z=step.get("z", 0.08),
                        approach_height=step.get("approach_height", 0.15),
                        yaw=step.get("yaw", 0.0),
                    )

                elif skill == "move":
                    r = self.move_cartesian_ik(
                        x=step["x"],
                        y=step["y"],
                        z=step["z"],
                        roll=step.get("roll"),
                        pitch=step.get("pitch"),
                        yaw=step.get("yaw"),
                        confirmed=True,
                    )

                elif skill == "open_gripper":
                    r = self.gripper_move(width=step.get("width", 0.08))

                elif skill == "grasp":
                    r = self.gripper_grasp(
                        width=step.get("width", 0.03),
                        force=step.get("force", 70.0),
                        speed=step.get("speed", 0.1),
                    )

                elif skill == "home":
                    # First move up and unrotate yaw in steps to avoid IK failures
                    state = self.get_state()
                    cur_yaw = state.ee_orientation[2]
                    # If wrist is significantly rotated, step through intermediate yaw
                    if abs(cur_yaw) > 0.3:
                        self.move_cartesian_ik(
                            x=state.ee_position[0], y=state.ee_position[1], z=0.35,
                            roll=3.14159, pitch=0.0, yaw=cur_yaw,
                            confirmed=True,
                        )
                        # Step to half yaw
                        half_yaw = cur_yaw / 2
                        self.move_cartesian_ik(
                            x=0.4, y=0.0, z=0.35,
                            roll=3.14159, pitch=0.0, yaw=half_yaw,
                            confirmed=True,
                        )
                    r = self.move_cartesian_ik(
                        x=0.307, y=0.0, z=0.487,
                        roll=3.14159, pitch=0.0, yaw=0.0,
                        confirmed=True,
                    )

                elif skill == "wait":
                    seconds = step.get("seconds", 1.0)
                    time.sleep(seconds)
                    r = {"success": True, "waited": seconds}

                else:
                    r = {"success": False, "error": f"Unknown skill: {skill}"}

                # Update skill log with execution result
                if _logging:
                    self._skill_logger.update_last_result(r)

                step_ms = (time.time() - t_step) * 1000
                results.append({
                    "step": i + 1,
                    "skill": skill,
                    "duration_ms": round(step_ms),
                    "result": r,
                })

                # Stop on failure
                if not r.get("success", True):
                    logger.warning(f"Plan step {i+1} failed: {r}")
                    break

            except Exception as e:
                logger.error(f"Plan step {i+1} exception: {e}")
                results.append({
                    "step": i + 1,
                    "skill": skill,
                    "error": str(e),
                })
                break

        total_ms = (time.time() - t_plan_start) * 1000
        completed = len(results)
        all_ok = all(r.get("result", {}).get("success", True) for r in results if "error" not in r)

        # Final position
        final_pos = None
        if self._connected:
            state = self.get_state()
            final_pos = {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
                "gripper_width": round(state.gripper_width, 4),
            }

        return {
            "success": all_ok,
            "steps_completed": completed,
            "steps_total": len(steps),
            "total_duration_ms": round(total_ms),
            "final_position": final_pos,
            "results": results,
        }


# Singleton controller
_controller: Optional[FrankaController] = None


def get_controller() -> FrankaController:
    global _controller
    if _controller is None:
        _controller = FrankaController()
    return _controller
