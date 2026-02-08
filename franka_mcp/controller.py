"""
Franka Panda arm controller wrapper.

Provides a safe, high-level interface to panda-py/libfranka.
"""

import os
import logging
import threading
import time
from typing import Optional
from dataclasses import dataclass
import numpy as np

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


def _solve_ik(
    position: np.ndarray,
    orientation_quat: np.ndarray,
    current_joints: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Solve analytical IK for a target Cartesian pose.

    Args:
        position: Target [x, y, z] in meters
        orientation_quat: Target quaternion [x, y, z, w]
        current_joints: Current 7 joint angles (used as seed)

    Returns:
        Best joint solution (7,) or None if no valid solution found.
    """
    if not PANDA_PY_AVAILABLE:
        return None

    # Build 4x4 SE(3) target pose
    R = quaternion_to_rotation_matrix(orientation_quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position

    # Panda joint limits for filtering
    joint_limits = np.array([
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ])

    best_solution = None
    best_cost = float('inf')

    # Try multiple q_7 values to explore the null space
    q7_candidates = [0.785, 0.0, -0.785, 1.57]

    for q7 in q7_candidates:
        try:
            solutions = panda_py.ik_full(T, current_joints, q7)
        except Exception as e:
            logger.debug(f"IK failed for q7={q7}: {e}")
            continue

        if solutions is None or len(solutions) == 0:
            continue

        for sol in solutions:
            sol = np.array(sol)
            if sol.shape != (7,):
                continue

            # Check joint limits (with small margin)
            margin = 0.01
            in_limits = True
            for i in range(7):
                if sol[i] < joint_limits[i, 0] + margin or sol[i] > joint_limits[i, 1] - margin:
                    in_limits = False
                    break
            if not in_limits:
                continue

            # Verify with FK: position error < 2mm
            try:
                fk_pose = panda_py.fk(sol)
                fk_pos = fk_pose[:3, 3]
                pos_error = np.linalg.norm(fk_pos - position)
                if pos_error > 0.002:
                    continue
            except Exception:
                continue

            # Cost: minimize joint travel + prefer q7 near 0.785 (good picking orientation)
            cost = np.sum((sol - current_joints) ** 2) + 2.0 * (sol[6] - 0.785) ** 2
            if cost < best_cost:
                best_cost = cost
                best_solution = sol

    if best_solution is not None:
        logger.info(f"IK solution found (cost={best_cost:.3f}, q7={best_solution[6]:.3f})")
    else:
        logger.warning("No valid IK solution found")

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

    @property
    def connected(self) -> bool:
        return self._connected

    def get_state(self) -> RobotState:
        """Get current robot state."""
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        state = self._robot.get_state()
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

        return RobotState(
            joint_positions=list(state.q),
            ee_position=(float(position[0]), float(position[1]), float(position[2])),
            ee_orientation=(roll, pitch, yaw),
            gripper_width=self._gripper.read_once().width if self._gripper else 0.04,
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

        solution = _solve_ik(target_pos, target_quat, current_joints)

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

            if pos_error > 0.002:
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
            success = self._robot.move_to_joint_position(
                solution,
                speed_factor=0.15,
            )

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

        return self.move_cartesian(
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
            success = self._robot.move_to_joint_position(
                np.array(joints),
                speed_factor=0.2
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
                success = self._robot.move_to_joint_position(
                    np.array(joints),
                    speed_factor=speed_factor,
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
            success = self._gripper.move(width, speed)
            return {"success": success, "width": width}
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
            success = self._gripper.grasp(width, speed, force)
            return {"success": success, "width": width, "force": force}
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
        x_offset: float = 0.04,
        approach_height: float = 0.15,
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
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        steps = []
        target_x = x + x_offset

        # Standard top-down picking orientation (roll=pi, pitch=0, yaw=0)
        import math
        pick_roll = math.pi
        pick_pitch = 0.0
        pick_yaw = 0.0

        # Step 1: Open gripper
        result = self.gripper_move(0.08)
        steps.append({"action": "open_gripper", "result": result})

        # Step 2: Move above target (IK with straight-down orientation)
        result = self.move_cartesian_ik(
            target_x, y, approach_height,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
        steps.append({"action": "approach_above", "method": "ik", "result": result})

        # Step 3: Lower in increments (avoid large joint changes that trigger reflex)
        state = self.get_state()
        current_z = state.ee_position[2]
        step_z = 0.04  # 4cm increments
        while current_z - step_z > z + step_z:
            intermediate_z = current_z - step_z
            result = self.move_cartesian_ik(
                target_x, y, intermediate_z,
                roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
            steps.append({"action": "lower_step", "method": "ik", "target_z": round(intermediate_z, 4), "result": result})
            state = self.get_state()
            current_z = state.ee_position[2]

        # Final lower to grasp height
        result = self.move_cartesian_ik(
            target_x, y, z,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw, confirmed=True)
        steps.append({"action": "lower_to_grasp", "method": "ik", "result": result})
        state = self.get_state()
        actual_z = state.ee_position[2]
        steps.append({"action": "check_z", "target_z": z, "actual_z": round(actual_z, 4)})

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

    def place_at(
        self,
        x: float,
        y: float,
        z: float = 0.08,
        approach_height: float = 0.15,
    ) -> dict:
        """
        Place a held object at the given robot coordinates.

        Executes: move above target -> lower -> release -> retreat up.

        Args:
            x, y: Target position in robot frame (meters)
            z: Place height (default: 0.08 for gentle placement)
            approach_height: Height to approach/retreat from
        """
        if not self._connected:
            return {"success": False, "error": "Not connected to robot"}

        steps = []

        # Standard top-down orientation for placing
        import math
        place_roll = math.pi
        place_pitch = 0.0
        place_yaw = 0.0

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

    def start_jog(self) -> dict:
        """Start gamepad jog mode. Polls gamepad and executes small moves."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if hasattr(self, '_jog_active') and self._jog_active:
            return {"success": False, "error": "Jog already active"}

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
        self._jog_thread = threading.Thread(target=self._jog_loop, daemon=True)
        self._jog_thread.start()

        return {
            "success": True,
            "message": "Gamepad jog mode enabled",
            "controller": result.get("controller", ""),
            "controls": {
                "left_stick": "X-Y motion",
                "right_stick_y": "Z motion (up/down)",
                "A": "Grasp (30mm, 70N)",
                "B": "Open gripper",
                "X": "Cycle speed (slow/medium/fast)",
                "LB_hold": "Fine mode (1mm steps)",
                "Back": "Stop jog mode",
            },
            "speed": "medium (5mm/step)",
        }

    def _jog_loop(self):
        """Background thread: read gamepad, execute moves."""
        logger.info("Jog loop started")
        move_interval = 0.05  # ~20Hz max move rate

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
                    elif event.action == "open_gripper":
                        try:
                            self.gripper_move(0.08)
                        except Exception as e:
                            logger.warning(f"Gripper open failed: {e}")

                if not self._jog_active:
                    break

                # Execute move if sticks are displaced (IK-based for pure Cartesian motion)
                if state.dx != 0 or state.dy != 0 or state.dz != 0:
                    try:
                        current = self.get_state()
                        target_x = current.ee_position[0] + state.dx
                        target_y = current.ee_position[1] + state.dy
                        target_z = current.ee_position[2] + state.dz
                        self.move_cartesian_ik(
                            target_x, target_y, target_z,
                            confirmed=True,
                        )
                    except Exception as e:
                        logger.warning(f"Jog move failed: {e}")

                time.sleep(move_interval)

            except Exception as e:
                logger.error(f"Jog loop error: {e}")
                time.sleep(0.5)

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

        if self._connected:
            s = self.get_state()
            result["position"] = {
                "x": round(s.ee_position[0], 4),
                "y": round(s.ee_position[1], 4),
                "z": round(s.ee_position[2], 4),
            }
            result["gripper_width"] = round(s.gripper_width, 4)

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


# Singleton controller
_controller: Optional[FrankaController] = None


def get_controller() -> FrankaController:
    global _controller
    if _controller is None:
        _controller = FrankaController()
    return _controller
