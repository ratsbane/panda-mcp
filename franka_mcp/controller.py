"""
Franka Panda arm controller wrapper.

Provides a safe, high-level interface to panda-py/libfranka.
"""

import os
import logging
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
        self._ee_pos = np.array(positions[0])
        self._ee_quat = np.array(orientations[0])
        logger.info(f"Mock pose move to {positions[0]}")
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


# Singleton controller
_controller: Optional[FrankaController] = None


def get_controller() -> FrankaController:
    global _controller
    if _controller is None:
        _controller = FrankaController()
    return _controller
