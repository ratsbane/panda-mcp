"""Hardware wrapper for Panda + Gripper with timeout protection.

Extracted from franka_mcp/controller.py's _call_with_timeout pattern.
All blocking panda-py/libfranka calls are wrapped in ThreadPoolExecutor
with explicit timeouts and shutdown(wait=False).
"""

import os
import logging
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

# Timeout constants (same as controller.py)
MOTION_TIMEOUT_S = 15.0
GRIPPER_TIMEOUT_S = 10.0
STATE_TIMEOUT_S = 5.0

# Grasp detection thresholds
FORCE_DROP_THRESHOLD = 2.0  # Fz change (N) suggesting object dropped

MOCK_MODE = os.environ.get("FRANKA_MOCK", "0") == "1"

if not MOCK_MODE:
    try:
        import panda_py
        from panda_py import Panda
        from panda_py.libfranka import Gripper
        PANDA_PY_AVAILABLE = True
    except ImportError:
        PANDA_PY_AVAILABLE = False
        logging.warning("panda-py not available, running in mock mode")
        MOCK_MODE = True
else:
    PANDA_PY_AVAILABLE = False


class MockRobot:
    """Mock robot for testing without hardware."""

    def __init__(self):
        self._joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self._ee_pos = np.array([0.4, 0.0, 0.3])
        self._ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def q(self):
        return self._joints.copy()

    def get_state(self):
        class MockState:
            def __init__(self, joints):
                self.q = joints
                self.current_errors = None
                self.O_F_ext_hat_K = [0.0] * 6
                self.tau_ext_hat_filtered = [0.0] * 7
        return MockState(self._joints)

    def get_position(self):
        return self._ee_pos.copy()

    def get_orientation(self):
        return self._ee_quat.copy()

    def move_to_joint_position(self, positions, speed_factor=0.2,
                                dq_threshold=0.01, success_threshold=0.05):
        if isinstance(positions, list) and len(positions) > 0 and isinstance(positions[0], (list, np.ndarray)):
            # Waypoint sequence - go to last
            self._joints = np.array(positions[-1])
        else:
            self._joints = np.array(positions).flatten()
        logger.info(f"Mock joint move")
        return True

    def move_to_pose(self, positions, orientations, speed_factor=0.2):
        if positions:
            self._ee_pos = np.array(positions[-1])
            self._ee_quat = np.array(orientations[-1])
        return True

    def stop_controller(self):
        pass

    def get_robot(self):
        return self

    def stop(self):
        pass

    def recover(self):
        pass

    def teaching_mode(self, active):
        logger.info(f"Mock teaching mode: {active}")


class MockGripper:
    """Mock gripper for testing."""

    def __init__(self):
        self._width = 0.04
        self._is_grasped = False

    def read_once(self):
        class MockGripperState:
            def __init__(self, width, is_grasped):
                self.width = width
                self.is_grasped = is_grasped
        return MockGripperState(self._width, self._is_grasped)

    def move(self, width, speed=0.1):
        self._width = float(np.clip(width, 0.0, 0.08))
        self._is_grasped = False
        return True

    def grasp(self, width, speed, force, epsilon_inner=0.005, epsilon_outer=0.005):
        self._width = width
        self._is_grasped = True
        return True

    def stop(self):
        return True


class RobotProxy:
    """Wraps Panda + Gripper with timeout-protected blocking calls.

    All methods return plain Python dicts/lists (no panda-py objects)
    so results can be serialized over ZMQ.
    """

    def __init__(self):
        self._panda = None
        self._gripper = None
        self._connected = False
        self._mock = MOCK_MODE
        self._stop_event = threading.Event()

    def connect(self, hostname: str) -> dict:
        """Connect to Panda and Gripper at hostname."""
        if self._mock:
            self._panda = MockRobot()
            self._gripper = MockGripper()
            self._connected = True
            return {"connected": True, "mock": True}

        try:
            self._panda = Panda(hostname)
            try:
                self._panda.recover()
            except Exception as e:
                logger.warning(f"Auto-recovery skipped: {e}")
            self._gripper = Gripper(hostname)
            self._connected = True
            logger.info(f"Connected to Franka at {hostname}")
            return {"connected": True, "mock": False}
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return {"connected": False, "error": str(e)}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def panda(self):
        """Raw panda-py Panda object (for servo/RTC access)."""
        return self._panda

    @property
    def raw_gripper(self):
        """Raw panda-py Gripper object (for servo access)."""
        return self._gripper

    def _call_with_timeout(self, fn, *args, timeout=STATE_TIMEOUT_S,
                           label="call", recover=False, **kwargs):
        """Execute fn with timeout. Extracted from controller.py."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(f"{label} timed out after {timeout}s")
            try:
                self._panda.stop_controller()
            except Exception:
                pass
            try:
                self._panda.get_robot().stop()
            except Exception:
                pass
            if recover:
                try:
                    self._panda.recover()
                except Exception:
                    pass
            raise TimeoutError(f"{label} timed out after {timeout}s")
        finally:
            executor.shutdown(wait=False)

    def get_state(self) -> dict:
        """Read full robot state. Returns plain dict."""
        if not self._connected:
            raise RuntimeError("Not connected")

        state = self._call_with_timeout(
            self._panda.get_state, timeout=STATE_TIMEOUT_S, label="get_state")
        position = self._panda.get_position()
        orientation = self._panda.get_orientation()

        # Gripper
        gripper_width = 0.04
        is_grasped = False
        try:
            gs = self._call_with_timeout(
                self._gripper.read_once, timeout=STATE_TIMEOUT_S, label="gripper_read")
            gripper_width = float(gs.width)
            is_grasped = bool(getattr(gs, 'is_grasped', False))
        except Exception as e:
            logger.warning(f"Gripper read failed in get_state: {e}")

        # Force/torque
        ft_ext = None
        tau_ext = None
        try:
            if hasattr(state, 'O_F_ext_hat_K'):
                ft_ext = [float(f) for f in state.O_F_ext_hat_K]
            if hasattr(state, 'tau_ext_hat_filtered'):
                tau_ext = [float(t) for t in state.tau_ext_hat_filtered]
        except Exception:
            pass

        # Error state
        has_error = False
        error_msg = None
        if hasattr(state, 'current_errors') and state.current_errors:
            has_error = True
            error_msg = str(state.current_errors)

        return {
            "q": list(float(v) for v in state.q),
            "ee_position": [float(v) for v in position],
            "ee_orientation": [float(v) for v in orientation],
            "gripper_width": gripper_width,
            "is_grasped": is_grasped,
            "has_error": has_error,
            "error_message": error_msg,
            "O_F_ext_hat_K": ft_ext,
            "tau_ext_hat_filtered": tau_ext,
        }

    def get_q(self) -> list:
        """Get current joint positions (fast, no state read)."""
        if not self._connected:
            raise RuntimeError("Not connected")
        return [float(v) for v in self._panda.q]

    def move_joints(self, q, speed_factor=0.15, dq_threshold=0.01,
                    success_threshold=0.05) -> dict:
        """Move to joint configuration. q can be a single config or list of waypoints."""
        if not self._connected:
            raise RuntimeError("Not connected")

        # Convert to proper format
        if isinstance(q, list) and len(q) > 0 and isinstance(q[0], (list, tuple)):
            # List of waypoints
            positions = [np.array(wp) for wp in q]
        else:
            positions = np.array(q).flatten()

        try:
            self._call_with_timeout(
                self._panda.move_to_joint_position, positions,
                speed_factor=speed_factor,
                dq_threshold=dq_threshold,
                success_threshold=success_threshold,
                timeout=MOTION_TIMEOUT_S, label="move_joints", recover=True,
            )
            return {"success": True}
        except TimeoutError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_to_pose(self, positions, orientations, speed_factor=0.1) -> dict:
        """Move via Cartesian planner (move_to_pose). Legacy — IK is preferred."""
        if not self._connected:
            raise RuntimeError("Not connected")
        try:
            pos_list = [np.array(p) for p in positions]
            ori_list = [np.array(o) for o in orientations]
            self._panda.move_to_pose(pos_list, ori_list, speed_factor=speed_factor)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_joints_monitored(self, q, speed_factor=0.15,
                              force_baseline=None,
                              dq_threshold=0.01,
                              success_threshold=0.05) -> dict:
        """Move to joint config with force monitoring during motion.

        Polls O_F_ext_hat_K every 100ms. Returns force events if Fz
        changes by more than FORCE_DROP_THRESHOLD from baseline.
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        if isinstance(q, list) and len(q) > 0 and isinstance(q[0], (list, tuple)):
            positions = [np.array(wp) for wp in q]
        else:
            positions = np.array(q).flatten()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self._panda.move_to_joint_position, positions,
            speed_factor=speed_factor,
            dq_threshold=dq_threshold,
            success_threshold=success_threshold,
        )
        force_events = []

        try:
            if force_baseline is not None:
                baseline_fz = float(force_baseline[2]) if len(force_baseline) > 2 else 0.0
                start = time.time()
                while time.time() - start < MOTION_TIMEOUT_S:
                    try:
                        future.result(timeout=0.1)
                        return {"success": True, "force_events": force_events}
                    except FuturesTimeoutError:
                        # Check force during motion
                        try:
                            state = self._panda.get_state()
                            if hasattr(state, 'O_F_ext_hat_K'):
                                fz = float(state.O_F_ext_hat_K[2])
                                fz_delta = abs(fz - baseline_fz)
                                if fz_delta > FORCE_DROP_THRESHOLD:
                                    event = {
                                        "time": round(time.time() - start, 3),
                                        "fz_baseline": round(baseline_fz, 2),
                                        "fz_current": round(fz, 2),
                                        "fz_delta": round(fz_delta, 2),
                                    }
                                    force_events.append(event)
                                    logger.warning(f"Force drop: {event}")
                        except Exception:
                            pass
                # Timed out
                raise FuturesTimeoutError()
            else:
                future.result(timeout=MOTION_TIMEOUT_S)
                return {"success": True, "force_events": []}
        except FuturesTimeoutError:
            logger.error("move_joints_monitored timed out")
            try:
                self._panda.stop_controller()
                self._panda.get_robot().stop()
                self._panda.recover()
            except Exception:
                pass
            return {"success": False, "error": "timeout", "force_events": force_events}
        except Exception as e:
            return {"success": False, "error": str(e), "force_events": force_events}
        finally:
            executor.shutdown(wait=False)

    def gripper_move(self, width, speed=0.1) -> dict:
        """Move gripper to width (no force)."""
        if not self._gripper:
            return {"success": False, "error": "No gripper"}
        try:
            result = self._call_with_timeout(
                self._gripper.move, width, speed,
                timeout=GRIPPER_TIMEOUT_S, label="gripper_move")
            return {"success": True, "width": width}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gripper_grasp(self, width, speed=0.1, force=20.0,
                      epsilon_inner=0.005, epsilon_outer=0.005) -> dict:
        """Grasp at width with force."""
        if not self._gripper:
            return {"success": False, "error": "No gripper"}
        try:
            result = self._call_with_timeout(
                self._gripper.grasp, width, speed, force,
                epsilon_inner, epsilon_outer,
                timeout=GRIPPER_TIMEOUT_S, label="gripper_grasp")
            return {"success": True, "width": width, "force": force}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gripper_read(self) -> dict:
        """Read gripper state."""
        if not self._gripper:
            return {"success": False, "error": "No gripper"}
        try:
            gs = self._call_with_timeout(
                self._gripper.read_once, timeout=STATE_TIMEOUT_S, label="gripper_read")
            return {
                "width": float(gs.width),
                "is_grasped": bool(getattr(gs, 'is_grasped', False)),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def gripper_stop(self) -> dict:
        """Stop gripper motion."""
        if not self._gripper:
            return {"success": False, "error": "No gripper"}
        try:
            self._gripper.stop()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop(self) -> dict:
        """Emergency stop: stop controller, stop robot, recover."""
        errors = []
        try:
            self._panda.stop_controller()
        except Exception as e:
            errors.append(f"stop_controller: {e}")
        try:
            self._panda.get_robot().stop()
        except Exception as e:
            errors.append(f"libfranka stop: {e}")
        try:
            if self._gripper:
                self._gripper.stop()
        except Exception as e:
            errors.append(f"gripper stop: {e}")
        try:
            self._panda.recover()
        except Exception as e:
            errors.append(f"recover: {e}")

        return {"success": True, "warnings": errors} if errors else {"success": True}

    def recover(self) -> dict:
        """Recover from error state."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}
        try:
            self._panda.recover()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def teaching_mode(self, active: bool) -> dict:
        """Enable/disable teaching mode (gravity compensation)."""
        if not self._connected:
            return {"success": False, "error": "Not connected"}
        if self._mock:
            return {"success": True, "teaching_mode": active, "mock": True}
        try:
            self._panda.teaching_mode(active)
            return {"success": True, "teaching_mode": active}
        except Exception as e:
            return {"success": False, "error": str(e)}
