"""franka_rt/servo.py - Smooth servo descent for pick operations.

Two modes:
1. Waypoint mode (default): Pre-compute IK waypoints at 5mm Z intervals,
   execute as single move_to_joint_position call. Reliable, no controller
   transition issues. Uses panda-py's trajectory planner for smooth,
   velocity/power-safe motion.

2. NUDGE mode (future): JointPosition impedance controller with real-time
   camera corrections at ~30Hz. Requires trained NUDGE model.

Key design decisions:
- IK seeding uses chain from previous waypoint (not actual panda.q)
- 5mm Z steps give smooth trajectory with fine-grained IK control
- Waypoints include XY blend from approach position to target position
- Grasp detection uses gripper width threshold
- Two-phase descent: fast to SLOW_START_HEIGHT, then slow for final approach
- Collision thresholds raised during descent, restored in finally block
- Recovery from trajectory errors with retry from actual position
"""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    import panda_py
    from panda_py.controllers import JointPosition
    PANDA_AVAILABLE = True
except ImportError:
    PANDA_AVAILABLE = False
    logger.warning("panda-py not available for servo")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Camera daemon endpoint
CAMERA_IPC = "ipc:///tmp/camera-daemon.sock"

# IK parameters
IK_POSITION_VERIFY_M = 0.005   # FK verification threshold (5mm)
JOINT_LIMIT_MARGIN = 0.10      # Stay this far from joint limits (radians)

# Panda joint limits (same as controller.py)
_JOINT_LIMITS = np.array([
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])

# Waypoint descent parameters
WAYPOINT_Z_STEP = 0.005        # 5mm between waypoints
DESCENT_SPEED_FACTOR = 0.15    # Speed for move_to_joint_position

# Grasp detection
GRIP_CLOSED_EMPTY_M = 0.0015   # Width below this = empty grasp

# NUDGE correction limits (for future NUDGE mode)
MAX_CORRECTION_PER_STEP = 0.005  # 5mm max XY correction per loop iteration
CORRECTION_LOWPASS = 0.4         # Low-pass filter for corrections

# Two-phase descent: slow down for last 30mm
SLOW_START_HEIGHT = 0.030       # Start slowing 30mm above grasp
SLOW_SPEED_FACTOR = 0.12        # Speed for final approach phase

# Collision recovery
MAX_RECOVERIES = 3              # Give up after this many reflexes per descent

# Relaxed collision thresholds during descent
# Lower = contact detection, Upper = reflex trigger
# Defaults are 20/20. Raise both to tolerate light contact.
DESCENT_CONTACT_TORQUE = [30.0] * 7     # Joint torque contact threshold (Nm)
DESCENT_REFLEX_TORQUE = [40.0] * 7      # Joint torque reflex threshold (Nm)
DESCENT_CONTACT_FORCE = [30.0] * 6      # Cartesian force contact threshold (N)
DESCENT_REFLEX_FORCE = [40.0] * 6       # Cartesian force reflex threshold (N)

# Default (tight) collision thresholds to restore after descent
DEFAULT_COLLISION_TORQUE = [20.0] * 7
DEFAULT_COLLISION_FORCE = [20.0] * 6

# Settle time after reaching grasp height (let vibrations damp)
SETTLE_TIME_S = 0.15


class NudgeRTServo:
    """Smooth servo descent with optional NUDGE visual corrections.

    Default mode: pre-compute IK waypoints, execute as single trajectory.
    NUDGE mode: JointPosition controller with real-time camera corrections.
    """

    def __init__(self, panda, gripper):
        self._panda = panda
        self._gripper = gripper
        self._camera_sock = None
        self._nudge = None

    def _orientation_quat(self, yaw=0.0):
        """Get quaternion [x,y,z,w] for downward-pointing gripper with given yaw."""
        cr = np.cos(np.pi / 2)
        sr = np.sin(np.pi / 2)
        cp, sp = 1.0, 0.0
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([x, y, z, w])

    def _solve_ik(self, x, y, z, q_seed, yaw=0.0):
        """Solve IK for target position. Returns q (7,) or None."""
        position = np.array([x, y, z])
        orientation = self._orientation_quat(yaw)
        margin = JOINT_LIMIT_MARGIN

        q7_candidates = [
            q_seed[6],
            q_seed[6] + 0.2, q_seed[6] - 0.2,
            q_seed[6] + 0.5, q_seed[6] - 0.5,
            0.785, 0.0, -0.785, 1.57,
        ]

        for q7 in q7_candidates:
            q7 = max(_JOINT_LIMITS[6, 0] + margin, min(_JOINT_LIMITS[6, 1] - margin, q7))
            try:
                sol = panda_py.ik(position, orientation, q_seed, q7)
            except Exception:
                continue
            if sol is None:
                continue

            q_arr = np.array(sol).flatten()
            if q_arr.shape != (7,):
                continue

            in_limits = all(
                _JOINT_LIMITS[i, 0] + margin <= q_arr[i] <= _JOINT_LIMITS[i, 1] - margin
                for i in range(7)
            )
            if not in_limits:
                continue

            try:
                fk_pose = panda_py.fk(q_arr)
                if np.linalg.norm(fk_pose[:3, 3] - position) < IK_POSITION_VERIFY_M:
                    return q_arr
            except Exception:
                continue

        return None

    def _connect_camera(self):
        """Initialize ZMQ context for camera access."""
        import zmq
        self._zmq_ctx = zmq.Context.instance()
        logger.info("Servo camera ready (fresh-socket-per-frame)")

    def _get_frame(self):
        """Get latest camera frame via fresh socket."""
        import zmq
        if not hasattr(self, '_zmq_ctx') or self._zmq_ctx is None:
            return None
        sock = self._zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.setsockopt_string(zmq.SUBSCRIBE, "frame")
        try:
            sock.connect(CAMERA_IPC)
            parts = sock.recv_multipart()
            if len(parts) != 3:
                return None
            frame = np.frombuffer(parts[2], dtype=np.uint8)
            return cv2.imdecode(frame, cv2.IMREAD_COLOR)
        except Exception:
            return None
        finally:
            sock.close()

    def _close_camera(self):
        """No persistent socket to close — fresh-socket approach."""
        self._zmq_ctx = None

    def _load_nudge(self, model_path):
        """Load NUDGE ONNX model for inference."""
        try:
            from common.nudge.servo import NUDGEServoLoop
            self._nudge = NUDGEServoLoop(model_path)
            logger.info(f"NUDGE model loaded: {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load NUDGE model: {e}")
            self._nudge = None
            return False

    def _raise_collision_thresholds(self):
        """Raise collision thresholds for softer contact during descent.

        Uses 8-argument form: lower/upper for both acceleration and nominal phases.
        Lower = contact detection, Upper = reflex trigger.
        """
        try:
            self._panda.get_robot().set_collision_behavior(
                DESCENT_CONTACT_TORQUE, DESCENT_REFLEX_TORQUE,
                DESCENT_CONTACT_TORQUE, DESCENT_REFLEX_TORQUE,
                DESCENT_CONTACT_FORCE, DESCENT_REFLEX_FORCE,
                DESCENT_CONTACT_FORCE, DESCENT_REFLEX_FORCE,
            )
            logger.debug("Collision thresholds raised for descent")
        except Exception as e:
            logger.warning(f"Could not raise collision thresholds: {e}")

    def _restore_collision_thresholds(self):
        """Restore default collision thresholds."""
        try:
            self._panda.get_robot().set_collision_behavior(
                DEFAULT_COLLISION_TORQUE, DEFAULT_COLLISION_TORQUE,
                DEFAULT_COLLISION_TORQUE, DEFAULT_COLLISION_TORQUE,
                DEFAULT_COLLISION_FORCE, DEFAULT_COLLISION_FORCE,
                DEFAULT_COLLISION_FORCE, DEFAULT_COLLISION_FORCE,
            )
            logger.debug("Collision thresholds restored")
        except Exception:
            pass

    def _compute_waypoints(self, start_x, start_y, start_z,
                           target_x, target_y, target_z,
                           seed_q, yaw, result):
        """Compute IK waypoints for descent trajectory.

        Blends XY linearly from start to target over the full Z descent.
        Returns list of joint configurations.
        """
        z_levels = []
        current_z = start_z
        while current_z - WAYPOINT_Z_STEP > target_z:
            current_z -= WAYPOINT_Z_STEP
            z_levels.append(current_z)
        z_levels.append(target_z)

        total_descent = start_z - target_z
        waypoints = []
        q_seed = seed_q.copy()

        for z_level in z_levels:
            if total_descent > 0:
                t = (start_z - z_level) / total_descent
            else:
                t = 1.0
            wp_x = start_x + t * (target_x - start_x)
            wp_y = start_y + t * (target_y - start_y)

            q_sol = self._solve_ik(wp_x, wp_y, z_level, q_seed, yaw)
            if q_sol is not None:
                waypoints.append(q_sol)
                q_seed = q_sol
            else:
                result["ik_failures"] += 1
                logger.warning(
                    f"IK failed at z={z_level:.4f} ({wp_x:.3f},{wp_y:.3f})")

        return waypoints

    def _split_waypoints_at_height(self, waypoints, start_z, target_z, split_height):
        """Split waypoints into fast (above split_height) and slow (below).

        Returns (fast_waypoints, slow_waypoints).
        """
        if start_z <= split_height or not waypoints:
            return [], waypoints

        total_descent = start_z - target_z
        if total_descent <= 0:
            return [], waypoints

        # Find the split index: which waypoint corresponds to split_height?
        split_z = target_z + split_height
        n_above = 0
        for i, wp in enumerate(waypoints):
            fk_pose = panda_py.fk(wp)
            wp_z = float(fk_pose[2, 3])
            if wp_z > split_z:
                n_above = i + 1
            else:
                break

        if n_above == 0:
            return [], waypoints
        if n_above >= len(waypoints):
            return waypoints, []

        return waypoints[:n_above], waypoints[n_above:]

    def execute(
        self,
        x: float, y: float, z: float = 0.013,
        yaw: float = 0.0,
        grasp_width: float = 0.03,
        grasp_force: float = 70.0,
        descent_rate: float = 0.040,
        model_path: str = None,
        target_bbox: list = None,
        nudge_gain: float = 0.7,
        speed_factor: float = 0.15,
        **kwargs,
    ) -> dict:
        """Execute smooth descent + grasp.

        Without NUDGE model: pre-compute waypoints, single trajectory.
        With NUDGE model: (future) impedance controller with camera corrections.

        Two-phase descent:
        1. Fast phase: from current height down to SLOW_START_HEIGHT above grasp.
           Uses speed_factor for rapid approach.
        2. Slow phase: last SLOW_START_HEIGHT. Uses SLOW_SPEED_FACTOR for
           gentle contact and collision tolerance.

        Collision thresholds are raised for the entire descent and restored
        in the finally block.

        Args:
            x, y: Target position in robot frame (meters)
            z: Grasp height (meters, default 0.013 = table)
            yaw: Gripper rotation (radians)
            grasp_width: Expected object width for grasp (meters)
            grasp_force: Grasp force in Newtons
            speed_factor: Motion speed for fast phase (0.0-1.0, default 0.15)

        Returns:
            dict with results
        """
        result = {
            "success": False,
            "phases": [],
            "corrections": [],
            "ik_failures": 0,
            "n_corrections": 0,
            "recoveries": 0,
        }
        start_time = time.time()

        try:
            # ---- Setup ----
            current_q = np.array(self._panda.q)
            fk_current = panda_py.fk(current_q)
            start_x = float(fk_current[0, 3])
            start_y = float(fk_current[1, 3])
            start_z = float(fk_current[2, 3])
            result["start_z"] = round(start_z, 4)

            target_x, target_y = float(x), float(y)

            logger.info(
                f"Servo descent: ({start_x:.3f},{start_y:.3f},{start_z:.3f}) -> "
                f"({target_x:.3f},{target_y:.3f},{z:.3f}), "
                f"step={WAYPOINT_Z_STEP*1000:.0f}mm, "
                f"slow_start={SLOW_START_HEIGHT*1000:.0f}mm"
            )

            # ---- Raise collision thresholds ----
            self._raise_collision_thresholds()

            # ---- Compute waypoints ----
            waypoints = self._compute_waypoints(
                start_x, start_y, start_z,
                target_x, target_y, z,
                current_q, yaw, result,
            )

            result["n_waypoints"] = len(waypoints)
            logger.info(
                f"Generated {len(waypoints)} waypoints "
                f"({result['ik_failures']} IK failures)"
            )

            if len(waypoints) < 3:
                result["error"] = f"Too few waypoints ({len(waypoints)})"
                result["phases"].append("waypoint_error")
                return result

            result["phases"].append("waypoints_computed")

            # ---- Split into fast + slow phases ----
            fast_wps, slow_wps = self._split_waypoints_at_height(
                waypoints, start_z, z, SLOW_START_HEIGHT)

            result["n_fast_waypoints"] = len(fast_wps)
            result["n_slow_waypoints"] = len(slow_wps)
            logger.info(
                f"Descent phases: {len(fast_wps)} fast + {len(slow_wps)} slow waypoints"
            )

            # ---- Execute fast descent ----
            if fast_wps:
                descent_start = time.time()
                try:
                    self._panda.move_to_joint_position(
                        fast_wps,
                        speed_factor=speed_factor,
                        dq_threshold=0.01,
                        success_threshold=0.05,
                    )
                    result["phases"].append("fast_descent_complete")
                except Exception as e:
                    logger.warning(f"Fast descent error: {e}")
                    result["phases"].append("fast_descent_error")
                    result["recoveries"] += 1

                    if not self._try_recovery(result):
                        return result

                    # After recovery, recompute remaining waypoints from actual pos
                    actual_q = np.array(self._panda.q)
                    fk_actual = panda_py.fk(actual_q)
                    actual_z = float(fk_actual[2, 3])
                    logger.info(f"Recovered at z={actual_z:.4f}, recomputing slow phase")

                    # Recompute slow waypoints from current position
                    slow_wps = self._compute_waypoints(
                        float(fk_actual[0, 3]), float(fk_actual[1, 3]), actual_z,
                        target_x, target_y, z,
                        actual_q, yaw, result,
                    )

                result["fast_descent_time_s"] = round(time.time() - descent_start, 2)

            # ---- Execute slow descent (final approach) ----
            if slow_wps:
                slow_start = time.time()
                try:
                    self._panda.move_to_joint_position(
                        slow_wps,
                        speed_factor=SLOW_SPEED_FACTOR,
                        dq_threshold=0.01,
                        success_threshold=0.05,
                    )
                    result["phases"].append("slow_descent_complete")
                except Exception as e:
                    logger.warning(f"Slow descent error: {e}")
                    result["phases"].append("slow_descent_error")
                    result["recoveries"] += 1

                    if not self._try_recovery(result):
                        return result

                    # After recovery, try to finish descent from actual position
                    actual_q = np.array(self._panda.q)
                    fk_actual = panda_py.fk(actual_q)
                    actual_z = float(fk_actual[2, 3])
                    remaining = actual_z - z

                    if remaining > 0.003:  # More than 3mm above target
                        logger.info(
                            f"Re-descending from z={actual_z:.4f} "
                            f"({remaining*1000:.0f}mm remaining)")
                        retry_wps = self._compute_waypoints(
                            float(fk_actual[0, 3]), float(fk_actual[1, 3]),
                            actual_z, target_x, target_y, z,
                            actual_q, yaw, result,
                        )
                        if retry_wps:
                            try:
                                self._panda.move_to_joint_position(
                                    retry_wps,
                                    speed_factor=SLOW_SPEED_FACTOR,
                                    dq_threshold=0.01,
                                    success_threshold=0.05,
                                )
                                result["phases"].append("slow_retry_complete")
                            except Exception as e2:
                                logger.warning(f"Slow retry also failed: {e2}")
                                result["phases"].append("slow_retry_error")
                                # Don't try again, just proceed to grasp

                result["slow_descent_time_s"] = round(time.time() - slow_start, 2)

            descent_time = time.time() - start_time
            result["descent_time_s"] = round(descent_time, 2)

            # ---- Verify final position ----
            actual_q = np.array(self._panda.q)
            fk_final = panda_py.fk(actual_q)
            actual_x = float(fk_final[0, 3])
            actual_y = float(fk_final[1, 3])
            actual_z = float(fk_final[2, 3])
            pos_error = np.linalg.norm([actual_x - target_x, actual_y - target_y, actual_z - z])
            logger.info(
                f"Descent done: ({actual_x:.3f},{actual_y:.3f},{actual_z:.4f}) "
                f"target ({target_x:.3f},{target_y:.3f},{z:.4f}), "
                f"error={pos_error*1000:.1f}mm"
            )
            result["position_error_mm"] = round(pos_error * 1000, 1)

            # ---- Settle ----
            if SETTLE_TIME_S > 0:
                time.sleep(SETTLE_TIME_S)
                result["phases"].append("settled")

            # ---- Grasp ----
            try:
                self._gripper.grasp(
                    grasp_width, 0.1, grasp_force,
                    epsilon_inner=0.005, epsilon_outer=0.005,
                )
            except Exception as e:
                logger.warning(f"Grasp call returned: {e}")

            # Read gripper state
            try:
                gs = self._gripper.read_once()
                grip_width = float(gs.width)
                result["gripper_width"] = round(grip_width, 4)
                result["is_grasped"] = bool(getattr(gs, 'is_grasped', False))

                if grip_width < GRIP_CLOSED_EMPTY_M:
                    result["error"] = "Gripper closed empty - no object grasped"
                    result["phases"].append("grasp_empty")
                else:
                    result["success"] = True
                    result["phases"].append("grasped")
            except Exception as e:
                logger.warning(f"Gripper read failed: {e}")
                result["success"] = True
                result["phases"].append("grasp_unverified")

            # Record final position
            try:
                fk_final = panda_py.fk(np.array(self._panda.q))
                result["final_position"] = {
                    "x": round(float(fk_final[0, 3]), 4),
                    "y": round(float(fk_final[1, 3]), 4),
                    "z": round(float(fk_final[2, 3]), 4),
                }
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Servo pick error: {e}", exc_info=True)
            result["error"] = str(e)

        finally:
            # Restore default collision thresholds
            self._restore_collision_thresholds()

            # Cleanup
            self._close_camera()
            if self._nudge is not None:
                try:
                    self._nudge.stop()
                except Exception:
                    pass

        result["total_time_s"] = round(time.time() - start_time, 2)
        return result

    def _try_recovery(self, result: dict) -> bool:
        """Attempt to recover from a trajectory error.

        Returns True if recovered and descent can continue,
        False if recovery limit exceeded.
        """
        if result["recoveries"] > MAX_RECOVERIES:
            result["error"] = f"Too many collisions ({result['recoveries']})"
            result["phases"].append("recovery_limit")
            return False

        try:
            self._panda.recover()
            result["phases"].append(f"recovery_{result['recoveries']}")
            logger.info(f"Recovery {result['recoveries']}/{MAX_RECOVERIES} successful")

            # Re-raise collision thresholds (recovery may reset them)
            self._raise_collision_thresholds()
            return True
        except Exception as e:
            result["error"] = f"Recovery failed: {e}"
            result["phases"].append("recovery_failed")
            logger.error(f"Recovery failed: {e}")
            return False
