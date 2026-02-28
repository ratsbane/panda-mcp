"""franka_rt/servo.py - Real-time servo descent with NUDGE visual corrections.

Uses panda-py's JointPosition controller for continuous 1kHz impedance-controlled
descent. Camera frames + NUDGE inference at ~30Hz provide XY corrections.

Key design decisions:
- IK seeding uses COMMANDED target chain (not actual panda.q) to avoid cascade failures
- Descent rate of 40mm/s balances speed with controller tracking
- filter_coeff=0.3 gives responsive but smooth joint target updates
- Falls back to open-loop smooth descent if no NUDGE model provided
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

# JointPosition controller parameters
DEFAULT_STIFFNESS = np.array([600., 600., 600., 600., 250., 150., 50.])
DEFAULT_DAMPING = np.array([50., 50., 50., 20., 20., 20., 10.])

# Servo parameters
DEFAULT_DESCENT_RATE = 0.040   # 40 mm/s — fast but trackable
FILTER_COEFF = 0.3             # JointPosition LP filter (0=heavy smoothing, 1=passthrough)
LOOP_HZ = 30                   # Target servo loop rate
SETTLE_TIME_S = 0.2            # Hold at grasp height before grasping

# Grasp detection
GRIP_CLOSED_EMPTY_M = 0.0015   # Width below this = empty grasp

# NUDGE correction limits
MAX_CORRECTION_PER_STEP = 0.005  # 5mm max XY correction per loop iteration
CORRECTION_LOWPASS = 0.4         # Low-pass filter for corrections (0=ignore, 1=instant)


class NudgeRTServo:
    """Real-time servo using JointPosition controller + NUDGE visual corrections.

    The servo operates in phases:
    1. Start JointPosition controller (1kHz impedance loop)
    2. Descent loop at ~30Hz: lower Z, get camera frame, NUDGE → XY corrections
    3. Settle at grasp height (0.2s)
    4. Stop controller, grasp, verify

    IK targets are chain-seeded from the COMMANDED position, not the actual
    robot state (which lags behind). This prevents IK cascade failures.
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
        """Get latest camera frame via fresh socket (avoids CONFLATE multipart crash).

        Creates a new SUB socket per call, receives one multipart message, closes.
        ~5ms overhead per call, safe with multipart messages.
        """
        import zmq
        if not hasattr(self, '_zmq_ctx') or self._zmq_ctx is None:
            return None
        sock = self._zmq_ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
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

    def execute(
        self,
        x: float, y: float, z: float = 0.013,
        yaw: float = 0.0,
        grasp_width: float = 0.03,
        grasp_force: float = 70.0,
        descent_rate: float = DEFAULT_DESCENT_RATE,
        model_path: str = None,
        target_bbox: list = None,
        nudge_gain: float = 0.7,
        speed_factor: float = 0.5,
        **kwargs,
    ) -> dict:
        """Execute smooth descent with optional NUDGE corrections + grasp.

        Uses JointPosition controller (1kHz impedance) with ~30Hz target updates.
        Camera frames drive NUDGE XY corrections during descent.

        Args:
            x, y: Target position in robot frame (meters)
            z: Grasp height (meters, default 0.013 = table)
            yaw: Gripper rotation (radians)
            grasp_width: Expected object width for grasp (meters)
            grasp_force: Grasp force in Newtons
            descent_rate: Z descent speed (m/s, default 0.040)
            model_path: Path to NUDGE ONNX model (None = open-loop)
            target_bbox: [x1, y1, x2, y2] in pixels for NUDGE
            nudge_gain: Correction gain (0-1)
            speed_factor: (unused, descent_rate controls speed)

        Returns:
            dict with results
        """
        result = {
            "success": False,
            "phases": [],
            "corrections": [],
            "ik_failures": 0,
            "n_corrections": 0,
        }
        start_time = time.time()
        controller_active = False

        try:
            # ---- Setup ----
            current_q = np.array(self._panda.q)
            fk_current = panda_py.fk(current_q)
            current_z = float(fk_current[2, 3])
            result["start_z"] = round(current_z, 4)

            target_x, target_y = float(x), float(y)
            target_z = current_z  # Will ramp down

            logger.info(
                f"Servo descent: ({target_x:.3f}, {target_y:.3f}) "
                f"z={current_z:.3f} -> {z:.3f}, rate={descent_rate*1000:.0f}mm/s"
            )

            # Load NUDGE if model provided
            use_nudge = False
            if model_path and target_bbox:
                if self._load_nudge(model_path):
                    self._nudge.start(target_bbox_pixels=target_bbox)
                    self._connect_camera()
                    use_nudge = True
                    result["phases"].append("nudge_enabled")
                    logger.info(f"NUDGE corrections active, bbox={target_bbox}")

            # Accumulated correction (low-pass filtered)
            corr_x, corr_y = 0.0, 0.0

            # ---- Start JointPosition controller ----
            ctrl = JointPosition(
                stiffness=DEFAULT_STIFFNESS,
                damping=DEFAULT_DAMPING,
                filter_coeff=FILTER_COEFF,
            )
            self._panda.start_controller(ctrl)
            controller_active = True
            result["phases"].append("controller_started")

            # Set initial target to current position
            seed_q = current_q.copy()
            ctrl.set_control(current_q)

            # ---- Descent loop ----
            loop_period = 1.0 / LOOP_HZ
            descent_start = time.time()
            loop_count = 0

            while target_z > z:
                loop_start = time.time()
                loop_count += 1

                # Ramp Z down
                target_z -= descent_rate * loop_period
                target_z = max(target_z, z)

                # Get NUDGE correction from camera
                if use_nudge:
                    frame = self._get_frame()
                    if frame is not None:
                        correction = self._nudge.predict(frame)
                        if correction is not None:
                            dx, dy, _dz = correction
                            # Low-pass filter + clamp corrections
                            dx = max(-MAX_CORRECTION_PER_STEP, min(MAX_CORRECTION_PER_STEP, dx))
                            dy = max(-MAX_CORRECTION_PER_STEP, min(MAX_CORRECTION_PER_STEP, dy))
                            corr_x = corr_x * (1 - CORRECTION_LOWPASS) + dx * CORRECTION_LOWPASS
                            corr_y = corr_y * (1 - CORRECTION_LOWPASS) + dy * CORRECTION_LOWPASS
                            result["n_corrections"] += 1
                            if loop_count <= 3 or loop_count % 10 == 0:
                                result["corrections"].append({
                                    "step": loop_count,
                                    "dx_mm": round(corr_x * 1000, 1),
                                    "dy_mm": round(corr_y * 1000, 1),
                                })

                # Solve IK for corrected target
                ik_x = target_x + corr_x
                ik_y = target_y + corr_y
                q_sol = self._solve_ik(ik_x, ik_y, target_z, seed_q, yaw)

                if q_sol is not None:
                    ctrl.set_control(q_sol)
                    seed_q = q_sol  # Chain from COMMANDED, not actual
                else:
                    result["ik_failures"] += 1

                # Rate limit
                elapsed = time.time() - loop_start
                if elapsed < loop_period:
                    time.sleep(loop_period - elapsed)

            descent_time = time.time() - descent_start
            result["descent_time_s"] = round(descent_time, 2)
            result["loop_count"] = loop_count
            result["phases"].append("descent_complete")

            # ---- Settle at grasp height ----
            settle_start = time.time()
            while time.time() - settle_start < SETTLE_TIME_S:
                # Keep commanding the final target
                q_sol = self._solve_ik(target_x + corr_x, target_y + corr_y, z, seed_q, yaw)
                if q_sol is not None:
                    ctrl.set_control(q_sol)
                    seed_q = q_sol
                time.sleep(loop_period)
            result["phases"].append("settled")

            # ---- Stop controller ----
            self._panda.stop_controller()
            controller_active = False
            time.sleep(0.05)

            # Verify final position
            actual_q = np.array(self._panda.q)
            fk_final = panda_py.fk(actual_q)
            actual_z = float(fk_final[2, 3])
            logger.info(
                f"Descent done: z={actual_z:.4f} (target {z:.4f}), "
                f"{loop_count} loops, {result['ik_failures']} IK fails, "
                f"{result['n_corrections']} corrections"
            )

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
            # Safety: stop controller if still active
            if controller_active:
                try:
                    self._panda.stop_controller()
                except Exception:
                    pass

            # Cleanup
            self._close_camera()
            if self._nudge is not None:
                try:
                    self._nudge.stop()
                except Exception:
                    pass

        result["total_time_s"] = round(time.time() - start_time, 2)
        return result
