"""
VLA inference client for Franka Panda.

Runs a background control loop that:
1. Captures camera frame + robot state
2. Requests a CHUNK of actions from the inference server (1 HTTP call = 50 actions)
3. Applies predicted joint deltas one at a time at high speed
4. Repeats when the chunk is exhausted

This minimizes network round-trips (1 per 50 steps) and uses speed_factor=1.0
for fast execution of small joint moves.
"""

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Safety limits for joint deltas per step
MAX_JOINT_DELTA_RAD = 0.05  # ~2.9 degrees per step
MAX_GRIPPER_DELTA_M = 0.01  # 10mm per step
GRIPPER_THRESHOLD = 0.002  # 2mm change triggers gripper action
CONSECUTIVE_ERROR_LIMIT = 5
REQUEST_TIMEOUT_S = 5.0
CHUNK_SIZE = 50  # actions per server request (matches SmolVLA chunk_size)
ACTIONS_PER_MOVE = 10  # accumulate N action deltas into one robot move

# Panda joint limits (radians)
JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass
class VLAStatus:
    """Current VLA control loop status."""
    active: bool = False
    task: str = ""
    step_count: int = 0
    server_url: str = ""
    last_inference_ms: float = 0.0
    last_action: list = field(default_factory=list)
    consecutive_errors: int = 0
    last_error: str = ""
    hz: float = 0.0
    chunk_remaining: int = 0


class VLAClient:
    """Background VLA control loop for Franka Panda."""

    def __init__(self, controller, server_url: str, task: str):
        self._controller = controller
        self._server_url = server_url.rstrip("/")
        self._task = task

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._status = VLAStatus(
            task=task,
            server_url=server_url,
        )

        # Camera client (lazy init)
        self._camera = None

    def _init_camera(self):
        """Initialize ZeroMQ camera client."""
        if self._camera is not None:
            return True

        try:
            from camera_daemon.client import CameraClient
            self._camera = CameraClient()
            if self._camera.connect():
                logger.info("VLA: Camera connected")
                return True
            else:
                logger.error("VLA: Camera daemon not available")
                return False
        except ImportError:
            logger.error("VLA: camera_daemon package not found")
            return False

    def start(self) -> dict:
        """Start the VLA control loop."""
        # Initialize camera
        if not self._init_camera():
            return {"success": False, "error": "Camera not available"}

        # Reset the server's action queue
        try:
            resp = requests.post(
                f"{self._server_url}/reset",
                json={"task": self._task},
                timeout=REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
        except Exception as e:
            return {"success": False, "error": f"Server reset failed: {e}"}

        # Start control loop thread
        self._stop_event.clear()
        with self._lock:
            self._status.active = True
            self._status.step_count = 0
            self._status.consecutive_errors = 0
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

        return {"success": True, "message": f"VLA control active: {self._task}"}

    def stop(self) -> dict:
        """Stop the VLA control loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

        with self._lock:
            self._status.active = False
            status = self._get_status_snapshot()

        return {
            "success": True,
            "message": "VLA control stopped",
            "steps_completed": status.step_count,
            "task": status.task,
        }

    def _fetch_action_chunk(self) -> list[list[float]]:
        """Capture observation and fetch a chunk of actions from the server."""
        # Get current robot state
        state = self._controller.get_state()
        joint_positions = list(state.joint_positions)
        gripper_width = float(state.gripper_width)
        obs_state = joint_positions + [gripper_width]

        # Capture camera frame as JPEG
        jpeg_bytes = self._camera.get_frame_jpeg()
        if jpeg_bytes is None:
            raise RuntimeError("No camera frame")
        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        # Request full chunk from server
        resp = requests.post(
            f"{self._server_url}/predict",
            json={
                "state": obs_state,
                "image_b64": image_b64,
                "task": self._task,
                "n_actions": CHUNK_SIZE,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        result = resp.json()
        return result["actions"]

    def _apply_action(self, action: list[float]):
        """Apply a single action (7 joint deltas + gripper delta) to the robot."""
        state = self._controller.get_state()
        current_joints = np.array(list(state.joint_positions))
        gripper_width = float(state.gripper_width)

        # Safety clip
        joint_deltas = np.clip(action[:7], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
        gripper_delta = float(np.clip(action[7], -MAX_GRIPPER_DELTA_M, MAX_GRIPPER_DELTA_M))

        # Compute and clamp target joints
        target_joints = np.clip(
            current_joints + joint_deltas,
            JOINT_LIMITS_LOWER,
            JOINT_LIMITS_UPPER,
        )

        # Execute joint move at max speed (moves are tiny, so this is safe)
        self._controller._robot.move_to_joint_position(
            target_joints.tolist(), speed_factor=1.0
        )

        # Handle gripper
        if gripper_delta > GRIPPER_THRESHOLD:
            try:
                self._controller.gripper_move(gripper_width + gripper_delta)
            except Exception as e:
                logger.debug(f"VLA gripper open: {e}")
        elif gripper_delta < -GRIPPER_THRESHOLD:
            try:
                self._controller.gripper_grasp(
                    width=max(0.0, gripper_width + gripper_delta),
                    force=70,
                    speed=0.1,
                )
            except Exception as e:
                logger.debug(f"VLA gripper close: {e}")

    def _control_loop(self):
        """Background thread: fetch chunk → accumulate N deltas → move → repeat.

        Accumulates ACTIONS_PER_MOVE action deltas into one larger robot move.
        This reduces the number of blocking move_to_joint_position calls by 10x,
        producing visibly faster and smoother motion.
        """
        logger.info(f"VLA control loop started: {self._task}")
        move_count = 0
        action_count = 0
        loop_start_time = time.monotonic()
        action_queue: deque[list[float]] = deque()

        while not self._stop_event.is_set():
            try:
                # Refill action queue when empty
                if not action_queue:
                    t0 = time.monotonic()
                    chunk = self._fetch_action_chunk()
                    infer_ms = (time.monotonic() - t0) * 1000
                    action_queue.extend(chunk)
                    logger.info(
                        f"VLA: fetched {len(chunk)} actions in {infer_ms:.0f}ms"
                    )
                    with self._lock:
                        self._status.last_inference_ms = infer_ms

                # Accumulate N action deltas into one move
                n_to_pop = min(ACTIONS_PER_MOVE, len(action_queue))
                accumulated_joints = np.zeros(7)
                accumulated_gripper = 0.0
                last_action = None

                for _ in range(n_to_pop):
                    action = action_queue.popleft()
                    last_action = action
                    accumulated_joints += np.array(action[:7])
                    accumulated_gripper += action[7]

                # Apply the accumulated move
                combined_action = accumulated_joints.tolist() + [accumulated_gripper]
                self._apply_action(combined_action)

                move_count += 1
                action_count += n_to_pop
                elapsed_total = time.monotonic() - loop_start_time
                with self._lock:
                    self._status.step_count = action_count
                    self._status.last_action = combined_action
                    self._status.consecutive_errors = 0
                    self._status.last_error = ""
                    self._status.hz = action_count / elapsed_total if elapsed_total > 0 else 0
                    self._status.chunk_remaining = len(action_queue)

            except Exception as e:
                with self._lock:
                    self._status.consecutive_errors += 1
                    self._status.last_error = str(e)
                    errors = self._status.consecutive_errors

                logger.warning(f"VLA step error ({errors}/{CONSECUTIVE_ERROR_LIMIT}): {e}")

                if errors >= CONSECUTIVE_ERROR_LIMIT:
                    logger.error("VLA: Too many consecutive errors, stopping")
                    break

                # Clear stale actions on error so we re-fetch
                action_queue.clear()
                time.sleep(0.5)
                continue

        with self._lock:
            self._status.active = False
        logger.info(f"VLA control loop ended: {action_count} actions in {move_count} moves")

    def _get_status_snapshot(self) -> VLAStatus:
        """Return a copy of current status (caller must hold lock)."""
        s = self._status
        return VLAStatus(
            active=s.active,
            task=s.task,
            step_count=s.step_count,
            server_url=s.server_url,
            last_inference_ms=s.last_inference_ms,
            last_action=list(s.last_action),
            consecutive_errors=s.consecutive_errors,
            last_error=s.last_error,
            hz=s.hz,
            chunk_remaining=s.chunk_remaining,
        )

    def get_status(self) -> dict:
        """Get current VLA status as dict."""
        with self._lock:
            s = self._get_status_snapshot()

        return {
            "active": s.active,
            "task": s.task,
            "step_count": s.step_count,
            "server_url": s.server_url,
            "last_inference_ms": round(s.last_inference_ms, 1),
            "last_action": [round(a, 6) for a in s.last_action] if s.last_action else [],
            "consecutive_errors": s.consecutive_errors,
            "last_error": s.last_error,
            "hz": round(s.hz, 1),
            "chunk_remaining": s.chunk_remaining,
        }
