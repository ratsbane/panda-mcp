"""ZMQ client for franka-rt server.

Provides convenience methods matching the panda-py API surface that
controller.py expects, but routes everything through ZMQ to franka-rt.
"""

import logging
import zmq
import msgpack

from franka_rt.protocol import (
    CMD_ENDPOINT, STOP_ENDPOINT,
    CMD_CONNECT, CMD_GET_STATE, CMD_GET_Q,
    CMD_MOVE_JOINTS, CMD_MOVE_JOINTS_MONITORED, CMD_MOVE_TO_POSE,
    CMD_GRIPPER_MOVE, CMD_GRIPPER_GRASP, CMD_GRIPPER_READ, CMD_GRIPPER_STOP,
    CMD_STOP, CMD_RECOVER, CMD_TEACHING_MODE,
    CMD_SERVO_PICK, CMD_SERVO_STATUS, CMD_PING,
    make_request, parse_response,
)

logger = logging.getLogger(__name__)

# Default timeout for ZMQ recv (ms). Most commands complete within
# their own hardware timeouts (15s for motion, 10s for gripper).
# Add buffer for ZMQ overhead.
DEFAULT_TIMEOUT_MS = 20_000
MOTION_TIMEOUT_MS = 20_000
GRIPPER_TIMEOUT_MS = 15_000
STATE_TIMEOUT_MS = 10_000


class RTClientError(Exception):
    """Error from franka-rt server."""
    pass


class RTClientTimeout(RTClientError):
    """ZMQ recv timed out waiting for franka-rt response."""
    pass


class FrankaRTClient:
    """ZMQ DEALER client for franka-rt server.

    Thread-safe for send_command (each call uses its own socket recv).
    Emergency stop uses a separate PUB socket.
    """

    def __init__(self):
        self._ctx = zmq.Context.instance()
        self._cmd_sock = None
        self._stop_sock = None
        self._connected_to_server = False

    def connect_to_server(self):
        """Connect ZMQ sockets to franka-rt server."""
        if self._cmd_sock is not None:
            return  # Already connected

        # Command socket: DEALER
        self._cmd_sock = self._ctx.socket(zmq.DEALER)
        self._cmd_sock.setsockopt(zmq.LINGER, 1000)
        self._cmd_sock.connect(CMD_ENDPOINT)

        # Stop socket: PUB (emergency stop)
        self._stop_sock = self._ctx.socket(zmq.PUB)
        self._stop_sock.setsockopt(zmq.LINGER, 0)
        self._stop_sock.connect(STOP_ENDPOINT)

        self._connected_to_server = True
        logger.info(f"Connected to franka-rt at {CMD_ENDPOINT}")

    def close(self):
        """Close ZMQ sockets."""
        if self._cmd_sock:
            self._cmd_sock.close()
            self._cmd_sock = None
        if self._stop_sock:
            self._stop_sock.close()
            self._stop_sock = None
        self._connected_to_server = False

    def send_command(self, cmd: str, timeout_ms: int = DEFAULT_TIMEOUT_MS,
                     **args) -> dict:
        """Send command to franka-rt and wait for response.

        Returns the 'result' field from the response, or raises RTClientError
        if the server returned an error.
        """
        if not self._connected_to_server:
            self.connect_to_server()

        request = make_request(cmd, **args)
        self._cmd_sock.send(request)

        # Wait for response with timeout
        if self._cmd_sock.poll(timeout_ms, zmq.POLLIN):
            frames = self._cmd_sock.recv_multipart()
            # DEALER receives: [empty, data] or just [data]
            data = frames[-1]
            resp = parse_response(data)

            if resp.get("error"):
                raise RTClientError(resp["error"])
            return resp.get("result", {})
        else:
            raise RTClientTimeout(
                f"No response from franka-rt after {timeout_ms}ms for {cmd}")

    def emergency_stop(self):
        """Send emergency stop on PUB channel. Never blocks."""
        if self._stop_sock:
            self._stop_sock.send(b"STOP")
            logger.warning("Emergency stop sent")
        # Also try command channel stop
        try:
            self.send_command(CMD_STOP, timeout_ms=5000)
        except Exception:
            pass

    # --- Convenience methods matching panda-py API ---

    def connect(self, hostname: str = "192.168.0.253") -> dict:
        """Connect franka-rt to the robot hardware."""
        return self.send_command(CMD_CONNECT, hostname=hostname)

    def ping(self) -> dict:
        """Check if franka-rt is alive."""
        return self.send_command(CMD_PING, timeout_ms=3000)

    def get_state(self) -> dict:
        """Get full robot state as dict.

        Returns: {q, ee_position, ee_orientation, gripper_width, is_grasped,
                  has_error, error_message, O_F_ext_hat_K, tau_ext_hat_filtered}
        """
        return self.send_command(CMD_GET_STATE, timeout_ms=STATE_TIMEOUT_MS)

    def get_q(self) -> list:
        """Get current joint positions (7 floats)."""
        return self.send_command(CMD_GET_Q, timeout_ms=STATE_TIMEOUT_MS)

    def move_to_joint_position(self, q, speed_factor=0.15,
                                dq_threshold=0.01,
                                success_threshold=0.05) -> dict:
        """Move to joint configuration. q: list of 7 floats or list of waypoints."""
        # Ensure q is a plain list
        q_list = _to_list(q)
        return self.send_command(
            CMD_MOVE_JOINTS,
            timeout_ms=MOTION_TIMEOUT_MS,
            q=q_list,
            speed_factor=speed_factor,
            dq_threshold=dq_threshold,
            success_threshold=success_threshold,
        )

    def move_to_pose(self, positions, orientations, speed_factor=0.1) -> dict:
        """Move via Cartesian planner (legacy — IK is preferred)."""
        pos_list = [_to_list(p) for p in positions]
        ori_list = [_to_list(o) for o in orientations]
        return self.send_command(
            CMD_MOVE_TO_POSE, timeout_ms=MOTION_TIMEOUT_MS,
            positions=pos_list, orientations=ori_list,
            speed_factor=speed_factor,
        )

    def move_joints_monitored(self, q, speed_factor=0.15,
                              force_baseline=None,
                              dq_threshold=0.01,
                              success_threshold=0.05) -> dict:
        """Move with force monitoring. Returns {success, force_events}."""
        q_list = _to_list(q)
        bl = _to_list(force_baseline) if force_baseline is not None else None
        return self.send_command(
            CMD_MOVE_JOINTS_MONITORED,
            timeout_ms=MOTION_TIMEOUT_MS,
            q=q_list,
            speed_factor=speed_factor,
            force_baseline=bl,
            dq_threshold=dq_threshold,
            success_threshold=success_threshold,
        )

    def gripper_move(self, width: float, speed: float = 0.1) -> dict:
        """Move gripper to width (no force)."""
        return self.send_command(
            CMD_GRIPPER_MOVE, timeout_ms=GRIPPER_TIMEOUT_MS,
            width=float(width), speed=float(speed),
        )

    def gripper_grasp(self, width: float, speed: float = 0.1,
                      force: float = 20.0,
                      epsilon_inner: float = 0.005,
                      epsilon_outer: float = 0.005) -> dict:
        """Grasp at width with force."""
        return self.send_command(
            CMD_GRIPPER_GRASP, timeout_ms=GRIPPER_TIMEOUT_MS,
            width=float(width), speed=float(speed), force=float(force),
            epsilon_inner=float(epsilon_inner),
            epsilon_outer=float(epsilon_outer),
        )

    def gripper_read_once(self) -> dict:
        """Read gripper state. Returns {width, is_grasped}."""
        return self.send_command(CMD_GRIPPER_READ, timeout_ms=STATE_TIMEOUT_MS)

    def gripper_stop(self) -> dict:
        """Stop gripper motion."""
        return self.send_command(CMD_GRIPPER_STOP, timeout_ms=5000)

    def stop_controller(self) -> dict:
        """Stop all motion and recover."""
        return self.send_command(CMD_STOP, timeout_ms=5000)

    def recover(self) -> dict:
        """Recover from error state."""
        return self.send_command(CMD_RECOVER, timeout_ms=STATE_TIMEOUT_MS)

    def teaching_mode(self, active: bool) -> dict:
        """Enable/disable teaching mode."""
        return self.send_command(
            CMD_TEACHING_MODE, timeout_ms=STATE_TIMEOUT_MS,
            active=active,
        )

    def servo_pick(self, **kwargs) -> dict:
        """Run NUDGE servo pick (Phase 3)."""
        return self.send_command(CMD_SERVO_PICK, timeout_ms=30_000, **kwargs)

    def servo_status(self) -> dict:
        """Get servo status."""
        return self.send_command(CMD_SERVO_STATUS, timeout_ms=3000)


def _to_list(obj):
    """Convert numpy arrays to nested lists for msgpack serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        # Check if it's a list of arrays (waypoints)
        result = []
        for item in obj:
            if isinstance(item, np.ndarray):
                result.append(item.tolist())
            else:
                result.append(item)
        return result
    return obj
