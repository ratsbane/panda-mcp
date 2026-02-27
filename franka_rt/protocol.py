"""Shared protocol for franka-rt ZMQ communication.

Message format (msgpack):
  Request:  {"cmd": str, "args": dict, "id": str}
  Response: {"id": str, "result": any, "error": str|None}

Three ZMQ channels:
  Command:  DEALER↔ROUTER  ipc:///tmp/franka-rt-cmd.sock
  Stop:     PUB→SUB        ipc:///tmp/franka-rt-stop.sock
  State:    PUB→SUB        ipc:///tmp/franka-rt-state.sock  (future)
"""

import uuid
import msgpack
import numpy as np

# Socket endpoints
CMD_ENDPOINT = "ipc:///tmp/franka-rt-cmd.sock"
STOP_ENDPOINT = "ipc:///tmp/franka-rt-stop.sock"
STATE_ENDPOINT = "ipc:///tmp/franka-rt-state.sock"

# Commands
CMD_CONNECT = "connect"
CMD_GET_STATE = "get_state"
CMD_GET_Q = "get_q"
CMD_MOVE_JOINTS = "move_joints"
CMD_MOVE_JOINTS_MONITORED = "move_joints_monitored"
CMD_GRIPPER_MOVE = "gripper_move"
CMD_GRIPPER_GRASP = "gripper_grasp"
CMD_GRIPPER_READ = "gripper_read"
CMD_GRIPPER_STOP = "gripper_stop"
CMD_STOP = "stop"
CMD_RECOVER = "recover"
CMD_TEACHING_MODE = "teaching_mode"
CMD_MOVE_TO_POSE = "move_to_pose"
CMD_SERVO_PICK = "servo_pick"
CMD_SERVO_STATUS = "servo_status"
CMD_PING = "ping"


def make_request(cmd: str, **args) -> bytes:
    """Create a msgpack-encoded request message."""
    msg = {
        "cmd": cmd,
        "args": args,
        "id": uuid.uuid4().hex[:8],
    }
    return msgpack.packb(_prepare(msg), use_bin_type=True)


def parse_request(data: bytes) -> dict:
    """Decode a msgpack request message."""
    return msgpack.unpackb(data, raw=False)


def make_response(msg_id: str, result=None, error=None) -> bytes:
    """Create a msgpack-encoded response message."""
    msg = {
        "id": msg_id,
        "result": _prepare(result),
        "error": error,
    }
    return msgpack.packb(msg, use_bin_type=True)


def parse_response(data: bytes) -> dict:
    """Decode a msgpack response message."""
    return msgpack.unpackb(data, raw=False)


def _prepare(obj):
    """Recursively convert numpy types to Python natives for msgpack."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _prepare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare(v) for v in obj]
    return obj
