"""franka-rt ZMQ server: holds panda-py connection, serves commands over ZMQ.

Usage:
    python -m franka_rt [--hostname 192.168.0.253]

Three ZMQ sockets:
    Command (ROUTER): Receives requests, sends responses. ROUTER allows
        multiple clients and handles stop arriving during blocking moves.
    Stop (SUB): Emergency stop channel - never queued behind commands.
    State (PUB): Broadcast during RTC loops (future).
"""

import logging
import signal
import time

import zmq
import msgpack

from .protocol import (
    CMD_ENDPOINT, STOP_ENDPOINT, STATE_ENDPOINT,
    CMD_CONNECT, CMD_GET_STATE, CMD_GET_Q,
    CMD_MOVE_JOINTS, CMD_MOVE_JOINTS_MONITORED, CMD_MOVE_TO_POSE,
    CMD_GRIPPER_MOVE, CMD_GRIPPER_GRASP, CMD_GRIPPER_READ, CMD_GRIPPER_STOP,
    CMD_STOP, CMD_RECOVER, CMD_TEACHING_MODE,
    CMD_SERVO_PICK, CMD_SERVO_STATUS, CMD_PING,
    make_response, parse_request,
)
from .robot_proxy import RobotProxy

logger = logging.getLogger(__name__)


class FrankaRTServer:
    """ZMQ server wrapping RobotProxy."""

    def __init__(self, hostname: str = "192.168.0.253"):
        self.hostname = hostname
        self.proxy = RobotProxy()
        self._running = False
        self._ctx = None

    def run(self):
        """Main event loop. Blocks until SIGINT/SIGTERM or stop."""
        self._ctx = zmq.Context()

        # Command socket: ROUTER (async replies, multiple clients)
        cmd_sock = self._ctx.socket(zmq.ROUTER)
        cmd_sock.bind(CMD_ENDPOINT)

        # Stop socket: SUB (receives emergency stop signals)
        stop_sock = self._ctx.socket(zmq.SUB)
        stop_sock.bind(STOP_ENDPOINT)
        stop_sock.subscribe(b"")

        # State socket: PUB (future: broadcast during RTC)
        state_sock = self._ctx.socket(zmq.PUB)
        state_sock.bind(STATE_ENDPOINT)

        poller = zmq.Poller()
        poller.register(cmd_sock, zmq.POLLIN)
        poller.register(stop_sock, zmq.POLLIN)

        self._running = True
        logger.info(f"franka-rt server starting (hostname={self.hostname})")
        logger.info(f"  CMD: {CMD_ENDPOINT}")
        logger.info(f"  STOP: {STOP_ENDPOINT}")

        # Install signal handlers
        def _shutdown(signum, frame):
            logger.info(f"Signal {signum} received, shutting down")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            while self._running:
                events = dict(poller.poll(100))  # 100ms timeout

                # Emergency stop has highest priority
                if stop_sock in events:
                    stop_sock.recv()  # consume the message
                    logger.warning("EMERGENCY STOP received")
                    self.proxy.stop()

                # Process commands
                if cmd_sock in events:
                    frames = cmd_sock.recv_multipart()
                    # ROUTER frames: [client_id, empty, data]
                    if len(frames) >= 3:
                        client_id = frames[0]
                        data = frames[-1]
                    elif len(frames) == 2:
                        # Some DEALER clients don't send empty frame
                        client_id = frames[0]
                        data = frames[1]
                    else:
                        continue

                    try:
                        msg = msgpack.unpackb(data, raw=False)
                        response = self._handle_command(msg)
                        reply = make_response(
                            msg.get("id", ""),
                            result=response.get("result"),
                            error=response.get("error"),
                        )
                        cmd_sock.send_multipart([client_id, b"", reply])
                    except Exception as e:
                        logger.error(f"Command handling error: {e}", exc_info=True)
                        reply = make_response(
                            msg.get("id", "") if 'msg' in dir() else "",
                            error=str(e),
                        )
                        try:
                            cmd_sock.send_multipart([client_id, b"", reply])
                        except Exception:
                            pass

        finally:
            logger.info("Shutting down franka-rt server")
            cmd_sock.close()
            stop_sock.close()
            state_sock.close()
            self._ctx.term()

    def _handle_command(self, msg: dict) -> dict:
        """Dispatch a command to the appropriate RobotProxy method."""
        cmd = msg.get("cmd")
        args = msg.get("args", {})

        try:
            if cmd == CMD_PING:
                return {"result": {"pong": True, "connected": self.proxy.connected}}

            elif cmd == CMD_CONNECT:
                hostname = args.get("hostname", self.hostname)
                result = self.proxy.connect(hostname)
                return {"result": result}

            elif cmd == CMD_GET_STATE:
                result = self.proxy.get_state()
                return {"result": result}

            elif cmd == CMD_GET_Q:
                result = self.proxy.get_q()
                return {"result": result}

            elif cmd == CMD_MOVE_JOINTS:
                result = self.proxy.move_joints(
                    q=args["q"],
                    speed_factor=args.get("speed_factor", 0.15),
                    dq_threshold=args.get("dq_threshold", 0.01),
                    success_threshold=args.get("success_threshold", 0.05),
                )
                return {"result": result}

            elif cmd == CMD_MOVE_TO_POSE:
                result = self.proxy.move_to_pose(
                    positions=args["positions"],
                    orientations=args["orientations"],
                    speed_factor=args.get("speed_factor", 0.1),
                )
                return {"result": result}

            elif cmd == CMD_MOVE_JOINTS_MONITORED:
                result = self.proxy.move_joints_monitored(
                    q=args["q"],
                    speed_factor=args.get("speed_factor", 0.15),
                    force_baseline=args.get("force_baseline"),
                    dq_threshold=args.get("dq_threshold", 0.01),
                    success_threshold=args.get("success_threshold", 0.05),
                )
                return {"result": result}

            elif cmd == CMD_GRIPPER_MOVE:
                result = self.proxy.gripper_move(
                    width=args["width"],
                    speed=args.get("speed", 0.1),
                )
                return {"result": result}

            elif cmd == CMD_GRIPPER_GRASP:
                result = self.proxy.gripper_grasp(
                    width=args["width"],
                    speed=args.get("speed", 0.1),
                    force=args.get("force", 20.0),
                    epsilon_inner=args.get("epsilon_inner", 0.005),
                    epsilon_outer=args.get("epsilon_outer", 0.005),
                )
                return {"result": result}

            elif cmd == CMD_GRIPPER_READ:
                result = self.proxy.gripper_read()
                return {"result": result}

            elif cmd == CMD_GRIPPER_STOP:
                result = self.proxy.gripper_stop()
                return {"result": result}

            elif cmd == CMD_STOP:
                result = self.proxy.stop()
                return {"result": result}

            elif cmd == CMD_RECOVER:
                result = self.proxy.recover()
                return {"result": result}

            elif cmd == CMD_TEACHING_MODE:
                result = self.proxy.teaching_mode(active=args.get("active", False))
                return {"result": result}

            elif cmd == CMD_SERVO_PICK:
                if self.proxy.is_mock:
                    return {"result": {"success": True, "mock": True, "phases": ["mock"]}}
                if not self.proxy.connected:
                    return {"error": "Not connected to robot"}
                from .servo import NudgeRTServo
                servo = NudgeRTServo(self.proxy.panda, self.proxy.raw_gripper)
                result = servo.execute(**args)
                return {"result": result}

            elif cmd == CMD_SERVO_STATUS:
                return {"result": {"available": True, "active": False}}

            else:
                return {"error": f"Unknown command: {cmd}"}

        except Exception as e:
            logger.error(f"Error handling {cmd}: {e}", exc_info=True)
            return {"error": str(e)}

    def shutdown(self):
        """Signal the server to stop."""
        self._running = False
