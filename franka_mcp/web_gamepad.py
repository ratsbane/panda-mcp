"""WebSocket-based remote gamepad for browser control of the Franka arm.

Runs a WebSocket server on a configurable port. Browser clients connect and send
gamepad state (stick positions, button events). Provides the same get_state()
interface as GamepadHandler so the jog loop can consume from either source.

Protocol (browser → server):
  {"type": "state", "dx": 0.01, "dy": 0.0, "dz": 0.0, "speed_name": "medium", ...}
  {"type": "event", "action": "grasp"}

Protocol (server → browser):
  {"type": "status", "position": {"x": 0.4, "y": 0.0, "z": 0.3}, ...}
"""

import asyncio
import json
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Reuse data classes from the USB gamepad module
try:
    from .gamepad import GamepadEvent, GamepadState
except ImportError:
    from gamepad import GamepadEvent, GamepadState

DEFAULT_PORT = 8766
WATCHDOG_TIMEOUT = 0.5  # Zero state if no input for this many seconds


class WebGamepad:
    """Virtual gamepad receiving input over WebSocket."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = GamepadState()
        self._events: list[GamepadEvent] = []
        self._last_input_time = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        # Status dict pushed to browser (updated by jog loop via update_status)
        self._status: dict = {}
        self._status_lock = threading.Lock()

    def start(self) -> dict:
        """Start the WebSocket server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        # Wait for server to bind
        time.sleep(0.3)
        return {
            "success": True,
            "controller": f"WebSocket ws://0.0.0.0:{self.port}",
            "port": self.port,
        }

    def _run_server(self):
        """Run asyncio event loop with WebSocket server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebGamepad server error: {e}")

    async def _serve(self):
        """Start WebSocket server and run until stopped."""
        import websockets

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        ):
            logger.info(f"WebGamepad listening on ws://{self.host}:{self.port}")
            while self._running:
                await asyncio.sleep(0.1)

    async def _handle_client(self, websocket):
        """Handle a single WebSocket client."""
        addr = websocket.remote_address
        logger.info(f"WebGamepad client connected: {addr}")
        self._clients.add(websocket)

        # Start status push task
        status_task = asyncio.create_task(self._push_status(websocket))

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self._process_input(data)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            status_task.cancel()
            self._clients.discard(websocket)
            logger.info(f"WebGamepad client disconnected: {addr}")
            # Zero out state on disconnect
            with self._lock:
                self._state = GamepadState()

    async def _push_status(self, websocket):
        """Push robot status to browser at ~5Hz."""
        import websockets

        try:
            while True:
                await asyncio.sleep(0.2)
                with self._status_lock:
                    status = self._status.copy()
                if status:
                    status["type"] = "status"
                    try:
                        await websocket.send(json.dumps(status))
                    except websockets.ConnectionClosed:
                        break
        except asyncio.CancelledError:
            pass

    def _process_input(self, data: dict):
        """Process a gamepad input message from the browser."""
        msg_type = data.get("type", "state")
        self._last_input_time = time.time()

        if msg_type == "state":
            with self._lock:
                self._state.dx = float(data.get("dx", 0))
                self._state.dy = float(data.get("dy", 0))
                self._state.dz = float(data.get("dz", 0))
                if "pitch" in data:
                    self._state.pitch = float(data["pitch"])
                if "yaw" in data:
                    self._state.yaw = float(data["yaw"])
                self._state.step_size = float(data.get("step_size", self._state.step_size))
                self._state.speed_name = data.get("speed_name", self._state.speed_name)
                self._state.fine_mode = bool(data.get("fine_mode", False))
                self._state.connected = True
                self._state.controller_name = data.get("controller", "Browser")

        elif msg_type == "event":
            action = data.get("action")
            valid_actions = ("grasp", "open_gripper", "stop_jog", "cycle_speed", "home")
            if action in valid_actions:
                with self._lock:
                    self._events.append(GamepadEvent(action, time.time()))
                logger.info(f"WebGamepad event: {action}")

        elif msg_type == "pitch_yaw":
            with self._lock:
                dp = float(data.get("delta_pitch", 0))
                dy = float(data.get("delta_yaw", 0))
                self._state.pitch = max(-1.0, min(1.0, self._state.pitch + dp))
                self._state.yaw = max(-1.0, min(1.0, self._state.yaw + dy))

    def get_state(self) -> GamepadState:
        """Get current state (same interface as GamepadHandler).

        Includes watchdog: zeros motion if no input received recently.
        """
        now = time.time()
        with self._lock:
            # Watchdog: if no input for WATCHDOG_TIMEOUT, zero out motion
            if self._last_input_time > 0 and (now - self._last_input_time) > WATCHDOG_TIMEOUT:
                self._state.dx = 0.0
                self._state.dy = 0.0
                self._state.dz = 0.0

            state = GamepadState(
                dx=self._state.dx,
                dy=self._state.dy,
                dz=self._state.dz,
                pitch=self._state.pitch,
                yaw=self._state.yaw,
                speed_name=self._state.speed_name,
                step_size=self._state.step_size,
                fine_mode=self._state.fine_mode,
                connected=self._state.connected,
                controller_name=self._state.controller_name,
                events=list(self._events),
            )
            self._events.clear()
            return state

    def update_status(self, position: dict, gripper_width: float, speed: str, **extra):
        """Update status dict that gets pushed to browser clients.

        Called by the jog loop to provide real-time feedback.
        """
        with self._status_lock:
            self._status = {
                "position": position,
                "gripper_width": round(gripper_width, 4),
                "speed": speed,
                **extra,
            }

    def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        # Clean up the event loop
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception:
                pass
        logger.info("WebGamepad stopped")

    @property
    def running(self) -> bool:
        return self._running
