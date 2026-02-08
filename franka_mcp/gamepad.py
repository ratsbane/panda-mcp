"""Gamepad input handler for Franka Panda arm jogging.

Polls a USB gamepad (Xbox 360 compatible) in a background thread and
provides delta velocities for incremental arm motion.

Axis mapping (user facing robot from front):
  Left stick up/down  → robot X (forward/back from base)
  Left stick left/right → robot Y (left/right)
  Right stick up/down → robot Z (up/down)

Button mapping (Xbox 360):
  A (0) → Grasp (close gripper with force on block)
  B (1) → Open gripper
  X (2) → Cycle speed: slow (1mm) / medium (5mm) / fast (10mm)
  Y (3) → unused
  LB (4) → Fine mode (1mm steps while held)
  RB (5) → unused
  Back (6) → Stop jog mode
  Start (7) → unused
"""

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SPEED_PRESETS = {
    "slow": 0.010,    # 10mm per step
    "medium": 0.025,  # 25mm per step
    "fast": 0.050,    # 50mm per step
}
SPEED_NAMES = list(SPEED_PRESETS.keys())


@dataclass
class GamepadEvent:
    """A discrete event from a button press."""
    action: str  # "grasp", "open_gripper", "stop_jog", "cycle_speed"
    timestamp: float = 0.0


@dataclass
class GamepadState:
    """Current gamepad input state."""
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    speed_name: str = "medium"
    step_size: float = 0.005
    fine_mode: bool = False  # LB held = 1mm steps
    connected: bool = False
    controller_name: str = ""
    events: list = field(default_factory=list)


class GamepadHandler:
    """Polls a USB gamepad and provides delta commands for arm jogging."""

    def __init__(self, deadzone: float = 0.15):
        self.deadzone = deadzone
        self._speed_idx = 1  # Start at "medium"
        self._step_size = SPEED_PRESETS["medium"]
        self._fine_mode = False
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._state = GamepadState()
        self._events: list[GamepadEvent] = []
        self._joystick = None
        self._controller_name = ""

    def start(self) -> dict:
        """Initialize pygame joystick and start polling thread."""
        try:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            pygame.display.init()  # minimal init needed for event system
            pygame.joystick.init()

            count = pygame.joystick.get_count()
            if count == 0:
                return {"success": False, "error": "No gamepad detected"}

            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._controller_name = self._joystick.get_name()

            logger.info(f"Gamepad connected: {self._controller_name} "
                        f"({self._joystick.get_numaxes()} axes, "
                        f"{self._joystick.get_numbuttons()} buttons)")

            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

            return {
                "success": True,
                "controller": self._controller_name,
                "axes": self._joystick.get_numaxes(),
                "buttons": self._joystick.get_numbuttons(),
            }
        except Exception as e:
            logger.error(f"Gamepad init failed: {e}")
            return {"success": False, "error": str(e)}

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to axis value."""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale remaining range to 0-1
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _poll_loop(self):
        """Background thread: poll gamepad at ~60Hz, update state."""
        import pygame

        while self._running:
            try:
                pygame.event.pump()

                # Read analog sticks
                lx = self._apply_deadzone(self._joystick.get_axis(0))  # Left X
                ly = self._apply_deadzone(self._joystick.get_axis(1))  # Left Y
                ry = self._apply_deadzone(self._joystick.get_axis(4))  # Right Y

                # Check LB (button 4) for fine mode
                self._fine_mode = self._joystick.get_button(4) == 1

                # Effective step size
                effective_step = 0.003 if self._fine_mode else self._step_size

                # Map to robot deltas
                # Left stick up (-1) = +X (forward from base)
                # Left stick right (+1) = +Y (robot left)
                # Right stick up (-1) = +Z (up)
                dx = -ly * effective_step
                dy = lx * effective_step
                dz = -ry * effective_step

                with self._lock:
                    self._state.dx = dx
                    self._state.dy = dy
                    self._state.dz = dz
                    self._state.step_size = effective_step
                    self._state.fine_mode = self._fine_mode
                    self._state.speed_name = "fine" if self._fine_mode else SPEED_NAMES[self._speed_idx]
                    self._state.connected = True
                    self._state.controller_name = self._controller_name

                # Check button presses (edge detection)
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        self._handle_button(event.button)

            except Exception as e:
                logger.warning(f"Gamepad poll error: {e}")

            time.sleep(1 / 60.0)  # 60Hz poll

    def _handle_button(self, button: int):
        """Handle button press events."""
        now = time.time()

        if button == 0:  # A = grasp
            with self._lock:
                self._events.append(GamepadEvent("grasp", now))
            logger.info("Gamepad: GRASP")

        elif button == 1:  # B = open gripper
            with self._lock:
                self._events.append(GamepadEvent("open_gripper", now))
            logger.info("Gamepad: OPEN GRIPPER")

        elif button == 2:  # X = cycle speed
            self._speed_idx = (self._speed_idx + 1) % len(SPEED_NAMES)
            self._step_size = SPEED_PRESETS[SPEED_NAMES[self._speed_idx]]
            logger.info(f"Gamepad: speed → {SPEED_NAMES[self._speed_idx]} ({self._step_size*1000:.0f}mm)")

        elif button == 6:  # Back = stop jog
            with self._lock:
                self._events.append(GamepadEvent("stop_jog", now))
            logger.info("Gamepad: STOP JOG")

    def get_state(self) -> GamepadState:
        """Get current gamepad state (thread-safe)."""
        with self._lock:
            state = GamepadState(
                dx=self._state.dx,
                dy=self._state.dy,
                dz=self._state.dz,
                speed_name=self._state.speed_name,
                step_size=self._state.step_size,
                fine_mode=self._state.fine_mode,
                connected=self._state.connected,
                controller_name=self._state.controller_name,
                events=list(self._events),
            )
            self._events.clear()
            return state

    def stop(self):
        """Stop polling and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            import pygame
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        logger.info("Gamepad stopped")

    @property
    def running(self) -> bool:
        return self._running
