"""Gamepad input handler for Franka Panda arm jogging.

Polls a USB gamepad (Xbox 360 compatible) in a background thread and
provides delta velocities for incremental arm motion.

Axis mapping (user facing robot from front):
  Left stick up/down  → robot X (forward/back from base)
  Left stick left/right → robot Y (left/right)
  Right stick up/down → robot Z (up/down)

D-pad (hat) mapping:
  D-pad up/down → pitch (tilt gripper forward/back)
  D-pad left/right → yaw (rotate gripper)

Button mapping (Xbox 360):
  A (0) → Grasp (close gripper with force on block)
  B (1) → Open gripper
  X (2) → Cycle speed: slow (1mm) / medium (5mm) / fast (10mm)
  Y (3) → Reset pitch/yaw to zero (straight down)
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
    "slow": 0.015,    # 15mm per step at full deflection
    "medium": 0.040,  # 40mm per step at full deflection
    "fast": 0.070,    # 70mm per step at full deflection
}
SPEED_NAMES = list(SPEED_PRESETS.keys())

# Angle step per D-pad press (radians)
ANGLE_STEP = 0.05        # ~2.9 degrees per press
ANGLE_STEP_FINE = 0.01   # ~0.6 degrees in fine mode


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
    pitch: float = 0.0   # Accumulated pitch (radians), 0 = straight down
    yaw: float = 0.0     # Accumulated yaw (radians), 0 = default
    speed_name: str = "medium"
    step_size: float = 0.005
    fine_mode: bool = False  # LB held = 1mm steps
    connected: bool = False
    controller_name: str = ""
    events: list = field(default_factory=list)


class GamepadHandler:
    """Polls a USB gamepad and provides delta commands for arm jogging."""

    def __init__(self, deadzone: float = 0.35):
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
        # Accumulated gripper orientation (radians)
        self._pitch = 0.0
        self._yaw = 0.0
        self._prev_hat = (0, 0)  # For edge detection on D-pad

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

            # Calibrate axis centers — read resting position to cancel hardware bias
            import pygame
            pygame.event.pump()
            time.sleep(0.05)
            pygame.event.pump()
            self._axis_offsets = {}
            for ax in [0, 1, 4]:
                raw = self._joystick.get_axis(ax)
                if abs(raw) < 0.25:  # Only offset if small resting bias
                    self._axis_offsets[ax] = raw
                    if abs(raw) > 0.02:
                        logger.info(f"Gamepad: axis {ax} center offset {raw:.3f}")

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
        """Apply deadzone and quadratic response curve.

        Quadratic curve gives more precision at small deflections
        and more speed at large deflections:
          10% stick → 1% of max speed
          50% stick → 25% of max speed
          100% stick → 100% of max speed
        """
        if abs(value) < self.deadzone:
            return 0.0
        # Scale remaining range to 0-1
        sign = 1.0 if value > 0 else -1.0
        normalized = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        # Quadratic response curve for more dynamic range
        return sign * normalized * normalized

    def _poll_loop(self):
        """Background thread: poll gamepad at ~60Hz, update state."""
        import pygame

        while self._running:
            try:
                pygame.event.pump()

                # Read analog sticks (subtract resting offset to cancel bias)
                offsets = getattr(self, '_axis_offsets', {})
                lx = self._apply_deadzone(self._joystick.get_axis(0) - offsets.get(0, 0))  # Left X
                ly = self._apply_deadzone(self._joystick.get_axis(1) - offsets.get(1, 0))  # Left Y
                ry = self._apply_deadzone(self._joystick.get_axis(4) - offsets.get(4, 0))  # Right Y

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

                # Read D-pad hat for pitch/yaw control (edge-triggered)
                if self._joystick.get_numhats() > 0:
                    hat = self._joystick.get_hat(0)  # (x, y) each -1/0/+1
                    angle_step = ANGLE_STEP_FINE if self._fine_mode else ANGLE_STEP

                    # Edge detection: only change on new press, not while held
                    if hat != self._prev_hat:
                        hat_x, hat_y = hat
                        # D-pad up/down → pitch (up = tilt forward = positive pitch)
                        if hat_y != 0:
                            self._pitch += hat_y * angle_step
                            self._pitch = max(-1.0, min(1.0, self._pitch))
                            logger.info(f"Gamepad: pitch → {self._pitch:.3f} rad ({self._pitch * 57.3:.1f}°)")
                        # D-pad left/right → yaw
                        if hat_x != 0:
                            self._yaw += hat_x * angle_step
                            self._yaw = max(-1.0, min(1.0, self._yaw))
                            logger.info(f"Gamepad: yaw → {self._yaw:.3f} rad ({self._yaw * 57.3:.1f}°)")
                        self._prev_hat = hat

                with self._lock:
                    self._state.dx = dx
                    self._state.dy = dy
                    self._state.dz = dz
                    self._state.pitch = self._pitch
                    self._state.yaw = self._yaw
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

        elif button == 3:  # Y = reset pitch/yaw to zero
            self._pitch = 0.0
            self._yaw = 0.0
            logger.info("Gamepad: pitch/yaw reset to 0 (straight down)")

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
