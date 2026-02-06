"""
Controller for the SO-ARM100 robot arm.

Uses Feetech STS3215 servos via serial communication.
Includes port discovery, diagnostics, and calibration.
"""

import glob
import json
import math
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    from scservo_sdk import PortHandler, protocol_packet_handler, COMM_SUCCESS
    HAS_SERVO_SDK = True
except ImportError:
    HAS_SERVO_SDK = False

logger = logging.getLogger(__name__)

# STS3215 Register addresses
ADDR_ID = 5
ADDR_MODEL = 3
ADDR_OPERATING_MODE = 33
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 58
ADDR_PRESENT_LOAD = 60
ADDR_VOLTAGE = 62
ADDR_TEMPERATURE = 63
ADDR_MOVING = 66

# Motor configuration
MOTOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}
EXPECTED_MOTOR_COUNT = 6
STS3215_MODEL_NUMBER = 777

# Conversion constants
TICKS_PER_REV = 4096
CENTER_POSITION = 2048

# Default position limits (conservative, updated by calibration)
DEFAULT_POSITION_LIMITS = {
    'shoulder_pan': (500, 3500),
    'shoulder_lift': (700, 2300),
    'elbow_flex': (800, 3100),
    'wrist_flex': (900, 3300),
    'wrist_roll': (150, 3950),
    'gripper': (1800, 3500),
}

# Diagnostic thresholds
VOLTAGE_MIN = 6.0  # Below this, power supply issue
VOLTAGE_MAX = 13.0  # Above this, overvoltage
VOLTAGE_USB_ONLY = 5.5  # Below this, likely USB power only (no external)
TEMP_WARNING = 50  # Celsius
TEMP_CRITICAL = 65  # Celsius
COMM_RETRIES = 3
STALL_LOAD_THRESHOLD = 40.0  # % load indicating stall

# Calibration file location
CALIBRATION_DIR = Path(__file__).parent
CALIBRATION_FILE = CALIBRATION_DIR / 'calibration.json'


def ticks_to_degrees(ticks: int) -> float:
    """Convert encoder ticks to degrees from center."""
    return (ticks - CENTER_POSITION) * 360.0 / TICKS_PER_REV


def degrees_to_ticks(degrees: float) -> int:
    """Convert degrees from center to encoder ticks."""
    return int(degrees * TICKS_PER_REV / 360.0 + CENTER_POSITION)


def ticks_to_radians(ticks: int) -> float:
    """Convert encoder ticks to radians from center."""
    return (ticks - CENTER_POSITION) * 2 * math.pi / TICKS_PER_REV


def radians_to_ticks(radians: float) -> int:
    """Convert radians from center to encoder ticks."""
    return int(radians * TICKS_PER_REV / (2 * math.pi) + CENTER_POSITION)


@dataclass
class MotorDiagnostic:
    """Diagnostic result for a single motor."""
    name: str
    motor_id: int
    responding: bool
    position: Optional[int] = None
    voltage: Optional[float] = None
    temperature: Optional[int] = None
    load_pct: Optional[float] = None
    torque_enabled: Optional[bool] = None
    issues: list[str] = field(default_factory=list)
    severity: str = "ok"  # ok, warning, error


@dataclass
class SO100State:
    """Current state of the SO-ARM100."""
    positions: dict[str, int]  # ticks
    positions_deg: dict[str, float]  # degrees
    voltages: dict[str, float]  # volts
    temperatures: dict[str, int]  # celsius
    torque_enabled: dict[str, bool]
    moving: dict[str, bool]


def discover_ports() -> list[dict]:
    """
    Scan serial ports for connected SO-ARM100 arms.

    Returns a list of dicts with port info and discovered motors.
    """
    if not HAS_SERVO_SDK:
        return [{"error": "scservo_sdk not installed"}]

    candidates = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    if not candidates:
        return []

    results = []
    for port_path in candidates:
        port_info = {"port": port_path, "motors": [], "is_so100": False}
        try:
            port = PortHandler(port_path)
            if not port.openPort():
                port_info["error"] = "Failed to open"
                results.append(port_info)
                continue

            port.setBaudRate(1000000)
            ph = protocol_packet_handler()

            # Scan motor IDs 1-6 for STS3215 servos
            found_motors = []
            for motor_id in range(1, 7):
                model, result, _ = ph.read2ByteTxRx(port, motor_id, ADDR_MODEL)
                if result == COMM_SUCCESS:
                    pos, _, _ = ph.read2ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
                    voltage, _, _ = ph.read1ByteTxRx(port, motor_id, ADDR_VOLTAGE)
                    found_motors.append({
                        "id": motor_id,
                        "model": model,
                        "is_sts3215": model == STS3215_MODEL_NUMBER,
                        "position": pos,
                        "voltage": voltage / 10.0,
                    })

            port.closePort()

            port_info["motors"] = found_motors
            port_info["motor_count"] = len(found_motors)
            # An SO-ARM100 has exactly 6 STS3215 motors with IDs 1-6
            sts_count = sum(1 for m in found_motors if m["is_sts3215"])
            port_info["is_so100"] = sts_count == EXPECTED_MOTOR_COUNT

        except Exception as e:
            port_info["error"] = str(e)

        results.append(port_info)

    return results


class SO100Controller:
    """Controller for the SO-ARM100 robot arm."""

    def __init__(self):
        self.port: Optional[PortHandler] = None
        self.packet_handler: Optional[protocol_packet_handler] = None
        self.connected = False
        self.port_name: Optional[str] = None
        self._mock = not HAS_SERVO_SDK
        self.position_limits = dict(DEFAULT_POSITION_LIMITS)
        self._load_calibration()

    def _load_calibration(self):
        """Load calibration data from file if it exists."""
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE) as f:
                    cal = json.load(f)
                if "position_limits" in cal:
                    self.position_limits.update(cal["position_limits"])
                    logger.info(f"Loaded calibration from {CALIBRATION_FILE}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")

    def _save_calibration(self, data: dict):
        """Save calibration data to file."""
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved calibration to {CALIBRATION_FILE}")

    def connect(self, port: str = '/dev/ttyACM0', baudrate: int = 1000000) -> dict:
        """Connect to the SO-ARM100 via serial port."""
        if self._mock:
            logger.warning("Running in mock mode - no servo SDK available")
            self.connected = True
            self.port_name = port
            return {"connected": True, "mock": True, "port": port}

        self.port = PortHandler(port)
        self.packet_handler = protocol_packet_handler()

        if not self.port.openPort():
            raise RuntimeError(f"Failed to open port {port}")

        if not self.port.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate to {baudrate}")

        self.connected = True
        self.port_name = port

        # Verify motors are responding
        found_motors = []
        missing_motors = []
        for name, motor_id in MOTOR_IDS.items():
            pos, result, _ = self.packet_handler.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
            if result == COMM_SUCCESS:
                found_motors.append(name)
            else:
                missing_motors.append(name)

        if missing_motors:
            logger.warning(f"Missing motors: {missing_motors}")

        logger.info(f"Connected to SO-ARM100 on {port}, found motors: {found_motors}")

        return {
            "connected": True,
            "port": port,
            "baudrate": baudrate,
            "motors_found": found_motors,
            "motors_missing": missing_motors,
        }

    def disconnect(self):
        """Disconnect from the arm."""
        if self.port and not self._mock:
            for motor_id in MOTOR_IDS.values():
                try:
                    self.packet_handler.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, 0)
                except Exception:
                    pass
            self.port.closePort()
        self.connected = False
        self.port = None
        self.packet_handler = None

    def _check_connected(self):
        """Raise error if not connected."""
        if not self.connected:
            raise RuntimeError("Not connected to SO-ARM100. Call connect() first.")

    def _read_with_retry(self, motor_id: int, addr: int, num_bytes: int = 2) -> tuple:
        """Read a register with retries. Returns (value, success)."""
        for attempt in range(COMM_RETRIES):
            if num_bytes == 1:
                val, result, err = self.packet_handler.read1ByteTxRx(self.port, motor_id, addr)
            else:
                val, result, err = self.packet_handler.read2ByteTxRx(self.port, motor_id, addr)
            if result == COMM_SUCCESS:
                return val, True
            time.sleep(0.01)
        return 0, False

    def _read_load_pct(self, motor_id: int) -> Optional[float]:
        """Read load as a signed percentage."""
        load, ok = self._read_with_retry(motor_id, ADDR_PRESENT_LOAD)
        if not ok:
            return None
        if load > 1023:
            return -(load - 1024) / 10.0
        return load / 10.0

    def get_state(self) -> SO100State:
        """Get current state of all motors."""
        self._check_connected()

        if self._mock:
            return SO100State(
                positions={name: CENTER_POSITION for name in MOTOR_NAMES},
                positions_deg={name: 0.0 for name in MOTOR_NAMES},
                voltages={name: 8.4 for name in MOTOR_NAMES},
                temperatures={name: 25 for name in MOTOR_NAMES},
                torque_enabled={name: False for name in MOTOR_NAMES},
                moving={name: False for name in MOTOR_NAMES},
            )

        positions = {}
        positions_deg = {}
        voltages = {}
        temperatures = {}
        torque_enabled = {}
        moving = {}

        for name, motor_id in MOTOR_IDS.items():
            pos, _ = self._read_with_retry(motor_id, ADDR_PRESENT_POSITION)
            voltage, _ = self._read_with_retry(motor_id, ADDR_VOLTAGE, 1)
            temp, _ = self._read_with_retry(motor_id, ADDR_TEMPERATURE, 1)
            torque, _ = self._read_with_retry(motor_id, ADDR_TORQUE_ENABLE, 1)
            mov, _ = self._read_with_retry(motor_id, ADDR_MOVING, 1)

            positions[name] = pos
            positions_deg[name] = ticks_to_degrees(pos)
            voltages[name] = voltage / 10.0
            temperatures[name] = temp
            torque_enabled[name] = bool(torque)
            moving[name] = bool(mov)

        return SO100State(
            positions=positions,
            positions_deg=positions_deg,
            voltages=voltages,
            temperatures=temperatures,
            torque_enabled=torque_enabled,
            moving=moving,
        )

    def diagnose(self) -> list[dict]:
        """
        Run diagnostics on all motors.

        Checks for:
        - Unresponsive motors (possible wire disconnect)
        - Voltage issues (no external power, overvoltage)
        - Temperature warnings
        - Communication reliability
        - Stall conditions (high load at rest)
        """
        self._check_connected()

        if self._mock:
            return [{"info": "Running in mock mode, no real diagnostics"}]

        diagnostics = []
        voltage_readings = []

        for name, motor_id in MOTOR_IDS.items():
            diag = MotorDiagnostic(name=name, motor_id=motor_id, responding=False)

            # Test communication with retries
            comm_successes = 0
            for _ in range(COMM_RETRIES):
                pos, result, _ = self.packet_handler.read2ByteTxRx(
                    self.port, motor_id, ADDR_PRESENT_POSITION
                )
                if result == COMM_SUCCESS:
                    comm_successes += 1
                    diag.position = pos
                time.sleep(0.01)

            if comm_successes == 0:
                diag.responding = False
                diag.issues.append("Motor not responding - check wire connection at back of motor")
                diag.severity = "error"
                diagnostics.append(asdict(diag))
                continue

            diag.responding = True
            if comm_successes < COMM_RETRIES:
                diag.issues.append(
                    f"Intermittent communication ({comm_successes}/{COMM_RETRIES} reads succeeded) "
                    "- possible loose wire"
                )
                diag.severity = "warning"

            # Read voltage
            voltage_raw, ok = self._read_with_retry(motor_id, ADDR_VOLTAGE, 1)
            if ok:
                diag.voltage = voltage_raw / 10.0
                voltage_readings.append(diag.voltage)

                if diag.voltage < VOLTAGE_USB_ONLY:
                    diag.issues.append(
                        f"Very low voltage ({diag.voltage}V) - external power supply "
                        "may not be connected (USB only provides ~5V)"
                    )
                    diag.severity = "error"
                elif diag.voltage < VOLTAGE_MIN:
                    diag.issues.append(
                        f"Low voltage ({diag.voltage}V) - check power supply"
                    )
                    diag.severity = max(diag.severity, "warning", key=["ok", "warning", "error"].index)
                elif diag.voltage > VOLTAGE_MAX:
                    diag.issues.append(
                        f"High voltage ({diag.voltage}V) - check power supply rating"
                    )
                    diag.severity = "error"

            # Read temperature
            temp, ok = self._read_with_retry(motor_id, ADDR_TEMPERATURE, 1)
            if ok:
                diag.temperature = temp
                if temp >= TEMP_CRITICAL:
                    diag.issues.append(
                        f"Critical temperature ({temp}°C) - motor may be stalled or overloaded. "
                        "Let it cool before operating."
                    )
                    diag.severity = "error"
                elif temp >= TEMP_WARNING:
                    diag.issues.append(f"Elevated temperature ({temp}°C) - monitor closely")
                    diag.severity = max(diag.severity, "warning", key=["ok", "warning", "error"].index)

            # Read torque enable
            torque, ok = self._read_with_retry(motor_id, ADDR_TORQUE_ENABLE, 1)
            if ok:
                diag.torque_enabled = bool(torque)

            # Read load (check for stall at rest)
            load_pct = self._read_load_pct(motor_id)
            if load_pct is not None:
                diag.load_pct = load_pct
                if abs(load_pct) > STALL_LOAD_THRESHOLD and diag.torque_enabled:
                    diag.issues.append(
                        f"High load at rest ({load_pct:.1f}%) - possible stall or obstruction"
                    )
                    diag.severity = max(diag.severity, "warning", key=["ok", "warning", "error"].index)

            diagnostics.append(asdict(diag))

        # Cross-motor checks
        summary = {
            "total_motors": len(MOTOR_IDS),
            "responding": sum(1 for d in diagnostics if d["responding"]),
            "errors": sum(1 for d in diagnostics if d["severity"] == "error"),
            "warnings": sum(1 for d in diagnostics if d["severity"] == "warning"),
        }

        # Check voltage consistency (big variation = possible wiring issue)
        if len(voltage_readings) >= 2:
            v_spread = max(voltage_readings) - min(voltage_readings)
            if v_spread > 1.5:
                summary["voltage_warning"] = (
                    f"Large voltage variation across motors ({min(voltage_readings):.1f}V - "
                    f"{max(voltage_readings):.1f}V) - check daisy-chain wiring"
                )

        if summary["responding"] == EXPECTED_MOTOR_COUNT and summary["errors"] == 0:
            summary["overall"] = "healthy"
        elif summary["responding"] == 0:
            summary["overall"] = "no_communication"
        elif summary["errors"] > 0:
            summary["overall"] = "errors_detected"
        else:
            summary["overall"] = "warnings"

        return {"summary": summary, "motors": diagnostics}

    # Preparation positions for calibrating each joint.
    # Moves other joints out of the way so the test joint has full ROM.
    # Values are in ticks (center=2048). Only joints that need repositioning
    # are listed; others remain at their current position.
    CALIBRATION_PREP = {
        'shoulder_pan': {
            # Raise arm high and extend upward so it can rotate freely
            'shoulder_lift': 2300,   # raised
            'elbow_flex': 2800,      # extended upward
            'wrist_flex': 2048,      # centered
        },
        'shoulder_lift': {
            # Fold elbow so arm is compact, won't hit table on downswing
            'shoulder_pan': 2048,    # centered
            'elbow_flex': 1200,      # folded back (arm compact)
            'wrist_flex': 2048,
        },
        'elbow_flex': {
            # Raise shoulder so elbow can sweep freely
            'shoulder_pan': 2048,
            'shoulder_lift': 2200,   # raised
            'wrist_flex': 2048,
        },
        'wrist_flex': {
            # Raise arm so wrist can sweep without hitting anything
            'shoulder_pan': 2048,
            'shoulder_lift': 2200,
            'elbow_flex': 2048,
        },
        'wrist_roll': {
            # Wrist roll is mostly unconstrained, just center other joints
            'shoulder_pan': 2048,
            'shoulder_lift': 2048,
        },
        'gripper': {
            # Gripper is unconstrained, no prep needed
        },
    }

    def calibrate_joint(self, joint: str, step_size: int = 25,
                        stall_threshold: float = 50.0) -> dict:
        """
        Calibrate a single joint's range of motion by sweeping in both directions.

        Automatically positions other joints out of the way before sweeping.
        Returns min/max positions and saves to calibration file.
        """
        self._check_connected()

        if joint not in MOTOR_IDS:
            raise ValueError(f"Unknown joint: {joint}. Valid: {MOTOR_NAMES}")

        motor_id = MOTOR_IDS[joint]

        # Step 1: Prepare other joints - move them out of the way
        prep = self.CALIBRATION_PREP.get(joint, {})
        if prep:
            logger.info(f"Preparing arm for {joint} calibration: moving {list(prep.keys())}")
            self.enable_torque()
            for prep_joint, prep_pos in prep.items():
                prep_id = MOTOR_IDS[prep_joint]
                self.packet_handler.write2ByteTxRx(
                    self.port, prep_id, ADDR_GOAL_POSITION, prep_pos
                )
                time.sleep(0.05)
            # Wait for preparation moves to complete
            time.sleep(1.5)

        # Step 2: Enable torque on test joint and read start position
        self.packet_handler.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, 1)
        time.sleep(0.05)

        start_pos, _ = self._read_with_retry(motor_id, ADDR_PRESENT_POSITION)

        def read_position_filtered(mid):
            """Read position, retrying on zero/invalid values."""
            for _ in range(5):
                val, ok = self._read_with_retry(mid, ADDR_PRESENT_POSITION)
                if ok and 10 < val < 4086:
                    return val
                time.sleep(0.01)
            return val  # return last attempt even if suspect

        def read_load_averaged(mid, samples=3):
            """Read load averaged over multiple samples to filter noise."""
            readings = []
            for _ in range(samples):
                load = self._read_load_pct(mid)
                if load is not None:
                    readings.append(abs(load))
                time.sleep(0.01)
            if not readings:
                return 0.0
            # Use median to reject outliers
            readings.sort()
            return readings[len(readings) // 2]

        def sweep(direction):
            target = start_pos
            stall_count = 0
            last_pos = start_pos

            for _ in range(160):
                target += direction * step_size
                if target < 50 or target > 4046:
                    return target - direction * step_size

                self.packet_handler.write2ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, target)
                time.sleep(0.12)

                actual = read_position_filtered(motor_id)
                load = read_load_averaged(motor_id)
                moved = abs(actual - last_pos)
                error = abs(actual - target)

                # Stall detection: high load OR motor not following commands
                is_stalled = load > stall_threshold or (error > 60 and moved < 5)

                if is_stalled:
                    stall_count += 1
                    if stall_count >= 4:
                        # Back off from stall point
                        backoff = target - direction * step_size * 4
                        self.packet_handler.write2ByteTxRx(
                            self.port, motor_id, ADDR_GOAL_POSITION, backoff
                        )
                        time.sleep(0.3)
                        return backoff
                else:
                    stall_count = 0
                last_pos = actual

            return target

        # Step 3: Sweep positive direction
        pos_limit = sweep(+1)

        # Return to start
        self.packet_handler.write2ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, start_pos)
        time.sleep(0.8)

        # Step 4: Sweep negative direction
        neg_limit = sweep(-1)

        # Return to start
        self.packet_handler.write2ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, start_pos)
        time.sleep(0.5)

        # Ensure min < max
        if neg_limit > pos_limit:
            neg_limit, pos_limit = pos_limit, neg_limit

        # Add safety margins
        safe_min = neg_limit + 50
        safe_max = pos_limit - 50

        # Update limits
        self.position_limits[joint] = (safe_min, safe_max)

        # Save calibration
        cal_data = {
            "position_limits": self.position_limits,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw_ranges": {
                joint: {
                    "min": neg_limit,
                    "max": pos_limit,
                    "safe_min": safe_min,
                    "safe_max": safe_max,
                    "range_ticks": pos_limit - neg_limit,
                    "range_degrees": round((pos_limit - neg_limit) * 360.0 / TICKS_PER_REV, 1),
                }
            },
        }

        # Merge with existing calibration
        if CALIBRATION_FILE.exists():
            try:
                with open(CALIBRATION_FILE) as f:
                    existing = json.load(f)
                existing["position_limits"].update(self.position_limits)
                if "raw_ranges" in existing:
                    existing["raw_ranges"].update(cal_data["raw_ranges"])
                else:
                    existing["raw_ranges"] = cal_data["raw_ranges"]
                existing["calibrated_at"] = cal_data["calibrated_at"]
                cal_data = existing
            except Exception:
                pass

        self._save_calibration(cal_data)

        return {
            "joint": joint,
            "min_ticks": neg_limit,
            "max_ticks": pos_limit,
            "safe_min": safe_min,
            "safe_max": safe_max,
            "range_ticks": pos_limit - neg_limit,
            "range_degrees": round((pos_limit - neg_limit) * 360.0 / TICKS_PER_REV, 1),
            "prep_positions": prep,
        }

    def enable_torque(self, motors: Optional[list[str]] = None):
        """Enable torque on specified motors (or all if None)."""
        self._check_connected()
        if self._mock:
            return

        if motors is None:
            motors = MOTOR_NAMES

        for name in motors:
            motor_id = MOTOR_IDS[name]
            self.packet_handler.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, 1)
            time.sleep(0.01)

    def disable_torque(self, motors: Optional[list[str]] = None):
        """Disable torque on specified motors (or all if None)."""
        self._check_connected()
        if self._mock:
            return

        if motors is None:
            motors = MOTOR_NAMES

        for name in motors:
            motor_id = MOTOR_IDS[name]
            self.packet_handler.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, 0)
            time.sleep(0.01)

    def move_joint(self, joint: str, position: int, enable_torque: bool = True) -> dict:
        """Move a single joint to a position (in ticks, 0-4096, center=2048)."""
        self._check_connected()

        if joint not in MOTOR_IDS:
            raise ValueError(f"Unknown joint: {joint}. Valid: {MOTOR_NAMES}")

        # Clamp to limits
        min_pos, max_pos = self.position_limits[joint]
        clamped = max(min_pos, min(max_pos, position))
        was_clamped = clamped != position

        if self._mock:
            return {"joint": joint, "target": clamped, "mock": True}

        motor_id = MOTOR_IDS[joint]

        if enable_torque:
            self.packet_handler.write1ByteTxRx(self.port, motor_id, ADDR_TORQUE_ENABLE, 1)
            time.sleep(0.01)

        self.packet_handler.write2ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, clamped)

        result = {"joint": joint, "target": clamped}
        if was_clamped:
            result["clamped_from"] = position
            result["limit"] = (min_pos, max_pos)
        return result

    def move_joints(self, positions: dict[str, int], enable_torque: bool = True) -> dict:
        """Move multiple joints to specified positions (ticks)."""
        self._check_connected()

        if enable_torque:
            self.enable_torque(list(positions.keys()))

        results = {}
        for joint, position in positions.items():
            result = self.move_joint(joint, position, enable_torque=False)
            results[joint] = result['target']

        return {"targets": results}

    def move_joints_degrees(self, positions: dict[str, float], enable_torque: bool = True) -> dict:
        """Move joints using degrees from center position."""
        ticks_positions = {joint: degrees_to_ticks(deg) for joint, deg in positions.items()}
        return self.move_joints(ticks_positions, enable_torque)

    def gripper_open(self, width: int = 2300) -> dict:
        """Open the gripper."""
        return self.move_joint('gripper', width)

    def gripper_close(self, width: int = 1700) -> dict:
        """Close the gripper."""
        return self.move_joint('gripper', width)

    def home(self) -> dict:
        """Move all joints to center position."""
        positions = {name: CENTER_POSITION for name in MOTOR_NAMES}
        return self.move_joints(positions)

    def wave(self) -> dict:
        """Perform a friendly wave gesture."""
        self._check_connected()

        state = self.get_state()
        home = state.positions.copy()
        self.enable_torque()

        # Extend elbow if folded
        if home['elbow_flex'] > 2500:
            self.move_joint('elbow_flex', 2048)
            time.sleep(0.8)
            home['elbow_flex'] = 2048

        # Raise shoulder
        self.move_joint('shoulder_lift', home['shoulder_lift'] + 300)
        time.sleep(0.5)

        # Wave shoulder pan
        for _ in range(3):
            self.move_joint('shoulder_pan', home['shoulder_pan'] + 400)
            time.sleep(0.35)
            self.move_joint('shoulder_pan', home['shoulder_pan'] - 400)
            time.sleep(0.35)

        # Return
        self.move_joint('shoulder_pan', home['shoulder_pan'])
        self.move_joint('shoulder_lift', home['shoulder_lift'])
        time.sleep(0.5)

        return {"gesture": "wave", "completed": True}

    def stop(self):
        """Stop all motion by disabling torque."""
        self.disable_torque()
        return {"stopped": True}


# Singleton controller instance
_controller: Optional[SO100Controller] = None


def get_controller() -> SO100Controller:
    """Get the singleton controller instance."""
    global _controller
    if _controller is None:
        _controller = SO100Controller()
    return _controller
