"""RealSense camera controller.

Supports Intel RealSense D405 (short-range, gripper-mount) and D435 (scene-level).

Two backends:
1. pyrealsense2 (preferred): Full SDK with aligned frames, post-processing filters,
   factory intrinsics, and proper depth scale. Requires building from source on aarch64.
2. V4L2 (fallback): Raw UVC access via v4l2-ctl + OpenCV. No SDK needed. Uses nominal
   intrinsics and device-specific depth scale. Color and depth are NOT pixel-aligned
   (hardware offset ~25mm on D435).

The backend is selected automatically: tries pyrealsense2, falls back to V4L2.
"""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MOCK_MODE = os.environ.get("REALSENSE_MOCK", "0") == "1"

logger = logging.getLogger(__name__)

# Intel RealSense USB product IDs
RS_PRODUCTS = {
    "0b5b": "D405",
    "0b07": "D435",
    "0b3a": "D435i",
    "0b64": "D405",
}
INTEL_VID = "8086"

# Nominal depth intrinsics for common resolutions (approximate).
# Without pyrealsense2, we can't read factory calibration from EEPROM.
# These are typical values; extrinsic calibration absorbs small errors.
NOMINAL_INTRINSICS = {
    "D435": {
        (848, 480): {"fx": 425.0, "fy": 425.0, "ppx": 424.0, "ppy": 240.0},
        (640, 480): {"fx": 382.0, "fy": 382.0, "ppx": 320.0, "ppy": 240.0},
        (1280, 720): {"fx": 638.0, "fy": 638.0, "ppx": 640.0, "ppy": 360.0},
    },
    "D435i": {
        (848, 480): {"fx": 425.0, "fy": 425.0, "ppx": 424.0, "ppy": 240.0},
        (640, 480): {"fx": 382.0, "fy": 382.0, "ppx": 320.0, "ppy": 240.0},
    },
    "D405": {
        (848, 480): {"fx": 425.0, "fy": 425.0, "ppx": 424.0, "ppy": 240.0},
        (640, 480): {"fx": 382.0, "fy": 382.0, "ppx": 320.0, "ppy": 240.0},
    },
}

# Default depth scale (meters per raw Z16 count)
DEFAULT_DEPTH_SCALE = {
    "D435": 0.001,    # 1mm per count
    "D435i": 0.001,
    "D405": 0.0001,   # 0.1mm per count
}


@dataclass
class CameraInfo:
    """Info about a connected RealSense device."""
    serial: str
    name: str
    firmware: str
    product_line: str
    usb_type: str


@dataclass
class RSConfig:
    """RealSense pipeline configuration."""
    width: int = 848
    height: int = 480
    fps: int = 30
    enable_color: bool = True
    enable_depth: bool = True
    align_to_color: bool = True
    # Post-processing filters (pyrealsense2 only)
    decimation_magnitude: int = 1
    spatial_alpha: float = 0.5
    spatial_delta: int = 20
    spatial_iterations: int = 2
    temporal_alpha: float = 0.4
    temporal_delta: int = 20
    hole_filling: int = 1


def _v4l2_discover() -> list:
    """Find all connected RealSense cameras via V4L2.

    Returns list of dicts with model, usb_path, and video_nodes.
    """
    # Get device groupings from v4l2-ctl
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
    except Exception:
        return []

    # Parse output: device name + indented /dev/video paths
    devices = []
    current_name = None
    current_nodes = []

    for line in result.stdout.splitlines():
        if line and not line.startswith("\t"):
            # Save previous device
            if current_name and current_nodes:
                devices.append((current_name, current_nodes))
            current_name = line.strip().rstrip(":")
            current_nodes = []
        elif line.strip().startswith("/dev/video"):
            current_nodes.append(line.strip())

    if current_name and current_nodes:
        devices.append((current_name, current_nodes))

    # Filter to RealSense devices and identify stream types
    rs_devices = []
    for name, nodes in devices:
        if "RealSense" not in name and "Intel" not in name:
            continue

        # Determine model via udevadm on first video node (gives USB PID)
        model = "Unknown"
        first_video = nodes[0] if nodes else None
        if first_video:
            try:
                r = subprocess.run(
                    ["udevadm", "info", first_video],
                    capture_output=True, text=True, timeout=3)
                for uline in r.stdout.splitlines():
                    if "ID_MODEL_ID=" in uline:
                        pid = uline.split("=", 1)[1].strip()
                        model = RS_PRODUCTS.get(pid, "Unknown")
                        break
            except Exception:
                pass
        # Fallback: check device name string
        if model == "Unknown":
            for pid, m in RS_PRODUCTS.items():
                if m.lower() in name.lower():
                    model = m
                    break

        # Probe each node for stream type
        video_nodes = {}
        for node in nodes:
            try:
                r = subprocess.run(
                    ["v4l2-ctl", "-d", node, "--list-formats"],
                    capture_output=True, text=True, timeout=3)
                output = r.stdout
                if "Z16" in output:
                    video_nodes["depth"] = node
                elif "GREY" in output or "Y8" in output or "Y16" in output:
                    if "ir_left" not in video_nodes:
                        video_nodes["ir_left"] = node
                    else:
                        video_nodes["ir_right"] = node
                elif "YUYV" in output or "MJPG" in output:
                    # D405 has no RGB sensor — its YUYV node is IR, not color.
                    # D435/D435i have a dedicated RGB sensor on a separate YUYV node.
                    if model == "D405":
                        if "ir_right" not in video_nodes:
                            video_nodes["ir_right"] = node
                    else:
                        if "color" not in video_nodes:
                            video_nodes["color"] = node
                        elif "ir_right" not in video_nodes:
                            video_nodes["ir_right"] = node
            except Exception:
                continue

        if video_nodes:
            rs_devices.append({
                "name": name,
                "model": model,
                "video_nodes": video_nodes,
            })

    return rs_devices


class RealSenseController:
    """Controller for Intel RealSense D405/D435 cameras.

    Automatically selects pyrealsense2 or V4L2 backend.
    """

    def __init__(self, config: Optional[RSConfig] = None):
        self.config = config or RSConfig()
        self._connected = False
        self._backend = None  # "rs2" or "v4l2"
        self._camera_info: Optional[CameraInfo] = None

        # Cached frames
        self._color_frame: Optional[np.ndarray] = None  # (H, W, 3) uint8 BGR
        self._depth_frame: Optional[np.ndarray] = None  # (H, W) uint16 raw
        self._depth_scale: float = 0.001  # meters per count
        self._capture_time: float = 0.0

        # Intrinsics for deprojection: {fx, fy, ppx, ppy}
        self._intrinsics: Optional[dict] = None

        # pyrealsense2 objects (rs2 backend only)
        self._pipeline = None
        self._profile = None
        self._align = None
        self._rs2_depth_intrinsics = None
        self._filters = []

        # V4L2 objects
        self._v4l2_color_cap = None
        self._v4l2_depth_device = None
        self._v4l2_model = None
        self._depth_resolution = None  # (w, h) for depth capture

        # Robot calibration (static camera: camera→robot direct)
        self._calibration_matrix: Optional[np.ndarray] = None  # 4x4 SE(3)
        self._calibration_path = Path("/home/doug/panda-mcp/calibration/realsense_calibration.npz")

        # Wrist calibration (wrist camera: camera→EE, needs FK to get robot frame)
        self._wrist_calibration: Optional[np.ndarray] = None  # 4x4 T_ee_camera
        self._wrist_calibration_path = Path("/home/doug/panda-mcp/calibration/wrist_calibration.npz")
        self._ee_pose: Optional[np.ndarray] = None  # 4x4 T_base_ee, set by caller

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def has_frames(self) -> bool:
        return self._depth_frame is not None

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def list_devices(self) -> dict:
        """List all connected RealSense devices."""
        if MOCK_MODE:
            return {"devices": [{"serial": "mock-001", "name": "Mock D405",
                                 "usb_type": "3.2"}]}

        # Try pyrealsense2 first
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = []
            for dev in ctx.query_devices():
                devices.append({
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "name": dev.get_info(rs.camera_info.name),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                    "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
                })
            return {"devices": devices, "count": len(devices), "backend": "rs2"}
        except ImportError:
            pass

        # Fall back to V4L2
        rs_devices = _v4l2_discover()
        devices = []
        for d in rs_devices:
            devices.append({
                "name": d["name"],
                "model": d["model"],
                "video_nodes": d["video_nodes"],
            })
        return {"devices": devices, "count": len(devices), "backend": "v4l2"}

    def connect(self, serial: Optional[str] = None,
                model: Optional[str] = None,
                width: Optional[int] = None,
                height: Optional[int] = None) -> dict:
        """Connect to a RealSense device.

        Args:
            serial: Device serial number (pyrealsense2 only).
            model: Model name to connect to ("D435", "D405"). Useful with V4L2
                   when multiple cameras are connected.
            width: Override capture width.
            height: Override capture height.

        Tries pyrealsense2 first, falls back to V4L2.
        """
        if width:
            self.config.width = width
        if height:
            self.config.height = height

        if MOCK_MODE:
            return self._connect_mock()

        # Try pyrealsense2
        try:
            import pyrealsense2 as rs
            result = self._connect_rs2(rs, serial, model=model)
            if result.get("success"):
                return result
            logger.warning(f"pyrealsense2 connect failed: {result.get('error')}")
        except ImportError:
            logger.info("pyrealsense2 not available, using V4L2 backend")

        # Fall back to V4L2
        return self._connect_v4l2(serial, model=model)

    def _connect_mock(self) -> dict:
        self._connected = True
        self._backend = "mock"
        self._camera_info = CameraInfo(
            serial="mock-001", name="Mock D405",
            firmware="0.0.0", product_line="D400", usb_type="3.2")
        self._color_frame = np.zeros(
            (self.config.height, self.config.width, 3), dtype=np.uint8)
        self._depth_frame = np.full(
            (self.config.height, self.config.width), 500, dtype=np.uint16)
        self._depth_scale = 0.001
        self._intrinsics = {
            "fx": self.config.width * 0.5, "fy": self.config.width * 0.5,
            "ppx": self.config.width * 0.5, "ppy": self.config.height * 0.5}
        return {"success": True, "mock": True, "device": "Mock D405"}

    def _connect_rs2(self, rs, serial: Optional[str],
                     model: Optional[str] = None) -> dict:
        """Connect via pyrealsense2 SDK."""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return {"success": False, "error": "No RealSense devices found"}

        target_device = None
        for dev in devices:
            dev_serial = dev.get_info(rs.camera_info.serial_number)
            dev_name = dev.get_info(rs.camera_info.name)
            if serial and dev_serial != serial:
                continue
            if model and model.upper() not in dev_name.upper():
                continue
            target_device = dev
            break

        if target_device is None:
            available = [f"{d.get_info(rs.camera_info.name)} ({d.get_info(rs.camera_info.serial_number)})"
                         for d in devices]
            return {"success": False,
                    "error": f"Device not found (serial={serial}, model={model}). Available: {available}"}

        self._camera_info = CameraInfo(
            serial=target_device.get_info(rs.camera_info.serial_number),
            name=target_device.get_info(rs.camera_info.name),
            firmware=target_device.get_info(rs.camera_info.firmware_version),
            product_line=target_device.get_info(rs.camera_info.product_line),
            usb_type=target_device.get_info(rs.camera_info.usb_type_descriptor))

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self._camera_info.serial)

        if self.config.enable_depth:
            config.enable_stream(rs.stream.depth,
                                 self.config.width, self.config.height,
                                 rs.format.z16, self.config.fps)
        if self.config.enable_color:
            config.enable_stream(rs.stream.color,
                                 self.config.width, self.config.height,
                                 rs.format.bgr8, self.config.fps)

        self._profile = self._pipeline.start(config)
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        if self.config.align_to_color and self.config.enable_color:
            self._align = rs.align(rs.stream.color)

        self._setup_rs2_filters(rs)

        # Warm up
        for _ in range(30):
            self._pipeline.wait_for_frames()

        self._backend = "rs2"
        self._connected = True
        self._load_calibration()

        logger.info(f"Connected via rs2: {self._camera_info.name} "
                    f"({self._camera_info.serial})")
        return {
            "success": True,
            "backend": "rs2",
            "device": self._camera_info.name,
            "serial": self._camera_info.serial,
            "firmware": self._camera_info.firmware,
            "resolution": f"{self.config.width}x{self.config.height}",
            "depth_scale_m": self._depth_scale,
            "calibrated": self._calibration_matrix is not None,
        }

    def _connect_v4l2(self, serial: Optional[str],
                      model: Optional[str] = None) -> dict:
        """Connect via V4L2 (no SDK)."""
        rs_devices = _v4l2_discover()
        if not rs_devices:
            return {"success": False,
                    "error": "No RealSense cameras found via V4L2"}

        # Pick device by model name, or first available
        device = None
        if model:
            for d in rs_devices:
                if d["model"].upper() == model.upper():
                    device = d
                    break
            if device is None:
                available = [d["model"] for d in rs_devices]
                return {"success": False,
                        "error": f"Model {model} not found. Available: {available}"}
        else:
            device = rs_devices[0]
        nodes = device["video_nodes"]

        if "depth" not in nodes:
            return {"success": False,
                    "error": f"No depth stream found. Nodes: {nodes}"}

        model = device.get("model", "Unknown")
        self._v4l2_depth_device = nodes["depth"]
        self._v4l2_model = model

        # Open color capture if available
        color_node = nodes.get("color")
        if color_node:
            self._v4l2_color_cap = cv2.VideoCapture(color_node, cv2.CAP_V4L2)
            self._v4l2_color_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._v4l2_color_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            if not self._v4l2_color_cap.isOpened():
                logger.warning(f"Failed to open color at {color_node}")
                self._v4l2_color_cap = None
            else:
                # Read back actual resolution (camera may not support requested)
                actual_w = int(self._v4l2_color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._v4l2_color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (actual_w, actual_h) != (self.config.width, self.config.height):
                    logger.info(f"Color resolution {actual_w}x{actual_h} "
                                f"(requested {self.config.width}x{self.config.height})")
                    # Use actual color resolution for depth too, so pixels align
                    self.config.width = actual_w
                    self.config.height = actual_h

        # Set depth resolution to match config (potentially adjusted for color)
        self._depth_resolution = (self.config.width, self.config.height)

        # Depth scale
        self._depth_scale = DEFAULT_DEPTH_SCALE.get(model, 0.001)

        # Nominal intrinsics (using final resolution)
        key = (self.config.width, self.config.height)
        model_intrinsics = NOMINAL_INTRINSICS.get(model, {})
        self._intrinsics = model_intrinsics.get(key)
        if self._intrinsics is None:
            # Estimate from resolution
            self._intrinsics = {
                "fx": self.config.width * 0.5,
                "fy": self.config.width * 0.5,
                "ppx": self.config.width * 0.5,
                "ppy": self.config.height * 0.5,
            }
            logger.warning(f"No intrinsics for {model} at {key}, using estimate")

        self._camera_info = CameraInfo(
            serial="v4l2",
            name=device["name"],
            firmware="unknown",
            product_line=model,
            usb_type="unknown")

        self._backend = "v4l2"
        self._connected = True
        self._load_calibration()

        logger.info(f"Connected via V4L2: {device['name']} "
                    f"(depth={self._v4l2_depth_device}, "
                    f"color={color_node or 'none'})")
        return {
            "success": True,
            "backend": "v4l2",
            "device": device["name"],
            "model": model,
            "video_nodes": nodes,
            "depth_scale_m": self._depth_scale,
            "intrinsics": self._intrinsics,
            "resolution": f"{self.config.width}x{self.config.height}",
            "calibrated": self._calibration_matrix is not None,
            "note": "Using nominal intrinsics. Build pyrealsense2 for factory calibration.",
        }

    def _setup_rs2_filters(self, rs):
        """Configure depth post-processing filters (rs2 only)."""
        self._filters = []
        if self.config.decimation_magnitude > 1:
            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude,
                           self.config.decimation_magnitude)
            self._filters.append(dec)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_smooth_alpha, self.config.spatial_alpha)
        spatial.set_option(rs.option.filter_smooth_delta, self.config.spatial_delta)
        spatial.set_option(rs.option.holes_fill, self.config.spatial_iterations)
        self._filters.append(spatial)
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, self.config.temporal_alpha)
        temporal.set_option(rs.option.filter_smooth_delta, self.config.temporal_delta)
        self._filters.append(temporal)
        hole = rs.hole_filling_filter()
        hole.set_option(rs.option.holes_fill, self.config.hole_filling)
        self._filters.append(hole)

    def disconnect(self) -> dict:
        """Stop pipeline and release device."""
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._profile = None
        self._align = None
        if self._v4l2_color_cap is not None:
            self._v4l2_color_cap.release()
            self._v4l2_color_cap = None
        self._connected = False
        self._backend = None
        self._color_frame = None
        self._depth_frame = None
        logger.info("RealSense disconnected")
        return {"success": True}

    def capture(self) -> dict:
        """Capture color + depth frames.

        Returns dict with frame stats.
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if MOCK_MODE:
            self._capture_time = time.time()
            return {"success": True, "mock": True}

        if self._backend == "rs2":
            return self._capture_rs2()
        elif self._backend == "v4l2":
            return self._capture_v4l2()
        else:
            return {"success": False, "error": f"Unknown backend: {self._backend}"}

    def _capture_rs2(self) -> dict:
        """Capture via pyrealsense2."""
        try:
            import pyrealsense2 as rs
            frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            if self._align:
                frames = self._align.process(frames)

            depth_frame = frames.get_depth_frame()
            if depth_frame:
                for f in self._filters:
                    depth_frame = f.process(depth_frame)
                self._depth_frame = np.asanyarray(depth_frame.get_data())
                self._rs2_depth_intrinsics = (
                    depth_frame.profile.as_video_stream_profile().intrinsics)
                # Also store as dict for consistent API
                intr = self._rs2_depth_intrinsics
                self._intrinsics = {
                    "fx": intr.fx, "fy": intr.fy,
                    "ppx": intr.ppx, "ppy": intr.ppy}

            color_frame = frames.get_color_frame()
            if color_frame:
                self._color_frame = np.asanyarray(color_frame.get_data())

            self._capture_time = time.time()
            return self._build_capture_result()

        except Exception as e:
            logger.exception("rs2 capture failed")
            return {"success": False, "error": str(e)}

    def _capture_v4l2(self) -> dict:
        """Capture via V4L2 (subprocess for depth, OpenCV for color)."""
        try:
            # Color
            if self._v4l2_color_cap is not None:
                ret, frame = self._v4l2_color_cap.read()
                if ret:
                    self._color_frame = frame
                else:
                    logger.warning("V4L2 color capture failed")

            # Depth via v4l2-ctl (use same resolution as color if available)
            if self._v4l2_depth_device:
                w, h = self._depth_resolution or (self.config.width, self.config.height)
                frame_bytes = w * h * 2
                tmp = "/tmp/rs_depth_capture.raw"

                result = subprocess.run([
                    "v4l2-ctl", "-d", self._v4l2_depth_device,
                    f"--set-fmt-video=width={w},height={h},pixelformat=Z16 ",
                    "--stream-mmap", "--stream-count=3",
                    f"--stream-to={tmp}",
                ], capture_output=True, text=True, timeout=5)

                if os.path.exists(tmp):
                    raw_size = os.path.getsize(tmp)
                    n_frames = raw_size // frame_bytes
                    if n_frames > 0:
                        with open(tmp, "rb") as f:
                            # Read last frame (most recent)
                            f.seek((n_frames - 1) * frame_bytes)
                            data = f.read(frame_bytes)
                        self._depth_frame = np.frombuffer(
                            data, dtype=np.uint16).reshape(h, w)
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

            self._capture_time = time.time()
            return self._build_capture_result()

        except subprocess.TimeoutExpired:
            logger.error("V4L2 depth capture timed out")
            return {"success": False, "error": "Depth capture timed out"}
        except Exception as e:
            logger.exception("V4L2 capture failed")
            return {"success": False, "error": str(e)}

    def _build_capture_result(self) -> dict:
        """Build standardized capture result from cached frames."""
        result = {"success": True, "timestamp": self._capture_time,
                  "backend": self._backend}

        if self._depth_frame is not None:
            valid = (self._depth_frame > 0) & (self._depth_frame < 65535)
            coverage = valid.sum() / valid.size
            result.update({
                "depth_shape": list(self._depth_frame.shape),
                "depth_min_mm": round(
                    float(self._depth_frame[valid].min() * self._depth_scale * 1000), 1
                ) if valid.any() else 0,
                "depth_max_mm": round(
                    float(self._depth_frame[valid].max() * self._depth_scale * 1000), 1
                ) if valid.any() else 0,
                "depth_coverage": round(float(coverage), 3),
            })

        if self._color_frame is not None:
            result["color_shape"] = list(self._color_frame.shape)

        return result

    def get_color_image(self) -> Optional[np.ndarray]:
        """Get last captured color frame as BGR numpy array."""
        return self._color_frame

    def get_depth_image(self) -> Optional[np.ndarray]:
        """Get last captured depth frame as uint16 raw array."""
        return self._depth_frame

    def get_depth_meters(self) -> Optional[np.ndarray]:
        """Get depth frame converted to float32 meters."""
        if self._depth_frame is None:
            return None
        depth_m = self._depth_frame.astype(np.float32) * self._depth_scale
        depth_m[self._depth_frame == 0] = 0
        depth_m[self._depth_frame >= 65535] = 0
        return depth_m

    def _deproject(self, px: int, py: int, depth_m: float) -> list:
        """Deproject pixel to 3D point in camera frame.

        Standard pinhole model: identical to rs2_deproject_pixel_to_point.
        """
        if self._intrinsics is None:
            return [0, 0, depth_m]
        fx = self._intrinsics["fx"]
        fy = self._intrinsics["fy"]
        ppx = self._intrinsics["ppx"]
        ppy = self._intrinsics["ppy"]
        x = (px - ppx) * depth_m / fx
        y = (py - ppy) * depth_m / fy
        return [x, y, depth_m]

    def get_depth_at(self, pixel_x: int, pixel_y: int, radius: int = 5) -> dict:
        """Get depth and 3D position at a pixel coordinate.

        Works with both rs2 and V4L2 backends.
        """
        if self._depth_frame is None:
            return {"valid": False, "error": "No depth frame. Call capture() first."}

        h, w = self._depth_frame.shape
        if not (0 <= pixel_x < w and 0 <= pixel_y < h):
            return {"valid": False,
                    "error": f"Pixel ({pixel_x},{pixel_y}) out of bounds ({w}x{h})"}

        if radius > 0:
            y0, y1 = max(0, pixel_y - radius), min(h, pixel_y + radius + 1)
            x0, x1 = max(0, pixel_x - radius), min(w, pixel_x + radius + 1)
            patch = self._depth_frame[y0:y1, x0:x1]
            valid = patch[(patch > 0) & (patch < 65535)]
            if len(valid) == 0:
                return {"valid": False, "error": "No valid depth in patch",
                        "pixel": [pixel_x, pixel_y]}
            depth_raw = int(np.median(valid))
        else:
            depth_raw = int(self._depth_frame[pixel_y, pixel_x])
            if depth_raw == 0 or depth_raw >= 65535:
                return {"valid": False, "error": "No depth at pixel",
                        "pixel": [pixel_x, pixel_y]}

        depth_m = depth_raw * self._depth_scale
        point_3d = self._deproject(pixel_x, pixel_y, depth_m)

        return {
            "valid": True,
            "pixel": [pixel_x, pixel_y],
            "depth_mm": round(depth_m * 1000, 1),
            "position_m": {
                "x": round(point_3d[0], 4),
                "y": round(point_3d[1], 4),
                "z": round(point_3d[2], 4),
            },
        }

    def get_robot_coords_at(self, pixel_x: int, pixel_y: int,
                            radius: int = 5) -> dict:
        """Get 3D position in both camera and robot frame.

        For wrist cameras: uses T_base_ee @ T_ee_camera @ p_camera.
        Requires set_ee_pose() to be called first with current EE pose.

        For static cameras: uses direct camera→robot transform.
        """
        depth_result = self.get_depth_at(pixel_x, pixel_y, radius)
        if not depth_result.get("valid"):
            return depth_result

        pos = depth_result["position_m"]
        cam_point = np.array([pos["x"], pos["y"], pos["z"], 1.0])

        # Wrist camera path: T_base_ee @ T_ee_camera @ p_camera
        # Only use wrist calibration for D405 (wrist-mounted camera)
        is_wrist = (self._camera_info is not None
                    and "D405" in (self._camera_info.product_line or ""))
        if is_wrist and self._wrist_calibration is not None:
            if self._ee_pose is None:
                depth_result["robot_m"] = None
                depth_result["calibration_error"] = (
                    "Wrist camera needs EE pose. Call set_ee_pose() first.")
                return depth_result
            robot_point = self._ee_pose @ self._wrist_calibration @ cam_point
            depth_result["robot_m"] = {
                "x": round(float(robot_point[0]), 4),
                "y": round(float(robot_point[1]), 4),
                "z": round(float(robot_point[2]), 4),
            }
            depth_result["calibration_type"] = "wrist"
            return depth_result

        # Static camera path: direct transform
        if self._calibration_matrix is None:
            depth_result["robot_m"] = None
            depth_result["calibration_error"] = "No calibration. Run calibration first."
            return depth_result

        robot_point = self._calibration_matrix @ cam_point
        depth_result["robot_m"] = {
            "x": round(float(robot_point[0]), 4),
            "y": round(float(robot_point[1]), 4),
            "z": round(float(robot_point[2]), 4),
        }
        depth_result["calibration_type"] = "static"
        return depth_result

    def get_pointcloud(self, downsample: int = 4) -> dict:
        """Get 3D pointcloud from depth frame."""
        if self._depth_frame is None:
            return {"success": False, "error": "No depth frame"}
        if self._intrinsics is None:
            return {"success": False, "error": "No intrinsics"}

        h, w = self._depth_frame.shape
        fx = self._intrinsics["fx"]
        fy = self._intrinsics["fy"]
        ppx = self._intrinsics["ppx"]
        ppy = self._intrinsics["ppy"]

        # Vectorized deprojection
        ys, xs = np.mgrid[0:h:downsample, 0:w:downsample]
        depth_sub = self._depth_frame[::downsample, ::downsample].astype(np.float32)
        valid_mask = (depth_sub > 0) & (depth_sub < 65535)
        depth_m = depth_sub * self._depth_scale

        zs = depth_m[valid_mask]
        px = xs[valid_mask].astype(np.float32)
        py = ys[valid_mask].astype(np.float32)

        point_x = (px - ppx) * zs / fx
        point_y = (py - ppy) * zs / fy
        points = np.stack([point_x, point_y, zs], axis=-1)

        result = {
            "success": True,
            "num_points": len(points),
            "downsample": downsample,
        }
        if len(points) > 0:
            result["bounds"] = {
                "x": [round(float(points[:, 0].min()), 3),
                      round(float(points[:, 0].max()), 3)],
                "y": [round(float(points[:, 1].min()), 3),
                      round(float(points[:, 1].max()), 3)],
                "z": [round(float(points[:, 2].min()), 3),
                      round(float(points[:, 2].max()), 3)],
            }
        return result

    def save_scan(self, path: str = "/tmp/realsense_scan.npz") -> dict:
        """Save current frames as compressed NPZ."""
        if self._depth_frame is None and self._color_frame is None:
            return {"success": False, "error": "No frames. Call capture() first."}

        try:
            data = {"timestamp": np.array(self._capture_time)}
            if self._depth_frame is not None:
                data["depth"] = self._depth_frame
                data["depth_scale"] = np.array(self._depth_scale)
            if self._color_frame is not None:
                data["color"] = self._color_frame
            if self._intrinsics is not None:
                data["intrinsics"] = np.array([
                    self._intrinsics["ppx"], self._intrinsics["ppy"],
                    self._intrinsics["fx"], self._intrinsics["fy"]])
            if self._calibration_matrix is not None:
                data["calibration_matrix"] = self._calibration_matrix

            np.savez_compressed(path, **data)
            size = Path(path).stat().st_size
            logger.info(f"Saved scan to {path} ({size} bytes)")
            return {"success": True, "path": path, "size_bytes": size}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_calibration(self, matrix: np.ndarray,
                         path: Optional[str] = None) -> dict:
        """Save camera-to-robot calibration matrix (4x4 SE(3))."""
        save_path = Path(path) if path else self._calibration_path
        self._calibration_matrix = matrix
        np.savez_compressed(str(save_path), transform=matrix)
        logger.info(f"Calibration saved to {save_path}")
        return {"success": True, "path": str(save_path)}

    def _load_calibration(self):
        """Load calibration if file exists."""
        if self._calibration_path.exists():
            try:
                data = np.load(str(self._calibration_path))
                self._calibration_matrix = data["transform"]
                logger.info(f"Loaded static calibration from {self._calibration_path}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")
        if self._wrist_calibration_path.exists():
            try:
                data = np.load(str(self._wrist_calibration_path))
                self._wrist_calibration = data["T_ee_camera"]
                logger.info(f"Loaded wrist calibration from {self._wrist_calibration_path}")
            except Exception as e:
                logger.warning(f"Failed to load wrist calibration: {e}")

    def set_ee_pose(self, position, rpy):
        """Set current end-effector pose for wrist camera transforms.

        Args:
            position: [x, y, z] in meters
            rpy: [roll, pitch, yaw] in radians
        """
        cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
        cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
        cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = position
        self._ee_pose = T

    def _preprocess_clahe(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel of LAB for lighting normalization."""
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _detect_colors_lab(self, bgr_frame: np.ndarray, depth_band: np.ndarray,
                           target_colors: list[str], min_area: int,
                           kernel: np.ndarray, debug: bool = False) -> list[dict]:
        """Detect colored objects using LAB color space (lighting-invariant).

        A channel: green(0) - red(255), center=128
        B channel: blue(0) - yellow(255), center=128
        """
        import math

        lab_ranges = {
            "red":    {"a_min": 155, "a_max": 255, "b_min": 118, "b_max": 170, "l_min": 30},
            "green":  {"a_min": 0,   "a_max": 105, "b_min": 128, "b_max": 255, "l_min": 30},
            "blue":   {"a_min": 100, "a_max": 145, "b_min": 0,   "b_max": 105, "l_min": 30},
            "orange": {"a_min": 135, "a_max": 175, "b_min": 150, "b_max": 255, "l_min": 50},
        }

        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        objects = []

        for color_name in target_colors:
            if color_name not in lab_ranges:
                continue
            r = lab_ranges[color_name]
            mask_l = lab[:, :, 0] >= r["l_min"]
            mask_a = (lab[:, :, 1] >= r["a_min"]) & (lab[:, :, 1] <= r["a_max"])
            mask_b = (lab[:, :, 2] >= r["b_min"]) & (lab[:, :, 2] <= r["b_max"])
            mask = (mask_l & mask_a & mask_b).astype(np.uint8) * 255
            mask = mask & depth_band
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if debug:
                debug_dir = Path("/tmp/detect_debug")
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / f"mask_{color_name}_lab.jpg"), mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bx, by, bw, bh = cv2.boundingRect(cnt)

                # Grasp yaw from minAreaRect
                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), angle = rect
                if rw > rh:
                    grasp_deg = angle + 90
                else:
                    grasp_deg = angle
                yaw = math.radians(grasp_deg)
                while yaw > math.pi / 2:
                    yaw -= math.pi
                while yaw < -math.pi / 2:
                    yaw += math.pi

                objects.append({
                    "color": color_name,
                    "pixel": [cx, cy],
                    "area": int(area),
                    "bbox": [bx, by, bw, bh],
                    "yaw_rad": round(yaw, 3),
                    "_cx": cx, "_cy": cy, "_bw": bw, "_bh": bh,
                })

        return objects

    def _detect_hsv(self, bgr_frame: np.ndarray, depth_band: np.ndarray,
                    target_colors: list[str], min_area: int,
                    kernel: np.ndarray, debug: bool = False,
                    method_label: str = "hsv") -> list[dict]:
        """Run HSV color detection. Returns raw detections (no depth/robot coords)."""
        import math

        color_ranges = {
            "red": [((0, 40, 40), (10, 255, 255)),
                    ((160, 40, 40), (180, 255, 255))],
            "green": [((35, 25, 25), (85, 255, 255))],
            "blue": [((80, 25, 25), (130, 255, 255))],
            "orange": [((10, 40, 40), (25, 255, 255))],
        }
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        objects = []

        for color_name in target_colors:
            if color_name not in color_ranges:
                continue
            ranges = color_ranges[color_name]
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = mask & depth_band
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if debug:
                debug_dir = Path("/tmp/detect_debug")
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / f"mask_{color_name}_{method_label}.jpg"), mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bx, by, bw, bh = cv2.boundingRect(cnt)

                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), angle = rect
                if rw > rh:
                    grasp_deg = angle + 90
                else:
                    grasp_deg = angle
                yaw = math.radians(grasp_deg)
                while yaw > math.pi / 2:
                    yaw -= math.pi
                while yaw < -math.pi / 2:
                    yaw += math.pi

                objects.append({
                    "color": color_name,
                    "pixel": [cx, cy],
                    "area": int(area),
                    "bbox": [bx, by, bw, bh],
                    "yaw_rad": round(yaw, 3),
                    "_cx": cx, "_cy": cy, "_bw": bw, "_bh": bh,
                })

        return objects

    def _resolve_detections_3d(self, raw_objects: list[dict],
                               workspace: Optional[dict],
                               is_wrist: bool) -> list[dict]:
        """Resolve raw 2D detections to 3D robot-frame positions via depth."""
        ws = workspace or {
            "x_min": 0.1, "x_max": 0.7,
            "y_min": -0.45, "y_max": 0.45,
            "z_min": -0.01, "z_max": 0.15,
        }
        resolved = []
        for obj in raw_objects:
            cx, cy = obj["_cx"], obj["_cy"]
            bw, bh = obj["_bw"], obj["_bh"]
            radius = max(3, min(bw, bh) // 4)
            depth_result = self.get_depth_at(cx, cy, radius=radius)
            if not depth_result.get("valid"):
                continue

            pos = depth_result["position_m"]
            cam_point = np.array([pos["x"], pos["y"], pos["z"], 1.0])
            if is_wrist and self._wrist_calibration is not None and self._ee_pose is not None:
                robot_point = self._ee_pose @ self._wrist_calibration @ cam_point
            else:
                robot_point = self._calibration_matrix @ cam_point

            rx = float(robot_point[0])
            ry = float(robot_point[1])
            rz = float(robot_point[2])
            if not (ws["x_min"] <= rx <= ws["x_max"]
                    and ws["y_min"] <= ry <= ws["y_max"]
                    and ws["z_min"] <= rz <= ws["z_max"]):
                continue

            resolved.append({
                "color": obj["color"],
                "pixel": obj["pixel"],
                "area": obj["area"],
                "bbox": obj["bbox"],
                "depth_mm": depth_result["depth_mm"],
                "robot_x": round(rx, 4),
                "robot_y": round(ry, 4),
                "robot_z": round(rz, 4),
                "yaw_rad": obj["yaw_rad"],
            })
        return resolved

    def detect_objects(self, colors: Optional[list[str]] = None,
                       min_area: int = 500,
                       workspace: Optional[dict] = None,
                       method: str = "auto",
                       debug: bool = False) -> dict:
        """Detect colored objects in the color frame and return 3D robot positions.

        Args:
            colors: List of color names to detect. Default: all.
            min_area: Minimum contour area in pixels.
            workspace: Robot-frame bounds for filtering.
            method: Detection method - "hsv", "clahe", "lab", or "auto".
                    "auto" tries CLAHE+HSV first, then LAB, then raw HSV.
            debug: Save debug images to /tmp/detect_debug/.

        Returns:
            Dict with 'objects' list, each containing color, pixel, robot
            position, area, bounding box, and grasp yaw.
        """
        # Capture fresh frames
        cap = self.capture()
        if not cap.get("success"):
            return {"success": False, "error": f"Capture failed: {cap.get('error')}"}

        color_frame = self._color_frame
        depth_frame = self._depth_frame
        if color_frame is None:
            return {"success": False, "error": "No color frame"}

        # Save color frame for skill_logger fallback
        try:
            cv2.imwrite("/tmp/skill_frame.jpg", color_frame)
        except Exception:
            pass

        if depth_frame is None:
            return {"success": False, "error": "No depth frame"}
        is_wrist = (self._camera_info is not None
                    and "D405" in (self._camera_info.product_line or ""))
        if is_wrist:
            if self._wrist_calibration is None:
                return {"success": False,
                        "error": "No wrist calibration. Run wrist calibration first."}
            if self._ee_pose is None:
                return {"success": False,
                        "error": "Wrist camera needs EE pose. Call set_ee_pose() first."}
        elif self._calibration_matrix is None:
            return {"success": False,
                    "error": "No calibration. Run calibration first."}

        all_colors = ["red", "green", "blue", "orange"]
        target_colors = colors or all_colors
        kernel = np.ones((5, 5), np.uint8)

        # Depth-band mask: only consider pixels at table height (400-1000mm)
        depth_mm = depth_frame.astype(np.float32) * self._depth_scale * 1000
        depth_band = ((depth_mm > 400) & (depth_mm < 1000)).astype(np.uint8) * 255

        # Debug: save raw and CLAHE images
        if debug:
            debug_dir = Path("/tmp/detect_debug")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "raw.jpg"), color_frame)
            clahe_frame = self._preprocess_clahe(color_frame)
            cv2.imwrite(str(debug_dir / "clahe.jpg"), clahe_frame)

        # Run detection based on method
        method_used = method
        if method == "hsv":
            raw = self._detect_hsv(color_frame, depth_band, target_colors,
                                   min_area, kernel, debug)
        elif method == "clahe":
            clahe_frame = self._preprocess_clahe(color_frame)
            raw = self._detect_hsv(clahe_frame, depth_band, target_colors,
                                   min_area, kernel, debug, method_label="clahe")
        elif method == "lab":
            raw = self._detect_colors_lab(color_frame, depth_band, target_colors,
                                          min_area, kernel, debug)
        elif method == "auto":
            # Try CLAHE+HSV first
            clahe_frame = self._preprocess_clahe(color_frame)
            raw = self._detect_hsv(clahe_frame, depth_band, target_colors,
                                   min_area, kernel, debug, method_label="clahe")
            method_used = "clahe"
            if not raw:
                # Try LAB
                raw = self._detect_colors_lab(color_frame, depth_band, target_colors,
                                              min_area, kernel, debug)
                method_used = "lab"
            if not raw:
                # Fall back to raw HSV
                raw = self._detect_hsv(color_frame, depth_band, target_colors,
                                       min_area, kernel, debug)
                method_used = "hsv"
        else:
            return {"success": False, "error": f"Unknown method: {method}"}

        # Resolve 2D detections to 3D robot coords
        objects = self._resolve_detections_3d(raw, workspace, is_wrist)

        # Debug: save annotated image
        if debug:
            debug_dir = Path("/tmp/detect_debug")
            annotated = color_frame.copy()
            for obj in objects:
                bx, by, bw, bh = obj["bbox"]
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                label = f"{obj['color']} ({obj['robot_x']:.3f}, {obj['robot_y']:.3f})"
                cv2.putText(annotated, label, (bx, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(debug_dir / "annotated.jpg"), annotated)

        # Sort by area descending
        objects.sort(key=lambda o: -o["area"])

        return {
            "success": True,
            "count": len(objects),
            "objects": objects,
            "method_used": method_used,
            "frame_shape": list(color_frame.shape),
        }

    def ground_object(self, query: str,
                      server_url: str = "http://spark:8090") -> dict:
        """Find an object using natural language via Qwen VLM grounding.

        Args:
            query: Natural language description (e.g. "the red block").
            server_url: Grounding server URL on Spark.

        Returns:
            Dict with bbox, center pixel, robot coords, depth.
        """
        from common.grounding_client import GroundingClient

        # Capture fresh frames
        cap = self.capture()
        if not cap.get("success"):
            return {"success": False, "error": f"Capture failed: {cap.get('error')}"}

        color_frame = self._color_frame
        depth_frame = self._depth_frame
        if color_frame is None:
            return {"success": False, "error": "No color frame"}
        if depth_frame is None:
            return {"success": False, "error": "No depth frame"}

        # Save for skill_logger
        try:
            cv2.imwrite("/tmp/skill_frame.jpg", color_frame)
        except Exception:
            pass

        # Check calibration
        is_wrist = (self._camera_info is not None
                    and "D405" in (self._camera_info.product_line or ""))
        if is_wrist:
            if self._wrist_calibration is None:
                return {"success": False,
                        "error": "No wrist calibration."}
            if self._ee_pose is None:
                return {"success": False,
                        "error": "Wrist camera needs EE pose."}
        elif self._calibration_matrix is None:
            return {"success": False,
                    "error": "No calibration. Run calibration first."}

        # Encode and send to grounding server
        h, w = color_frame.shape[:2]
        _, jpeg_buf = cv2.imencode(".jpg", color_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
        client = GroundingClient(server_url)
        t0 = time.time()
        result = client.ground(jpeg_buf.tobytes(), query, width=w, height=h)
        inference_ms = round((time.time() - t0) * 1000, 1)

        if not result.get("success"):
            return {"success": False,
                    "error": f"Grounding failed: {result.get('error', 'unknown')}",
                    "inference_ms": inference_ms}

        # Get center pixel from bbox
        bbox = result["bbox"]  # [x1, y1, x2, y2]
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        # Resolve to 3D
        depth_result = self.get_depth_at(cx, cy, radius=5)
        if not depth_result.get("valid"):
            return {"success": False,
                    "error": "No depth at detected location",
                    "bbox": bbox, "center_pixel": [cx, cy],
                    "inference_ms": inference_ms}

        pos = depth_result["position_m"]
        cam_point = np.array([pos["x"], pos["y"], pos["z"], 1.0])
        if is_wrist and self._wrist_calibration is not None and self._ee_pose is not None:
            robot_point = self._ee_pose @ self._wrist_calibration @ cam_point
        else:
            robot_point = self._calibration_matrix @ cam_point

        return {
            "success": True,
            "query": query,
            "bbox": bbox,
            "center_pixel": [cx, cy],
            "depth_mm": depth_result["depth_mm"],
            "robot_x": round(float(robot_point[0]), 4),
            "robot_y": round(float(robot_point[1]), 4),
            "robot_z": round(float(robot_point[2]), 4),
            "inference_ms": inference_ms,
        }


# Singleton
_global_controller: Optional[RealSenseController] = None


def get_realsense_controller() -> RealSenseController:
    """Get or create the global RealSense controller."""
    global _global_controller
    if _global_controller is None:
        _global_controller = RealSenseController()
    return _global_controller
