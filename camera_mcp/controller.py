"""
Camera controller for workspace observation.

Provides image capture from USB camera with basic processing.
"""

import os
import io
import base64
import logging
from typing import Optional
from dataclasses import dataclass

# Check for mock mode
MOCK_MODE = os.environ.get("CAMERA_MOCK", "0") == "1"

if not MOCK_MODE:
    try:
        import cv2
        import numpy as np
        from PIL import Image
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        logging.warning("OpenCV not available, running in mock mode")
        MOCK_MODE = True
else:
    CV2_AVAILABLE = False
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration."""
    device: int = 0  # /dev/video0
    width: int = 1280
    height: int = 720
    fps: int = 30
    jpeg_quality: int = 85


@dataclass 
class CameraInfo:
    """Camera information."""
    device: int
    width: int
    height: int
    fps: float
    backend: str
    is_mock: bool
    
    def to_dict(self) -> dict:
        return {
            "device": f"/dev/video{self.device}",
            "resolution": {"width": self.width, "height": self.height},
            "fps": self.fps,
            "backend": self.backend,
            "is_mock": self.is_mock,
        }


class CameraController:
    """
    Controller for USB camera.
    
    Handles capture, encoding, and basic image operations.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._capture = None
        self._connected = False
        self._mock_mode = MOCK_MODE
    
    def connect(self) -> dict:
        """Initialize camera capture."""
        if self._mock_mode:
            logger.info("Running camera in MOCK mode")
            self._connected = True
            return {"connected": True, "mock": True}
        
        try:
            self._capture = cv2.VideoCapture(self.config.device)
            
            if not self._capture.isOpened():
                return {"connected": False, "error": f"Could not open /dev/video{self.config.device}"}
            
            # Set resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Read actual settings (may differ from requested)
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
            
            self._connected = True
            
            logger.info(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            return {
                "connected": True,
                "mock": False,
                "resolution": {"width": actual_width, "height": actual_height},
                "fps": actual_fps,
            }
            
        except Exception as e:
            logger.error(f"Camera connection failed: {e}")
            return {"connected": False, "error": str(e)}
    
    def disconnect(self):
        """Release camera."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    def get_info(self) -> CameraInfo:
        """Get camera information."""
        if self._mock_mode:
            return CameraInfo(
                device=self.config.device,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
                backend="mock",
                is_mock=True,
            )
        
        if not self._connected or self._capture is None:
            raise RuntimeError("Camera not connected")
        
        return CameraInfo(
            device=self.config.device,
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._capture.get(cv2.CAP_PROP_FPS),
            backend=self._capture.getBackendName(),
            is_mock=False,
        )
    
    def _generate_mock_frame(self) -> bytes:
        """Generate a mock frame for testing."""
        # Create a simple test pattern
        width, height = self.config.width, self.config.height
        
        # Generate gradient with timestamp-like pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            frame[y, :, 0] = int(255 * y / height)  # Blue gradient
            frame[y, :, 2] = int(255 * (1 - y / height))  # Red gradient
        
        # Add a grid pattern
        for x in range(0, width, 100):
            frame[:, x:x+2, :] = 128
        for y in range(0, height, 100):
            frame[y:y+2, :, :] = 128
        
        # Add text indicating mock mode
        # (In real implementation, we'd use cv2.putText)
        
        # Encode to JPEG
        from PIL import Image
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
        return buffer.getvalue()
    
    def capture_frame(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: str = "jpeg",
    ) -> dict:
        """
        Capture a single frame.
        
        Args:
            width: Target width (optional, for resizing)
            height: Target height (optional, for resizing)
            format: Output format ("jpeg" or "png")
        
        Returns:
            dict with base64 encoded image and metadata
        """
        if not self._connected:
            return {"success": False, "error": "Camera not connected"}
        
        try:
            if self._mock_mode:
                jpeg_data = self._generate_mock_frame()
                return {
                    "success": True,
                    "format": "jpeg",
                    "width": self.config.width,
                    "height": self.config.height,
                    "size_bytes": len(jpeg_data),
                    "data": base64.b64encode(jpeg_data).decode("utf-8"),
                    "mock": True,
                }
            
            ret, frame = self._capture.read()
            
            if not ret:
                return {"success": False, "error": "Failed to capture frame"}
            
            # Resize if requested
            if width or height:
                current_h, current_w = frame.shape[:2]
                if width and not height:
                    height = int(current_h * width / current_w)
                elif height and not width:
                    width = int(current_w * height / current_h)
                frame = cv2.resize(frame, (width, height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode
            img = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            
            if format.lower() == "png":
                img.save(buffer, format="PNG")
                mime_type = "image/png"
            else:
                img.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
                mime_type = "image/jpeg"
            
            image_data = buffer.getvalue()
            
            return {
                "success": True,
                "format": format,
                "mime_type": mime_type,
                "width": img.width,
                "height": img.height,
                "size_bytes": len(image_data),
                "data": base64.b64encode(image_data).decode("utf-8"),
            }
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return {"success": False, "error": str(e)}
    
    def capture_burst(
        self,
        count: int = 5,
        interval_ms: int = 100,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict:
        """
        Capture multiple frames in sequence.
        
        Args:
            count: Number of frames to capture (1-30)
            interval_ms: Delay between captures in milliseconds
            width, height: Optional resize dimensions
        
        Returns:
            dict with list of base64 encoded frames
        """
        if not self._connected:
            return {"success": False, "error": "Camera not connected"}
        
        count = min(max(count, 1), 30)  # Clamp to 1-30
        
        frames = []
        
        try:
            import time
            
            for i in range(count):
                result = self.capture_frame(width=width, height=height)
                
                if result["success"]:
                    frames.append({
                        "index": i,
                        "data": result["data"],
                        "width": result["width"],
                        "height": result["height"],
                    })
                
                if i < count - 1:
                    time.sleep(interval_ms / 1000.0)
            
            return {
                "success": True,
                "frame_count": len(frames),
                "frames": frames,
            }
            
        except Exception as e:
            logger.error(f"Burst capture failed: {e}")
            return {"success": False, "error": str(e)}
    
    def set_resolution(self, width: int, height: int) -> dict:
        """Change camera resolution."""
        if not self._connected:
            return {"success": False, "error": "Camera not connected"}
        
        if self._mock_mode:
            self.config.width = width
            self.config.height = height
            return {"success": True, "width": width, "height": height, "mock": True}
        
        try:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return {
                "success": True,
                "requested": {"width": width, "height": height},
                "actual": {"width": actual_width, "height": actual_height},
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Singleton controller
_controller: Optional[CameraController] = None


def get_camera_controller() -> CameraController:
    global _controller
    if _controller is None:
        _controller = CameraController()
    return _controller
