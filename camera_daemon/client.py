"""
Camera client - subscribes to frames from camera daemon via ZeroMQ.

Uses a fresh ZMQ socket per frame request to guarantee freshness.
A persistent socket buffers frames between calls, causing staleness
when calls are seconds/minutes apart (e.g. MCP tool calls).

Usage:
    from camera_daemon import CameraClient, get_camera_client

    # Option 1: Use singleton
    client = get_camera_client()
    frame = client.get_frame()

    # Option 2: Create your own instance
    client = CameraClient()
    client.connect()
    frame = client.get_frame()
    client.disconnect()
"""

import cv2
import zmq
import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Match server defaults
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_IPC_ENDPOINT = "ipc:///tmp/camera-daemon.sock"


@dataclass
class CameraInfo:
    """Camera information from the daemon."""
    width: int
    height: int
    channels: int
    connected: bool
    endpoint: str


class CameraClient:
    """Subscribes to camera frames from the daemon."""

    def __init__(
        self,
        endpoint: str = DEFAULT_IPC_ENDPOINT,
        fallback_endpoint: str = DEFAULT_ENDPOINT,
        timeout_ms: int = 1000,
    ):
        """
        Initialize camera client.

        Args:
            endpoint: Primary ZMQ endpoint (default: IPC for speed)
            fallback_endpoint: Fallback if primary fails (default: TCP)
            timeout_ms: Receive timeout in milliseconds
        """
        self.endpoint = endpoint
        self.fallback_endpoint = fallback_endpoint
        self.timeout_ms = timeout_ms

        self.context: Optional[zmq.Context] = None
        self.connected = False
        self.active_endpoint = None

        # Cache last frame info
        self._last_width = 0
        self._last_height = 0
        self._last_channels = 0

    def connect(self) -> bool:
        """Connect to the camera daemon (validates endpoint)."""
        if self.connected:
            return True

        self.context = zmq.Context()

        # Try primary endpoint first (IPC)
        try:
            parts = self._fresh_recv(self.endpoint)
            self._parse_metadata(parts)
            self.connected = True
            self.active_endpoint = self.endpoint
            logger.info(f"Connected to camera daemon at {self.endpoint}")
            return True
        except zmq.Again:
            logger.warning(f"No response from {self.endpoint}, trying fallback...")
        except zmq.ZMQError as e:
            logger.warning(f"Could not connect to {self.endpoint}: {e}")

        # Try fallback endpoint (TCP)
        if self.fallback_endpoint:
            try:
                parts = self._fresh_recv(self.fallback_endpoint)
                self._parse_metadata(parts)
                self.connected = True
                self.active_endpoint = self.fallback_endpoint
                logger.info(f"Connected to camera daemon at {self.fallback_endpoint}")
                return True
            except (zmq.Again, zmq.ZMQError) as e:
                logger.error(f"Could not connect to fallback {self.fallback_endpoint}: {e}")

        # Failed to connect
        self.disconnect()
        return False

    def disconnect(self):
        """Disconnect from the daemon."""
        self.connected = False
        self.active_endpoint = None

        if self.context:
            self.context.term()
            self.context = None

    def _fresh_recv(self, endpoint: str) -> list:
        """Create a fresh ZMQ socket, receive one frame, close socket.

        This guarantees the frame was published AFTER the socket was created,
        eliminating any stale buffered frames. Same approach as the web
        viewer's _grab_usb_frame().

        Args:
            endpoint: ZMQ endpoint to connect to

        Returns:
            Raw multipart message parts

        Raises:
            zmq.Again: on timeout
            zmq.ZMQError: on connection failure
        """
        sock = self.context.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt_string(zmq.SUBSCRIBE, "frame")
        try:
            sock.connect(endpoint)
            parts = sock.recv_multipart()
            return parts
        finally:
            sock.close()

    def _parse_metadata(self, parts: list):
        """Parse metadata from multipart message and update cache."""
        if len(parts) >= 3:
            metadata = np.frombuffer(parts[1], dtype=np.int32)
            self._last_width = int(metadata[0])
            self._last_height = int(metadata[1])
            self._last_channels = int(metadata[2])

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera daemon.

        Creates a fresh ZMQ socket per call to guarantee frame freshness.

        Returns:
            numpy array (BGR format) or None if not connected/timeout
        """
        if not self.connected:
            if not self.connect():
                return None

        try:
            parts = self._fresh_recv(self.active_endpoint)

            if len(parts) != 3:
                logger.error(f"Expected 3 message parts, got {len(parts)}")
                return None

            topic, metadata_bytes, jpeg_bytes = parts
            self._parse_metadata(parts)

            # Decode JPEG
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode JPEG frame")
                return None

            return frame
        except zmq.Again:
            logger.warning("Timeout waiting for frame")
            return None
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return None

    def get_frame_jpeg(self) -> Optional[bytes]:
        """
        Get the latest frame as JPEG bytes (avoids decode/re-encode).

        Creates a fresh ZMQ socket per call to guarantee frame freshness.

        Returns:
            JPEG bytes or None if not connected/timeout
        """
        if not self.connected:
            if not self.connect():
                return None

        try:
            parts = self._fresh_recv(self.active_endpoint)

            if len(parts) != 3:
                return None

            _, metadata_bytes, jpeg_bytes = parts
            self._parse_metadata(parts)

            return bytes(jpeg_bytes)
        except zmq.Again:
            logger.warning("Timeout waiting for frame")
            return None
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return None

    def get_info(self) -> CameraInfo:
        """Get camera information."""
        return CameraInfo(
            width=self._last_width,
            height=self._last_height,
            channels=self._last_channels,
            connected=self.connected,
            endpoint=self.active_endpoint or "not connected",
        )

    def capture_burst(self, count: int, interval_ms: int = 100) -> list[np.ndarray]:
        """
        Capture multiple frames.

        Args:
            count: Number of frames to capture
            interval_ms: Not used (daemon controls frame rate)

        Returns:
            List of frames
        """
        frames = []
        for _ in range(count):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
        return frames


# Singleton client
_client: Optional[CameraClient] = None


def get_camera_client() -> CameraClient:
    """Get the singleton camera client."""
    global _client
    if _client is None:
        _client = CameraClient()
    return _client
