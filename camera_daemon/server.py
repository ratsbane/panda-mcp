#!/usr/bin/env python3
"""
Camera daemon - captures frames from USB camera and publishes via ZeroMQ.

This allows multiple consumers (camera-mcp, scene_viewer, etc.) to access
camera frames simultaneously without conflicts.

Usage:
    python -m camera_daemon.server [options]

    Or as systemd service:
    systemctl --user start camera-daemon
"""

import cv2
import zmq
import time
import signal
import logging
import argparse
import numpy as np
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_DEVICE = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_IPC_ENDPOINT = "ipc:///tmp/camera-daemon.sock"


class CameraDaemon:
    """Captures frames and publishes via ZeroMQ."""

    def __init__(
        self,
        device: int = DEFAULT_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        endpoint: str = DEFAULT_ENDPOINT,
        use_ipc: bool = True,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.endpoint = endpoint
        self.use_ipc = use_ipc
        self.ipc_endpoint = DEFAULT_IPC_ENDPOINT

        self.cap: Optional[cv2.VideoCapture] = None
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.running = False

        # Stats
        self.frame_count = 0
        self.start_time = 0
        self.last_stats_time = 0

    def start(self) -> bool:
        """Initialize camera and ZMQ socket."""
        # Open camera
        logger.info(f"Opening camera {self.device}...")
        self.cap = cv2.VideoCapture(self.device)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.device}")
            return False

        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps}fps")

        # Set up ZMQ publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)

        # Bind to TCP endpoint
        try:
            self.socket.bind(self.endpoint)
            logger.info(f"ZMQ publisher bound to {self.endpoint}")
        except zmq.ZMQError as e:
            logger.error(f"Failed to bind to {self.endpoint}: {e}")
            self.stop()
            return False

        # Also bind to IPC for local connections (faster)
        if self.use_ipc:
            try:
                self.socket.bind(self.ipc_endpoint)
                logger.info(f"ZMQ publisher also bound to {self.ipc_endpoint}")
            except zmq.ZMQError as e:
                logger.warning(f"Could not bind IPC endpoint: {e}")

        self.running = True
        self.start_time = time.time()
        self.last_stats_time = self.start_time

        return True

    def stop(self):
        """Clean up resources."""
        self.running = False

        if self.socket:
            self.socket.close()
            self.socket = None

        if self.context:
            self.context.term()
            self.context = None

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("Camera daemon stopped")

    def run(self):
        """Main capture and publish loop."""
        if not self.start():
            return

        frame_interval = 1.0 / self.fps

        logger.info("Starting capture loop...")

        try:
            while self.running:
                loop_start = time.time()

                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Encode frame as JPEG for efficient transport
                _, jpeg_data = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                # Publish frame with metadata
                # Topic: "frame"
                # Message: [width, height, channels, jpeg_bytes]
                height, width = frame.shape[:2]
                channels = frame.shape[2] if len(frame.shape) > 2 else 1

                metadata = np.array([width, height, channels], dtype=np.int32)

                self.socket.send_multipart([
                    b"frame",
                    metadata.tobytes(),
                    jpeg_data.tobytes(),
                ])

                self.frame_count += 1

                # Log stats periodically
                now = time.time()
                if now - self.last_stats_time >= 10.0:
                    elapsed = now - self.start_time
                    fps = self.frame_count / elapsed
                    logger.info(f"Stats: {self.frame_count} frames, {fps:.1f} fps avg")
                    self.last_stats_time = now

                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Camera daemon - captures and publishes frames via ZeroMQ"
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=DEFAULT_DEVICE,
        help=f"Camera device index (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "-W", "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Frame width (default: {DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "-H", "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Frame height (default: {DEFAULT_HEIGHT})"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Target FPS (default: {DEFAULT_FPS})"
    )
    parser.add_argument(
        "-e", "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=f"ZMQ endpoint (default: {DEFAULT_ENDPOINT})"
    )
    parser.add_argument(
        "--no-ipc",
        action="store_true",
        help="Disable IPC endpoint"
    )

    args = parser.parse_args()

    daemon = CameraDaemon(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        endpoint=args.endpoint,
        use_ipc=not args.no_ipc,
    )

    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        daemon.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    daemon.run()


if __name__ == "__main__":
    main()
