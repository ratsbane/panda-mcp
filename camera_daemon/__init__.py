"""Camera daemon - captures frames and publishes via ZeroMQ."""

from .client import CameraClient, get_camera_client

__all__ = ["CameraClient", "get_camera_client"]
