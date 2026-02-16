"""
Client for the Qwen2.5-VL visual grounding server running on Spark.

Usage:
    from common.grounding_client import GroundingClient

    client = GroundingClient("http://spark:8090")
    result = client.ground(image_bytes, "the red block", width=1280, height=720)
    # result = {"success": True, "bbox": [x1,y1,x2,y2], "center": [cx,cy], ...}
"""

import base64
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://spark:8090"


class GroundingClient:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        """Check if the grounding server is healthy."""
        try:
            req = urllib.request.Request(f"{self.server_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return data.get("model_loaded", False)
        except Exception as e:
            logger.warning(f"Grounding server health check failed: {e}")
            return False

    def ground(
        self,
        image_jpeg: bytes,
        query: str,
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """
        Ground a natural language query in an image.

        Args:
            image_jpeg: JPEG-encoded image bytes
            query: Natural language description (e.g. "the red block")
            width: Original image width (for coordinate scaling)
            height: Original image height

        Returns:
            Dict with keys: success, bbox, center, raw_output, inference_ms, error
        """
        img_b64 = base64.b64encode(image_jpeg).decode()

        payload = json.dumps({
            "image": img_b64,
            "query": query,
            "width": width,
            "height": height,
        }).encode()

        req = urllib.request.Request(
            f"{self.server_url}/ground",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.error(f"Grounding request failed: {e}")
            return {"success": False, "error": str(e)}
