"""
Client for the Moondream 2B inference server running on Spark.

Sends camera frames to the server and receives object detections
or point localizations. Used by the learned_pick pipeline.
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import json
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://spark:8091"


@dataclass
class MoondreamDetection:
    """A detected object from Moondream."""
    center_x: int  # pixel x
    center_y: int  # pixel y
    bbox_w: int  # bounding box width in pixels
    bbox_h: int  # bounding box height in pixels
    x_min: float  # normalized 0-1
    y_min: float
    x_max: float
    y_max: float


@dataclass
class MoondreamPoint:
    """A point localization from Moondream."""
    pixel_x: int
    pixel_y: int
    x: float  # normalized 0-1
    y: float


def _encode_frame(frame: np.ndarray) -> str:
    """Encode BGR frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _post_json(url: str, data: dict, timeout: float = 10.0) -> dict:
    """POST JSON and return response dict."""
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def detect(
    frame: np.ndarray,
    query: str,
    server_url: str = DEFAULT_SERVER_URL,
) -> list[MoondreamDetection]:
    """Detect objects matching a natural language query.

    Args:
        frame: BGR camera frame
        query: e.g. "green block", "red block", "wooden block"
        server_url: Moondream server URL

    Returns:
        List of MoondreamDetection sorted by area (largest first)
    """
    url = f"{server_url}/detect"
    b64 = _encode_frame(frame)

    try:
        result = _post_json(url, {"image_base64": b64, "query": query})
    except (URLError, OSError) as e:
        logger.error(f"Moondream server unreachable at {server_url}: {e}")
        return []

    detections = []
    for obj in result.get("objects", []):
        detections.append(MoondreamDetection(
            center_x=obj["center_x"],
            center_y=obj["center_y"],
            bbox_w=obj["bbox_w"],
            bbox_h=obj["bbox_h"],
            x_min=obj["x_min"],
            y_min=obj["y_min"],
            x_max=obj["x_max"],
            y_max=obj["y_max"],
        ))

    # Sort by area, largest first
    detections.sort(key=lambda d: d.bbox_w * d.bbox_h, reverse=True)

    logger.info(
        f"Moondream detect({query!r}): {len(detections)} objects, "
        f"latency={result.get('latency_ms', 0):.0f}ms"
    )
    return detections


def point(
    frame: np.ndarray,
    query: str,
    server_url: str = DEFAULT_SERVER_URL,
) -> Optional[MoondreamPoint]:
    """Get the point location of a queried object.

    Args:
        frame: BGR camera frame
        query: e.g. "the green block", "the tiger toy"
        server_url: Moondream server URL

    Returns:
        MoondreamPoint or None if nothing found
    """
    url = f"{server_url}/point"
    b64 = _encode_frame(frame)

    try:
        result = _post_json(url, {"image_base64": b64, "query": query})
    except (URLError, OSError) as e:
        logger.error(f"Moondream server unreachable at {server_url}: {e}")
        return None

    points = result.get("points", [])
    if not points:
        return None

    pt = points[0]
    logger.info(
        f"Moondream point({query!r}): ({pt['pixel_x']}, {pt['pixel_y']}), "
        f"latency={result.get('latency_ms', 0):.0f}ms"
    )
    return MoondreamPoint(
        pixel_x=pt["pixel_x"],
        pixel_y=pt["pixel_y"],
        x=pt["x"],
        y=pt["y"],
    )
