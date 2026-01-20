"""
Vision utilities for object detection and image processing.

Provides basic tools for finding objects in the workspace.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """Bounding box for detected object."""
    x: int  # top-left x
    y: int  # top-left y
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        cx, cy = self.center
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": cx,
            "center_y": cy,
            "area": self.area,
        }


@dataclass
class DetectedObject:
    """A detected object in the image."""
    bbox: BoundingBox
    confidence: float
    label: str
    color_bgr: Optional[tuple[int, int, int]] = None

    def to_dict(self) -> dict:
        result = {
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "label": self.label,
        }
        if self.color_bgr:
            result["color_bgr"] = self.color_bgr
        return result


def find_colored_objects(
    image: np.ndarray,
    color_lower: tuple[int, int, int],
    color_upper: tuple[int, int, int],
    min_area: int = 500,
    color_space: str = "hsv",
) -> list[DetectedObject]:
    """
    Find objects of a specific color range.

    Args:
        image: BGR image (from OpenCV)
        color_lower: Lower bound (H, S, V) or (B, G, R)
        color_upper: Upper bound (H, S, V) or (B, G, R)
        min_area: Minimum contour area to consider
        color_space: "hsv" or "bgr"

    Returns:
        List of detected objects
    """
    if color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        converted = image

    # Create mask
    mask = cv2.inRange(converted, np.array(color_lower), np.array(color_upper))

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Get average color in the region
        roi = image[y:y+h, x:x+w]
        avg_color = tuple(map(int, cv2.mean(roi)[:3]))

        objects.append(DetectedObject(
            bbox=BoundingBox(x, y, w, h),
            confidence=min(1.0, area / 10000),  # Simple confidence based on size
            label="colored_object",
            color_bgr=avg_color,
        ))

    # Sort by area (largest first)
    objects.sort(key=lambda o: o.bbox.area, reverse=True)

    return objects


def find_objects_by_contour(
    image: np.ndarray,
    min_area: int = 500,
    max_area: int = 100000,
) -> list[DetectedObject]:
    """
    Find objects using edge detection and contours.
    Good for finding distinct objects on a uniform background.

    Args:
        image: BGR image
        min_area: Minimum contour area
        max_area: Maximum contour area

    Returns:
        List of detected objects
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Get average color
        roi = image[y:y+h, x:x+w]
        avg_color = tuple(map(int, cv2.mean(roi)[:3]))

        objects.append(DetectedObject(
            bbox=BoundingBox(x, y, w, h),
            confidence=0.5,  # Medium confidence for contour-based
            label="object",
            color_bgr=avg_color,
        ))

    objects.sort(key=lambda o: o.bbox.area, reverse=True)
    return objects


def find_largest_object(
    image: np.ndarray,
    min_area: int = 500,
) -> Optional[DetectedObject]:
    """Find the largest distinct object in the image."""
    objects = find_objects_by_contour(image, min_area=min_area)
    return objects[0] if objects else None


def draw_detections(
    image: np.ndarray,
    objects: list[DetectedObject],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image."""
    result = image.copy()

    for obj in objects:
        bbox = obj.bbox
        cv2.rectangle(
            result,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            color,
            thickness,
        )

        # Draw center point
        cx, cy = bbox.center
        cv2.circle(result, (cx, cy), 5, color, -1)

        # Label
        label = f"{obj.label} ({obj.confidence:.2f})"
        cv2.putText(
            result,
            label,
            (bbox.x, bbox.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return result


# Common color ranges in HSV
COLOR_RANGES = {
    "red_low": ((0, 100, 100), (10, 255, 255)),
    "red_high": ((160, 100, 100), (180, 255, 255)),
    "green": ((35, 100, 100), (85, 255, 255)),
    "blue": ((100, 100, 100), (130, 255, 255)),
    "yellow": ((20, 100, 100), (35, 255, 255)),
    "orange": ((10, 100, 100), (20, 255, 255)),
    "purple": ((130, 100, 100), (160, 255, 255)),
    "white": ((0, 0, 200), (180, 30, 255)),
}


def find_colored_blocks(
    image: np.ndarray,
    colors: list[str] = None,
    min_area: int = 1000,
) -> list[DetectedObject]:
    """
    Find colored blocks (like wooden toy blocks).

    Args:
        image: BGR image
        colors: List of color names to look for (default: all)
        min_area: Minimum block size

    Returns:
        List of detected colored blocks
    """
    if colors is None:
        colors = list(COLOR_RANGES.keys())

    all_objects = []

    for color_name in colors:
        if color_name not in COLOR_RANGES:
            continue

        lower, upper = COLOR_RANGES[color_name]
        objects = find_colored_objects(image, lower, upper, min_area=min_area)

        for obj in objects:
            obj.label = f"{color_name}_block"

        all_objects.extend(objects)

    # Remove duplicates (overlapping detections)
    # Simple approach: keep largest when centers are close
    filtered = []
    for obj in sorted(all_objects, key=lambda o: o.bbox.area, reverse=True):
        cx, cy = obj.bbox.center
        is_duplicate = False

        for existing in filtered:
            ex, ey = existing.bbox.center
            distance = np.sqrt((cx - ex)**2 + (cy - ey)**2)
            if distance < 50:  # Within 50 pixels = probably same object
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(obj)

    return filtered
