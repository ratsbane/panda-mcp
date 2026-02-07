"""
Scene interpretation module.

Analyzes camera images and generates structured descriptions of the workspace.
Provides the core logic shared by CLI, MCP, and viewer interfaces.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .vision import (
    DetectedObject,
    find_objects_by_contour,
    find_colored_blocks,
    draw_detections,
    COLOR_RANGES,
)
from .calibration import get_transformer

# Lazy import for segmentation (heavy dependency)
_segmenter = None

# Lazy import for Hailo YOLO detector
_hailo_detector_checked = False
_hailo_detector = None


def _get_segmenter():
    """Get the segmenter singleton (lazy-loaded)."""
    global _segmenter
    if _segmenter is None:
        try:
            from .segmentation import get_segmenter
            _segmenter = get_segmenter()
        except ImportError:
            return None
    return _segmenter


def _get_hailo_detector():
    """Get the Hailo YOLO detector singleton (lazy-loaded). Returns None if unavailable."""
    global _hailo_detector_checked, _hailo_detector
    if not _hailo_detector_checked:
        _hailo_detector_checked = True
        try:
            from .hailo_detector import get_hailo_detector
            _hailo_detector = get_hailo_detector()
        except Exception:
            _hailo_detector = None
    return _hailo_detector


class SpatialRelation(Enum):
    """Spatial relationships between objects."""
    LEFT_OF = "left of"
    RIGHT_OF = "right of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in front of"
    BEHIND = "behind"
    NEAR = "near"
    OVERLAPPING = "overlapping"


@dataclass
class ObjectDescription:
    """Rich description of a detected object."""
    obj: DetectedObject
    position_description: str  # "left side", "center", "upper right", etc.
    size_description: str  # "small", "medium", "large"
    color_name: Optional[str] = None
    estimated_distance: Optional[str] = None  # "close", "mid", "far" based on Y position


@dataclass
class SpatialRelationship:
    """A spatial relationship between two objects."""
    object1_idx: int
    object2_idx: int
    relation: SpatialRelation

    def describe(self, objects: list[ObjectDescription]) -> str:
        obj1 = objects[self.object1_idx]
        obj2 = objects[self.object2_idx]
        name1 = obj1.obj.label
        name2 = obj2.obj.label
        return f"{name1} is {self.relation.value} {name2}"


@dataclass
class SceneDescription:
    """Complete description of a scene."""
    objects: list[ObjectDescription] = field(default_factory=list)
    relationships: list[SpatialRelationship] = field(default_factory=list)
    workspace_state: str = "unknown"  # "empty", "sparse", "cluttered"
    summary: str = ""

    def to_dict(self) -> dict:
        # Try to get calibration for robot coordinates
        transformer = None
        try:
            t = get_transformer()
            if t.calibration.homography is not None:
                transformer = t
        except Exception:
            pass

        objects_list = []
        for o in self.objects:
            obj_dict = {
                "label": o.obj.label,
                "position": o.position_description,
                "size": o.size_description,
                "color": o.color_name,
                "distance": o.estimated_distance,
                "bbox": o.obj.bbox.to_dict(),
                "confidence": o.obj.confidence,
            }
            if transformer is not None:
                cx, cy = o.obj.bbox.center
                rx, ry, rz = transformer.pixel_to_robot(cx, cy)
                obj_dict["robot_coords"] = {
                    "x": round(float(rx), 4),
                    "y": round(float(ry), 4),
                    "z": round(float(rz), 4),
                }
            objects_list.append(obj_dict)

        return {
            "object_count": len(self.objects),
            "objects": objects_list,
            "relationships": [
                {
                    "object1": self.objects[r.object1_idx].obj.label,
                    "object2": self.objects[r.object2_idx].obj.label,
                    "relation": r.relation.value,
                }
                for r in self.relationships
            ],
            "workspace_state": self.workspace_state,
            "summary": self.summary,
        }


def _get_position_description(bbox, image_width: int, image_height: int) -> str:
    """Describe position in image frame."""
    cx, cy = bbox.center

    # Horizontal position
    if cx < image_width * 0.33:
        h_pos = "left"
    elif cx > image_width * 0.67:
        h_pos = "right"
    else:
        h_pos = "center"

    # Vertical position (remember: top of image = far from camera typically)
    if cy < image_height * 0.33:
        v_pos = "far"
    elif cy > image_height * 0.67:
        v_pos = "near"
    else:
        v_pos = "middle"

    if h_pos == "center" and v_pos == "middle":
        return "center of workspace"
    elif h_pos == "center":
        return f"{v_pos} side"
    elif v_pos == "middle":
        return f"{h_pos} side"
    else:
        return f"{v_pos} {h_pos}"


def _get_size_description(area: int, image_area: int) -> str:
    """Describe relative size of object."""
    ratio = area / image_area
    if ratio < 0.01:
        return "very small"
    elif ratio < 0.03:
        return "small"
    elif ratio < 0.08:
        return "medium"
    elif ratio < 0.15:
        return "large"
    else:
        return "very large"


def _get_distance_estimate(cy: int, image_height: int) -> str:
    """Estimate distance based on vertical position (assumes top-down-ish view)."""
    if cy < image_height * 0.33:
        return "far"
    elif cy > image_height * 0.67:
        return "close"
    else:
        return "mid-range"


def _identify_color(bgr: tuple[int, int, int]) -> Optional[str]:
    """Try to identify a color name from BGR values."""
    if bgr is None:
        return None

    # Convert single pixel to HSV
    pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # Low saturation = grayscale
    if s < 30:
        if v < 50:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"

    # Check hue ranges
    if h < 10 or h > 160:
        return "red"
    elif h < 22:
        return "orange"
    elif h < 38:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 160:
        return "purple"

    return None


def _compute_relationships(objects: list[ObjectDescription]) -> list[SpatialRelationship]:
    """Compute spatial relationships between objects."""
    relationships = []

    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue

            cx1, cy1 = obj1.obj.bbox.center
            cx2, cy2 = obj2.obj.bbox.center

            dx = cx2 - cx1
            dy = cy2 - cy1

            # Determine primary relationship
            # Horizontal relationships
            if abs(dx) > abs(dy) * 1.5:  # Primarily horizontal
                if dx > 50:
                    relationships.append(SpatialRelationship(i, j, SpatialRelation.LEFT_OF))
                elif dx < -50:
                    relationships.append(SpatialRelationship(i, j, SpatialRelation.RIGHT_OF))
            # Vertical relationships (in image = depth in world)
            elif abs(dy) > abs(dx) * 1.5:
                if dy > 50:
                    relationships.append(SpatialRelationship(i, j, SpatialRelation.IN_FRONT_OF))
                elif dy < -50:
                    relationships.append(SpatialRelationship(i, j, SpatialRelation.BEHIND))
            # Close together
            elif abs(dx) < 100 and abs(dy) < 100:
                relationships.append(SpatialRelationship(i, j, SpatialRelation.NEAR))

    return relationships


def _generate_summary(objects: list[ObjectDescription], workspace_state: str) -> str:
    """Generate natural language summary of the scene."""
    if not objects:
        return "The workspace appears empty. No distinct objects detected."

    n = len(objects)

    # Check if we have YOLO class labels (not generic "object"/"colored_object")
    has_class_labels = any(
        o.obj.label not in ("object", "colored_object") and not o.obj.label.endswith("_block")
        for o in objects
    )

    # Build summary
    parts = []

    if has_class_labels:
        # Group by class label for richer summary
        label_counts: dict[str, int] = {}
        for obj in objects:
            label = obj.obj.label
            label_counts[label] = label_counts.get(label, 0) + 1

        # Build "2 persons, 1 cup, 1 bottle" style summary
        label_strs = []
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            if count == 1:
                label_strs.append(f"1 {label}")
            else:
                label_strs.append(f"{count} {label}s")

        if n == 1:
            obj = objects[0]
            color_str = f"{obj.color_name} " if obj.color_name else ""
            parts.append(f"There is one {color_str}{obj.obj.label} in the {obj.position_description}.")
        else:
            parts.append(f"Detected {n} objects: {', '.join(label_strs)}.")
    else:
        # Fallback: color-based summary (original behavior)
        colors: dict[str, int] = {}
        for obj in objects:
            color = obj.color_name or "unknown color"
            colors[color] = colors.get(color, 0) + 1

        if n == 1:
            obj = objects[0]
            color_str = f"{obj.color_name} " if obj.color_name else ""
            parts.append(f"There is one {obj.size_description} {color_str}object in the {obj.position_description}.")
        else:
            parts.append(f"There are {n} objects in the workspace.")

            # Mention colors if interesting
            if len(colors) > 1:
                color_strs = [f"{count} {color}" for color, count in colors.items() if color != "unknown color"]
                if color_strs:
                    parts.append(f"Colors detected: {', '.join(color_strs)}.")

    # Describe distribution
    if n > 1:
        positions = [o.position_description for o in objects]
        if all("left" in p for p in positions):
            parts.append("All objects are on the left side.")
        elif all("right" in p for p in positions):
            parts.append("All objects are on the right side.")
        elif all("center" in p for p in positions):
            parts.append("Objects are clustered in the center.")

    # Add workspace state
    if workspace_state == "cluttered":
        parts.append("The workspace is cluttered.")
    elif workspace_state == "sparse":
        parts.append("Objects are spread out.")

    return " ".join(parts)


def interpret_scene(
    image: np.ndarray,
    use_color_detection: bool = True,
    use_contour_detection: bool = True,
    use_segmentation: bool = False,
    use_yolo_detection: bool = False,
    min_area: int = 500,
) -> SceneDescription:
    """
    Analyze an image and generate a scene description.

    Args:
        image: BGR image from camera
        use_color_detection: Look for colored blocks
        use_contour_detection: Use edge-based detection
        use_segmentation: Use MobileSAM for instance segmentation (slower but more accurate)
        use_yolo_detection: Use Hailo-accelerated YOLOv8 for real object class labels
        min_area: Minimum object area in pixels

    Returns:
        SceneDescription with objects, relationships, and summary
    """
    height, width = image.shape[:2]
    image_area = width * height

    all_objects = []
    yolo_detected = False

    # YOLO detection (fastest, provides real class labels)
    if use_yolo_detection:
        detector = _get_hailo_detector()
        if detector is not None:
            yolo_objects = detector.detect(image)
            # Filter by area
            for obj in yolo_objects:
                if obj.bbox.area >= min_area:
                    all_objects.append(obj)
            if all_objects:
                yolo_detected = True

    # If YOLO handled detection, skip other methods
    if not yolo_detected:
        # Segmentation-based detection (most accurate, but slower)
        if use_segmentation:
            segmenter = _get_segmenter()
            if segmenter is not None:
                seg_objects = segmenter.segment(image)
                # Filter by area
                for obj in seg_objects:
                    if obj.bbox.area >= min_area:
                        all_objects.append(obj)

        # Color-based detection (if not using segmentation, or to augment it)
        if use_color_detection and not use_segmentation:
            colored = find_colored_blocks(image, min_area=min_area)
            all_objects.extend(colored)
        elif use_color_detection and use_segmentation:
            # Augment segmentation with color labels
            colored = find_colored_blocks(image, min_area=min_area)
            for seg_obj in all_objects:
                # Find matching color detection
                for color_obj in colored:
                    cx1, cy1 = seg_obj.bbox.center
                    cx2, cy2 = color_obj.bbox.center
                    if abs(cx1 - cx2) < 50 and abs(cy1 - cy2) < 50:
                        # Transfer color label
                        seg_obj.label = color_obj.label
                        break

        # Contour-based detection (may find objects other methods missed)
        if use_contour_detection and not use_segmentation:
            contour_objs = find_objects_by_contour(image, min_area=min_area)

            # Add only if not overlapping with existing detections
            for obj in contour_objs:
                cx, cy = obj.bbox.center
                is_duplicate = False

                for existing in all_objects:
                    ex, ey = existing.bbox.center
                    if abs(cx - ex) < 50 and abs(cy - ey) < 50:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    all_objects.append(obj)

    # Build rich descriptions
    object_descriptions = []
    for obj in all_objects:
        desc = ObjectDescription(
            obj=obj,
            position_description=_get_position_description(obj.bbox, width, height),
            size_description=_get_size_description(obj.bbox.area, image_area),
            color_name=_identify_color(obj.color_bgr),
            estimated_distance=_get_distance_estimate(obj.bbox.center[1], height),
        )
        object_descriptions.append(desc)

    # Determine workspace state
    if len(object_descriptions) == 0:
        workspace_state = "empty"
    elif len(object_descriptions) <= 2:
        workspace_state = "sparse"
    elif len(object_descriptions) <= 5:
        workspace_state = "moderate"
    else:
        workspace_state = "cluttered"

    # Compute relationships
    relationships = _compute_relationships(object_descriptions)

    # Generate summary
    summary = _generate_summary(object_descriptions, workspace_state)

    return SceneDescription(
        objects=object_descriptions,
        relationships=relationships,
        workspace_state=workspace_state,
        summary=summary,
    )


def annotate_scene(
    image: np.ndarray,
    scene: SceneDescription,
    show_labels: bool = True,
    show_relationships: bool = False,
) -> np.ndarray:
    """
    Draw scene annotations on an image.

    Args:
        image: BGR image
        scene: SceneDescription from interpret_scene
        show_labels: Draw object labels
        show_relationships: Draw lines between related objects

    Returns:
        Annotated image copy
    """
    result = image.copy()

    # Draw bounding boxes and labels
    for i, obj_desc in enumerate(scene.objects):
        obj = obj_desc.obj
        bbox = obj.bbox

        # Choose color based on detected color
        if obj_desc.color_name:
            color_map = {
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 0, 0),
                "yellow": (0, 255, 255),
                "orange": (0, 165, 255),
                "purple": (255, 0, 255),
                "white": (255, 255, 255),
                "black": (128, 128, 128),
                "gray": (180, 180, 180),
            }
            color = color_map.get(obj_desc.color_name, (0, 255, 0))
        else:
            # Use distinct colors for unlabeled objects
            palette = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
                       (255, 255, 100), (255, 100, 255), (100, 255, 255)]
            color = palette[i % len(palette)]

        # Draw segmentation mask if available
        if obj.mask is not None:
            overlay = result.copy()
            overlay[obj.mask] = color
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        # Draw rectangle
        cv2.rectangle(
            result,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            color,
            2,
        )

        # Draw center
        cx, cy = bbox.center
        cv2.circle(result, (cx, cy), 5, color, -1)

        # Draw label
        if show_labels:
            # Use class label if it's a real YOLO label, otherwise fall back to color/size
            has_class_label = obj.label not in ("object", "colored_object") and not obj.label.endswith("_block")
            if has_class_label:
                label = f"{obj.label} {obj.confidence:.2f}"
            else:
                label = f"{i+1}: {obj_desc.color_name or 'object'} ({obj_desc.size_description})"
            cv2.putText(
                result,
                label,
                (bbox.x, bbox.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Draw relationship lines
    if show_relationships:
        for rel in scene.relationships:
            obj1 = scene.objects[rel.object1_idx]
            obj2 = scene.objects[rel.object2_idx]
            pt1 = obj1.obj.bbox.center
            pt2 = obj2.obj.bbox.center
            cv2.line(result, pt1, pt2, (255, 255, 0), 1)

    # Draw summary at top
    cv2.putText(
        result,
        f"Objects: {len(scene.objects)} | State: {scene.workspace_state}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return result


def capture_and_interpret(
    camera_index: int = 0,
    warmup_frames: int = 5,
) -> tuple[np.ndarray, SceneDescription]:
    """
    Capture an image from camera and interpret it.

    Args:
        camera_index: Camera device index
        warmup_frames: Number of frames to skip for camera warmup

    Returns:
        Tuple of (image, scene_description)
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    try:
        # Warmup - discard initial frames
        for _ in range(warmup_frames):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        scene = interpret_scene(frame)
        return frame, scene

    finally:
        cap.release()


# CLI entry point
def main():
    """Command-line interface for scene interpretation."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Interpret a scene from camera or image file"
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Path to image file (instead of camera)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Save annotated image to this path"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of text"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color-based detection"
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="Disable contour-based detection"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum object area in pixels (default: 500)"
    )

    args = parser.parse_args()

    # Load or capture image
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            image, _ = capture_and_interpret(args.camera, warmup_frames=5)
            # Re-interpret with user options
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Interpret scene
    scene = interpret_scene(
        image,
        use_color_detection=not args.no_color,
        use_contour_detection=not args.no_contour,
        min_area=args.min_area,
    )

    # Output results
    if args.json:
        print(json.dumps(scene.to_dict(), indent=2))
    else:
        print(scene.summary)
        print()

        if scene.objects:
            print("Detected objects:")
            for i, obj in enumerate(scene.objects):
                color_str = f" ({obj.color_name})" if obj.color_name else ""
                print(f"  {i+1}. {obj.obj.label}{color_str}")
                print(f"      Position: {obj.position_description}")
                print(f"      Size: {obj.size_description}")
                print(f"      Distance: {obj.estimated_distance}")

        if scene.relationships:
            print("\nSpatial relationships:")
            for rel in scene.relationships:
                print(f"  - {rel.describe(scene.objects)}")

    # Save annotated image if requested
    if args.output:
        annotated = annotate_scene(image, scene, show_labels=True, show_relationships=True)
        cv2.imwrite(args.output, annotated)
        if not args.json:
            print(f"\nAnnotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
