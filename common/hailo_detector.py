"""
Hailo-accelerated YOLOv8 object detector.

Uses the Hailo-10H AI accelerator for fast YOLO inference (~27ms per frame).
Follows the lazy-singleton pattern used by MobileSAM segmentation.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .vision import DetectedObject, BoundingBox

logger = logging.getLogger(__name__)

# Default HEF model path (system-installed)
DEFAULT_MODEL_PATH = "/usr/share/hailo-models/yolov8m_h10.hef"

# YOLOv8 input size
MODEL_INPUT_SIZE = 640

# NMS output format: packed sequentially, max buffer = 80 classes × 501 = 40080 floats
NMS_MAX_DETECTIONS = 100
NMS_VALUES_PER_DET = 5  # y_min, x_min, y_max, x_max, score
NMS_VALUES_PER_CLASS = 1 + NMS_MAX_DETECTIONS * NMS_VALUES_PER_DET  # 501 (max per class)

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NUM_CLASSES = len(COCO_CLASSES)  # 80


class HailoYOLODetector:
    """
    YOLOv8 object detector using Hailo-10H accelerator.

    Uses InferModel API (required for Hailo-10H, InferVStreams not supported).
    VDevice is kept alive for process lifetime to avoid cleanup abort() issue.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, confidence_threshold: float = 0.3):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._vdevice = None
        self._infer_model = None
        self._configured_infer_model = None
        self._output_name = None
        self._loaded = False

    def load(self) -> bool:
        """Load the HEF model onto the Hailo device. Returns True if successful."""
        if self._loaded:
            return True

        try:
            from hailo_platform import HEF, VDevice, FormatType
        except ImportError:
            logger.warning("hailo_platform not available - Hailo detection disabled")
            return False

        model_path = Path(self._model_path)
        if not model_path.exists():
            logger.warning(f"HEF model not found: {model_path}")
            return False

        try:
            logger.info(f"Loading Hailo YOLOv8 model from {model_path}")
            self._vdevice = VDevice()
            self._infer_model = self._vdevice.create_infer_model(str(model_path))
            self._infer_model.output_format_type = FormatType.FLOAT32
            # Set NMS thresholds before configure() - lower score threshold
            # so detections aren't filtered out before we see them
            out = self._infer_model.output(self._infer_model.output_names[0])
            out.set_nms_score_threshold(0.2)
            out.set_nms_iou_threshold(0.45)
            self._configured_infer_model = self._infer_model.configure()
            self._output_name = self._infer_model.output_names[0]
            self._loaded = True
            logger.info("Hailo YOLOv8 model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Hailo model: {e}")
            self._vdevice = None
            self._infer_model = None
            self._configured_infer_model = None
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Letterbox resize BGR image to 640x640 for YOLO input.

        Returns:
            Tuple of (padded_image_rgb_uint8, transform_info)
        """
        orig_h, orig_w = image.shape[:2]

        # Compute scale to fit in 640x640 maintaining aspect ratio
        scale = min(MODEL_INPUT_SIZE / orig_w, MODEL_INPUT_SIZE / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR -> RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Pad to 640x640 (gray padding)
        pad_w = MODEL_INPUT_SIZE - new_w
        pad_h = MODEL_INPUT_SIZE - new_h
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        padded = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_rgb

        transform = {
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }

        return padded, transform

    def _parse_output(self, output_buffer: np.ndarray, transform: dict) -> list[DetectedObject]:
        """
        Parse packed NMS output from Hailo YOLOv8.

        Format: HAILO_NMS_BY_CLASS, packed sequentially (NOT fixed 501-stride).
        For each of 80 classes: [count, y_min, x_min, y_max, x_max, score, ...]
        Detections for each class are packed immediately after the count.
        Coordinates are normalized 0-1 relative to 640x640 input.
        Total buffer is 40080 floats (max capacity), actual data is packed at start.
        """
        buf = output_buffer.flatten()
        detections = []

        orig_w = transform["orig_w"]
        orig_h = transform["orig_h"]
        scale = transform["scale"]
        pad_left = transform["pad_left"]
        pad_top = transform["pad_top"]

        offset = 0
        for cls_id in range(NUM_CLASSES):
            if offset >= len(buf):
                break

            count = int(buf[offset])
            offset += 1

            if count <= 0:
                continue

            count = min(count, NMS_MAX_DETECTIONS)

            for i in range(count):
                if offset + NMS_VALUES_PER_DET > len(buf):
                    break

                y1_norm = buf[offset + 0]
                x1_norm = buf[offset + 1]
                y2_norm = buf[offset + 2]
                x2_norm = buf[offset + 3]
                score = buf[offset + 4]
                offset += NMS_VALUES_PER_DET

                if score < self._confidence_threshold:
                    continue

                # Convert normalized coords to 640x640 pixel coords
                x1_px = x1_norm * MODEL_INPUT_SIZE
                y1_px = y1_norm * MODEL_INPUT_SIZE
                x2_px = x2_norm * MODEL_INPUT_SIZE
                y2_px = y2_norm * MODEL_INPUT_SIZE

                # Remove letterbox padding
                x1_px -= pad_left
                y1_px -= pad_top
                x2_px -= pad_left
                y2_px -= pad_top

                # Scale back to original image coordinates
                x1_orig = x1_px / scale
                y1_orig = y1_px / scale
                x2_orig = x2_px / scale
                y2_orig = y2_px / scale

                # Clamp to image bounds
                x1_orig = max(0, min(orig_w, x1_orig))
                y1_orig = max(0, min(orig_h, y1_orig))
                x2_orig = max(0, min(orig_w, x2_orig))
                y2_orig = max(0, min(orig_h, y2_orig))

                w = x2_orig - x1_orig
                h = y2_orig - y1_orig
                if w < 1 or h < 1:
                    continue

                detections.append(DetectedObject(
                    bbox=BoundingBox(
                        x=int(x1_orig),
                        y=int(y1_orig),
                        width=int(w),
                        height=int(h),
                    ),
                    confidence=float(score),
                    label=COCO_CLASSES[cls_id],
                ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> list[DetectedObject]:
        """
        Run YOLOv8 detection on a BGR image.

        Args:
            image: BGR image from OpenCV
            confidence_threshold: Override default threshold (0.3)

        Returns:
            List of DetectedObjects with COCO class labels
        """
        if not self._loaded:
            if not self.load():
                return []

        old_threshold = self._confidence_threshold
        if confidence_threshold is not None:
            self._confidence_threshold = confidence_threshold

        try:
            # Preprocess
            padded, transform = self._preprocess(image)

            # Create inference bindings with pre-allocated output buffer
            bindings = self._configured_infer_model.create_bindings()
            bindings.input().set_buffer(padded)
            output_buffer = np.empty(NUM_CLASSES * NMS_VALUES_PER_CLASS, dtype=np.float32)
            bindings.output(self._output_name).set_buffer(output_buffer)

            # Run synchronous inference (timeout in ms)
            self._configured_infer_model.run([bindings], 10000)

            # Parse detections from the pre-allocated buffer (written in-place)
            detections = self._parse_output(output_buffer, transform)

            # Add color info from original image
            _add_color_to_detections(detections, image)

            return detections

        except Exception as e:
            logger.error(f"Hailo inference failed: {e}")
            return []

        finally:
            self._confidence_threshold = old_threshold


def _add_color_to_detections(detections: list[DetectedObject], image: np.ndarray) -> None:
    """Sample average BGR color from each detection's bounding box ROI."""
    h, w = image.shape[:2]
    for det in detections:
        bbox = det.bbox
        # Clamp ROI to image bounds
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2]
        avg_color = cv2.mean(roi)[:3]
        det.color_bgr = tuple(int(c) for c in avg_color)


# Singleton detector (lazy-loaded)
_detector: Optional[HailoYOLODetector] = None


def get_hailo_detector() -> Optional[HailoYOLODetector]:
    """
    Get the singleton Hailo YOLO detector.

    Returns None if Hailo hardware/drivers are unavailable.
    """
    global _detector
    if _detector is None:
        _detector = HailoYOLODetector()
        if not _detector.load():
            logger.info("Hailo YOLO detector not available, falling back to other methods")
            return None
    return _detector
