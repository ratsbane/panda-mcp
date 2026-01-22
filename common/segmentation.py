"""
MobileSAM segmentation module.

Provides object segmentation using MobileSAM with ONNX Runtime inference.
Designed for efficient CPU inference on Raspberry Pi.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np

from .vision import DetectedObject, BoundingBox

logger = logging.getLogger(__name__)

# Default model paths (relative to project root)
DEFAULT_ENCODER_PATH = "models/mobile_sam_encoder.onnx"
DEFAULT_DECODER_PATH = "models/mobile_sam_decoder.onnx"

# Model input size for MobileSAM
MODEL_INPUT_SIZE = 1024


@dataclass
class SegmentationConfig:
    """Configuration for segmentation."""
    encoder_path: str = DEFAULT_ENCODER_PATH
    decoder_path: str = DEFAULT_DECODER_PATH
    points_per_side: int = 16  # Grid density for automatic mask generation
    pred_iou_thresh: float = 0.7  # Minimum predicted IoU to keep mask
    stability_score_thresh: float = 0.85  # Minimum stability score
    min_mask_area: int = 500  # Minimum mask area in pixels
    max_masks: int = 20  # Maximum number of masks to return


class MobileSAMSegmenter:
    """
    MobileSAM segmentation using ONNX Runtime.

    Uses separate encoder and decoder models for flexibility.
    Supports automatic mask generation via point grid prompting.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        self._encoder = None
        self._decoder = None
        self._loaded = False

        # Find project root (look for common/ directory)
        self._project_root = Path(__file__).parent.parent

    def _resolve_path(self, path: str) -> Path:
        """Resolve model path relative to project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self._project_root / path

    def load(self) -> bool:
        """Load the ONNX models. Returns True if successful."""
        if self._loaded:
            return True

        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            return False

        encoder_path = self._resolve_path(self.config.encoder_path)
        decoder_path = self._resolve_path(self.config.decoder_path)

        if not encoder_path.exists():
            logger.warning(f"Encoder model not found: {encoder_path}")
            logger.info("Download MobileSAM models using: python -m common.segmentation --download")
            return False

        if not decoder_path.exists():
            logger.warning(f"Decoder model not found: {decoder_path}")
            return False

        try:
            # Use CPU execution provider for Raspberry Pi
            providers = ['CPUExecutionProvider']

            logger.info(f"Loading MobileSAM encoder from {encoder_path}")
            self._encoder = ort.InferenceSession(str(encoder_path), providers=providers)

            logger.info(f"Loading MobileSAM decoder from {decoder_path}")
            self._decoder = ort.InferenceSession(str(decoder_path), providers=providers)

            self._loaded = True
            logger.info("MobileSAM models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load MobileSAM models: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Preprocess image for MobileSAM encoder.

        Args:
            image: BGR image from OpenCV

        Returns:
            Tuple of (preprocessed tensor, transform info for unscaling)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get original dimensions
        orig_h, orig_w = image_rgb.shape[:2]

        # Resize to model input size (maintain aspect ratio, pad to square)
        scale = MODEL_INPUT_SIZE / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        padded = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize (ImageNet stats used by MobileSAM)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        normalized = (padded.astype(np.float32) - mean) / std

        # Keep HWC format - this encoder expects (H, W, 3) not (1, 3, H, W)
        tensor = normalized.astype(np.float32)

        transform_info = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": new_h,
            "new_w": new_w,
            "scale": scale,
        }

        return tensor, transform_info

    def _generate_point_grid(self, h: int, w: int) -> np.ndarray:
        """Generate a grid of point prompts."""
        n = self.config.points_per_side
        offset = 1 / (2 * n)
        points_one_side = np.linspace(offset, 1 - offset, n)
        points_x = points_one_side * w
        points_y = points_one_side * h
        points = np.stack(np.meshgrid(points_x, points_y), axis=-1).reshape(-1, 2)
        return points.astype(np.float32)

    def _postprocess_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        transform_info: dict,
    ) -> list[tuple[np.ndarray, float]]:
        """
        Postprocess decoder output masks.

        Args:
            masks: Raw mask outputs (N, 1, H, W)
            scores: Predicted IoU scores (N,)
            transform_info: Transform info from preprocessing

        Returns:
            List of (mask, score) tuples, filtered and resized to original dimensions
        """
        orig_h = transform_info["orig_h"]
        orig_w = transform_info["orig_w"]
        new_h = transform_info["new_h"]
        new_w = transform_info["new_w"]

        results = []

        for mask, score in zip(masks, scores):
            # Filter by score
            if score < self.config.pred_iou_thresh:
                continue

            # Remove batch and channel dims if present
            if mask.ndim == 4:
                mask = mask[0, 0]
            elif mask.ndim == 3:
                mask = mask[0]

            # Crop to actual image area (remove padding)
            mask = mask[:new_h, :new_w]

            # Resize to original dimensions
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )

            # Threshold to binary
            mask_binary = mask_resized > 0.5

            # Filter by area
            area = np.sum(mask_binary)
            if area < self.config.min_mask_area:
                continue

            results.append((mask_binary, float(score)))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Limit number of masks
        return results[:self.config.max_masks]

    def _compute_stability_score(self, mask: np.ndarray, mask_logits: np.ndarray) -> float:
        """Compute mask stability score."""
        # Stability = IoU between mask at threshold 0 and mask at threshold 0.5
        mask_high = mask_logits > 0.5
        mask_low = mask_logits > 0.0

        intersection = np.sum(mask_high & mask_low)
        union = np.sum(mask_high | mask_low)

        if union == 0:
            return 0.0
        return intersection / union

    def _nms_masks(
        self,
        masks: list[tuple[np.ndarray, float]],
        iou_threshold: float = 0.5,
    ) -> list[tuple[np.ndarray, float]]:
        """Apply non-maximum suppression to remove overlapping masks."""
        if not masks:
            return []

        keep = []
        suppressed = set()

        for i, (mask_i, score_i) in enumerate(masks):
            if i in suppressed:
                continue

            keep.append((mask_i, score_i))

            # Suppress overlapping masks with lower scores
            for j in range(i + 1, len(masks)):
                if j in suppressed:
                    continue

                mask_j, _ = masks[j]

                # Compute IoU
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    suppressed.add(j)

        return keep

    def segment(self, image: np.ndarray) -> list[DetectedObject]:
        """
        Segment objects in the image using automatic mask generation.

        Args:
            image: BGR image from OpenCV

        Returns:
            List of DetectedObjects with segmentation masks
        """
        if not self._loaded:
            if not self.load():
                logger.warning("MobileSAM not loaded, returning empty list")
                return []

        # Preprocess
        input_tensor, transform_info = self._preprocess_image(image)

        # Run encoder
        encoder_inputs = {self._encoder.get_inputs()[0].name: input_tensor}
        image_embedding = self._encoder.run(None, encoder_inputs)[0]

        # Generate point grid
        points = self._generate_point_grid(
            transform_info["new_h"],
            transform_info["new_w"]
        )

        # Run decoder for each point
        all_masks = []

        for point in points:
            # Prepare decoder inputs
            point_coords = np.array([[point]], dtype=np.float32)  # (1, 1, 2)
            point_labels = np.array([[1]], dtype=np.float32)  # 1 = foreground point

            decoder_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.array([0], dtype=np.float32),
                "orig_im_size": np.array([transform_info["new_h"], transform_info["new_w"]], dtype=np.float32),
            }

            try:
                outputs = self._decoder.run(None, decoder_inputs)
                masks = outputs[0]  # (1, num_masks, H, W)
                scores = outputs[1]  # (1, num_masks)

                # Take best mask for this point
                best_idx = np.argmax(scores[0])
                mask = masks[0, best_idx:best_idx+1]
                score = scores[0, best_idx]

                all_masks.append((mask, score))

            except Exception as e:
                logger.debug(f"Decoder failed for point {point}: {e}")
                continue

        # Postprocess and filter
        processed_masks = []
        for mask, score in all_masks:
            result = self._postprocess_masks(
                mask[np.newaxis, ...],
                np.array([score]),
                transform_info
            )
            processed_masks.extend(result)

        # NMS to remove duplicates
        final_masks = self._nms_masks(processed_masks, iou_threshold=0.5)

        # Convert to DetectedObjects
        objects = []
        for i, (mask, score) in enumerate(final_masks):
            # Get bounding box from mask
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))

            bbox = BoundingBox(
                x=x_min,
                y=y_min,
                width=x_max - x_min,
                height=y_max - y_min,
            )

            # Get average color from masked region
            color_bgr = None
            if np.any(mask):
                masked_pixels = image[mask]
                if len(masked_pixels) > 0:
                    avg_color = np.mean(masked_pixels, axis=0).astype(int)
                    color_bgr = tuple(avg_color)

            objects.append(DetectedObject(
                bbox=bbox,
                confidence=score,
                label="object",
                color_bgr=color_bgr,
                mask=mask,
            ))

        logger.info(f"MobileSAM found {len(objects)} objects")
        return objects

    def segment_with_prompt(
        self,
        image: np.ndarray,
        point: Optional[tuple[int, int]] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[DetectedObject]:
        """
        Segment a specific object using a point or box prompt.

        Args:
            image: BGR image
            point: (x, y) foreground point prompt
            box: (x1, y1, x2, y2) bounding box prompt

        Returns:
            DetectedObject with mask, or None if segmentation failed
        """
        if not self._loaded:
            if not self.load():
                return None

        if point is None and box is None:
            logger.warning("Must provide either point or box prompt")
            return None

        # Preprocess
        input_tensor, transform_info = self._preprocess_image(image)
        scale = transform_info["scale"]

        # Run encoder
        encoder_inputs = {self._encoder.get_inputs()[0].name: input_tensor}
        image_embedding = self._encoder.run(None, encoder_inputs)[0]

        # Prepare prompts
        if point is not None:
            point_coords = np.array([[[point[0] * scale, point[1] * scale]]], dtype=np.float32)
            point_labels = np.array([[1]], dtype=np.float32)
        else:
            # Use box corners as prompts
            x1, y1, x2, y2 = box
            point_coords = np.array([[[x1 * scale, y1 * scale], [x2 * scale, y2 * scale]]], dtype=np.float32)
            point_labels = np.array([[2, 3]], dtype=np.float32)  # 2=top-left, 3=bottom-right

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
            "orig_im_size": np.array([transform_info["new_h"], transform_info["new_w"]], dtype=np.float32),
        }

        try:
            outputs = self._decoder.run(None, decoder_inputs)
            masks = outputs[0]
            scores = outputs[1]

            # Take best mask
            best_idx = np.argmax(scores[0])
            mask = masks[0, best_idx:best_idx+1]
            score = scores[0, best_idx]

            # Postprocess
            results = self._postprocess_masks(
                mask[np.newaxis, ...],
                np.array([score]),
                transform_info
            )

            if not results:
                return None

            mask_binary, final_score = results[0]

            # Get bounding box
            ys, xs = np.where(mask_binary)
            if len(xs) == 0:
                return None

            bbox = BoundingBox(
                x=int(np.min(xs)),
                y=int(np.min(ys)),
                width=int(np.max(xs) - np.min(xs)),
                height=int(np.max(ys) - np.min(ys)),
            )

            # Get color
            masked_pixels = image[mask_binary]
            color_bgr = tuple(np.mean(masked_pixels, axis=0).astype(int)) if len(masked_pixels) > 0 else None

            return DetectedObject(
                bbox=bbox,
                confidence=final_score,
                label="object",
                color_bgr=color_bgr,
                mask=mask_binary,
            )

        except Exception as e:
            logger.error(f"Segmentation with prompt failed: {e}")
            return None


# Singleton segmenter (lazy-loaded)
_segmenter: Optional[MobileSAMSegmenter] = None


def get_segmenter(config: Optional[SegmentationConfig] = None) -> MobileSAMSegmenter:
    """Get the singleton segmenter instance."""
    global _segmenter
    if _segmenter is None:
        _segmenter = MobileSAMSegmenter(config)
    return _segmenter


def download_models(output_dir: str = "models") -> bool:
    """
    Download MobileSAM ONNX models from HuggingFace.

    Downloads from Acly/MobileSAM repository.
    """
    import urllib.request
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # MobileSAM model URLs from Acly/MobileSAM HuggingFace repo
    ENCODER_URL = "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx"
    DECODER_URL = "https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx"

    encoder_path = output_path / "mobile_sam_encoder.onnx"
    decoder_path = output_path / "mobile_sam_decoder.onnx"

    def download_with_progress(url: str, dest: Path):
        """Download with progress indicator."""
        print(f"Downloading {dest.name}...")

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                print(f"\r  {percent}% ({count * block_size // 1024 // 1024}MB)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook)
        print()  # newline after progress

    try:
        if not encoder_path.exists():
            download_with_progress(ENCODER_URL, encoder_path)
            print(f"  Encoder saved to {encoder_path}")
        else:
            print(f"Encoder already exists: {encoder_path}")

        if not decoder_path.exists():
            download_with_progress(DECODER_URL, decoder_path)
            print(f"  Decoder saved to {decoder_path}")
        else:
            print(f"Decoder already exists: {decoder_path}")

        print("\nMobileSAM models ready!")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit https://huggingface.co/Acly/MobileSAM")
        print("2. Download mobile_sam_image_encoder.onnx -> models/mobile_sam_encoder.onnx")
        print("3. Download sam_mask_decoder_single.onnx -> models/mobile_sam_decoder.onnx")
        return False


def main():
    """CLI for segmentation module."""
    import argparse

    parser = argparse.ArgumentParser(description="MobileSAM segmentation")
    parser.add_argument("--download", action="store_true", help="Download ONNX models")
    parser.add_argument("--test", type=str, help="Test segmentation on an image")
    parser.add_argument("--output", type=str, help="Output path for annotated image")

    args = parser.parse_args()

    if args.download:
        download_models()
        return

    if args.test:
        image = cv2.imread(args.test)
        if image is None:
            print(f"Could not load image: {args.test}")
            return

        segmenter = MobileSAMSegmenter()
        if not segmenter.load():
            print("Failed to load models. Try: python -m common.segmentation --download")
            return

        print("Running segmentation...")
        objects = segmenter.segment(image)
        print(f"Found {len(objects)} objects")

        for i, obj in enumerate(objects):
            print(f"  {i+1}. {obj.label} (confidence: {obj.confidence:.2f}, area: {obj.bbox.area})")

        if args.output and objects:
            # Draw masks
            result = image.copy()
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
            ]
            for i, obj in enumerate(objects):
                if obj.mask is not None:
                    color = colors[i % len(colors)]
                    overlay = result.copy()
                    overlay[obj.mask] = color
                    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

                    # Draw bbox
                    cv2.rectangle(
                        result,
                        (obj.bbox.x, obj.bbox.y),
                        (obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height),
                        color,
                        2
                    )

            cv2.imwrite(args.output, result)
            print(f"Saved annotated image to: {args.output}")


if __name__ == "__main__":
    main()
