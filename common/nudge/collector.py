"""
NUDGE data collector -- records frames + positions during pick approaches.

Self-supervised: successful grasp position IS the ground truth. Each frame
gets labeled with the discretized offset from current gripper position to
the final successful grasp position.

Data layout:
    data/nudge_approaches/
    ├── approach_0000/
    │   ├── metadata.json   # {target_bbox, target_type, success, final_xyz, timestamp}
    │   ├── labels.json     # [{frame_idx, gripper_xyz, target_bbox_px, dx/dy/dz_class, ...}]
    │   └── frames/
    │       ├── 000.jpg     # full frame resized to 224x224
    │       └── ...
    └── stats.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .discretize import continuous_to_class, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "nudge_approaches"


class NUDGECollector:
    """Records frames and positions during pick approaches for NUDGE training."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._active = False
        self._approach_dir: Optional[Path] = None
        self._target_bbox: Optional[List[float]] = None  # [x1,y1,x2,y2] in pixel coords
        self._target_type: Optional[str] = None
        self._target_id: Optional[str] = None
        self._frames: list[dict] = []
        self._t_start: float = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def _next_approach_id(self) -> int:
        existing = sorted(self.data_dir.glob("approach_*"))
        if not existing:
            return 0
        last = existing[-1].name.split("_")[1]
        return int(last) + 1

    def start_approach(
        self,
        target_bbox_pixels: List[float],
        target_type: str = "block",
        target_id: str = "",
    ):
        """
        Begin recording a new approach.

        Args:
            target_bbox_pixels: [x1, y1, x2, y2] in camera pixel coordinates
            target_type: e.g. "block", "cup", "wire"
            target_id: e.g. "red_block", "green_block"
        """
        if self._active:
            logger.warning("Approach already active, ending previous one")
            self.end_approach(success=False, final_gripper_xyz=(0, 0, 0))

        approach_id = self._next_approach_id()
        self._approach_dir = self.data_dir / f"approach_{approach_id:04d}"
        self._approach_dir.mkdir(parents=True)
        (self._approach_dir / "frames").mkdir()

        self._target_bbox = list(target_bbox_pixels)
        self._target_type = target_type
        self._target_id = target_id
        self._frames = []
        self._t_start = time.time()
        self._active = True

        logger.info(
            f"NUDGE approach {approach_id} started — "
            f"bbox={[round(b, 1) for b in target_bbox_pixels]}, type={target_type}"
        )

    def record_frame(
        self,
        frame: np.ndarray,
        gripper_xyz: Tuple[float, float, float],
        target_bbox_px: Optional[List[float]] = None,
    ):
        """
        Record one frame during approach.

        Args:
            frame: Full camera frame (BGR, any size -- will be resized to 224x224)
            gripper_xyz: Current EE position (x, y, z) in robot frame (meters)
            target_bbox_px: Updated target bbox [x1,y1,x2,y2] in camera pixels.
                           Uses initial bbox if None.
        """
        if not self._active:
            return

        bbox = target_bbox_px if target_bbox_px is not None else self._target_bbox

        # Resize frame to 224x224 and scale bbox accordingly
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (224, 224))
        sx, sy = 224.0 / w, 224.0 / h
        bbox_224 = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]

        # Save resized frame
        idx = len(self._frames)
        frame_path = self._approach_dir / "frames" / f"{idx:03d}.jpg"
        cv2.imwrite(str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self._frames.append({
            "frame_idx": idx,
            "timestamp": round(time.time() - self._t_start, 3),
            "gripper_x": round(float(gripper_xyz[0]), 5),
            "gripper_y": round(float(gripper_xyz[1]), 5),
            "gripper_z": round(float(gripper_xyz[2]), 5),
            "target_bbox_px": [round(b, 1) for b in bbox_224],
        })

    def end_approach(
        self,
        success: bool,
        final_gripper_xyz: Tuple[float, float, float],
    ) -> dict:
        """
        End approach and generate discrete labels.

        For successful grasps, each frame gets labeled with the discretized
        offset from its gripper position to the final grasp position.

        Args:
            success: Whether the grasp succeeded
            final_gripper_xyz: Final EE (x, y, z) at moment of grasp

        Returns:
            Summary dict
        """
        if not self._active:
            return {"error": "No active approach"}

        self._active = False
        duration = time.time() - self._t_start

        # Generate labels
        labels = []
        for f in self._frames:
            dx_m = final_gripper_xyz[0] - f["gripper_x"]
            dy_m = final_gripper_xyz[1] - f["gripper_y"]
            dz_m = final_gripper_xyz[2] - f["gripper_z"]

            labels.append({
                "frame_idx": f["frame_idx"],
                "gripper_x": f["gripper_x"],
                "gripper_y": f["gripper_y"],
                "gripper_z": f["gripper_z"],
                "target_bbox_px": f["target_bbox_px"],
                "dx_m": round(float(dx_m), 5),
                "dy_m": round(float(dy_m), 5),
                "dz_m": round(float(dz_m), 5),
                "dx_class": continuous_to_class(dx_m, axis="x"),
                "dy_class": continuous_to_class(dy_m, axis="y"),
                "dz_class": continuous_to_class(dz_m, axis="z"),
            })

        # Save labels
        with open(self._approach_dir / "labels.json", "w") as fp:
            json.dump(labels, fp, indent=2)

        # Save metadata
        metadata = {
            "target_bbox": self._target_bbox,
            "target_type": self._target_type,
            "target_id": self._target_id,
            "final_gripper_x": round(float(final_gripper_xyz[0]), 5),
            "final_gripper_y": round(float(final_gripper_xyz[1]), 5),
            "final_gripper_z": round(float(final_gripper_xyz[2]), 5),
            "success": success,
            "num_frames": len(self._frames),
            "duration_s": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._approach_dir / "metadata.json", "w") as fp:
            json.dump(metadata, fp, indent=2)

        approach_name = self._approach_dir.name
        logger.info(
            f"NUDGE {approach_name}: {'SUCCESS' if success else 'FAIL'}, "
            f"{len(self._frames)} frames, {duration:.1f}s"
        )

        self._update_stats()

        # Reset
        self._approach_dir = None
        self._target_bbox = None
        self._target_type = None
        self._target_id = None
        self._frames = []

        return {
            "approach": approach_name,
            "success": success,
            "frames": metadata["num_frames"],
            "duration_s": metadata["duration_s"],
        }

    def _update_stats(self):
        approaches = sorted(self.data_dir.glob("approach_*"))
        total = len(approaches)
        successful = 0
        total_frames = 0

        for a in approaches:
            meta_path = a / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as fp:
                    meta = json.load(fp)
                if meta.get("success"):
                    successful += 1
                total_frames += meta.get("num_frames", 0)

        stats = {
            "total_approaches": total,
            "successful": successful,
            "failed": total - successful,
            "total_frames": total_frames,
            "success_rate": round(successful / total, 3) if total > 0 else 0,
        }
        with open(self.data_dir / "stats.json", "w") as fp:
            json.dump(stats, fp, indent=2)

    def get_stats(self) -> dict:
        stats_path = self.data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as fp:
                stats = json.load(fp)
        else:
            stats = {
                "total_approaches": 0, "successful": 0, "failed": 0,
                "total_frames": 0, "success_rate": 0,
            }

        stats["active"] = self._active
        stats["data_dir"] = str(self.data_dir)

        if self._active:
            stats["current_approach"] = {
                "target_bbox": self._target_bbox,
                "target_type": self._target_type,
                "frames_so_far": len(self._frames),
                "duration_s": round(time.time() - self._t_start, 1),
            }

        return stats


# Singleton
_nudge_collector: Optional[NUDGECollector] = None


def get_nudge_collector() -> NUDGECollector:
    global _nudge_collector
    if _nudge_collector is None:
        _nudge_collector = NUDGECollector()
    return _nudge_collector
