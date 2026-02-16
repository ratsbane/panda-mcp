"""
SAWM data collector — records approach trajectories during pick_at().

Self-supervised: successful grasp position IS the ground truth label.
Every frame in a successful approach gets labeled with (final_grasp_xy - gripper_xy_at_frame).

Data layout:
    data/sawm_approaches/
    ├── approach_0000/
    │   ├── metadata.json     # target_xy, success, timestamps
    │   ├── frames/           # 224x224 progressive crops
    │   │   ├── 000.jpg
    │   │   └── ...
    │   └── labels.json       # [{crop_scale, dx, dy, gripper_xy}, ...]
    └── stats.json            # aggregate statistics
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .cropper import get_cropper

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "sawm_approaches"


class SAWMDataCollector:
    """Records progressive crops during pick approaches for SAWM training."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._active = False
        self._approach_dir: Optional[Path] = None
        self._target_xy: Optional[Tuple[float, float]] = None
        self._frames: list[dict] = []  # in-memory during approach
        self._t_start: float = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def _next_approach_id(self) -> int:
        """Find the next approach directory number."""
        existing = sorted(self.data_dir.glob("approach_*"))
        if not existing:
            return 0
        last = existing[-1].name.split("_")[1]
        return int(last) + 1

    def start_approach(self, target_robot_xy: Tuple[float, float]):
        """Begin recording a new approach trajectory."""
        if self._active:
            logger.warning("Approach already active, ending previous one")
            self.end_approach(success=False, final_gripper_xy=(0, 0))

        approach_id = self._next_approach_id()
        self._approach_dir = self.data_dir / f"approach_{approach_id:04d}"
        self._approach_dir.mkdir(parents=True)
        (self._approach_dir / "frames").mkdir()

        self._target_xy = target_robot_xy
        self._frames = []
        self._t_start = time.time()
        self._active = True

        logger.info(
            f"SAWM approach {approach_id} started — "
            f"target=({target_robot_xy[0]:.3f}, {target_robot_xy[1]:.3f})"
        )

    def record_frame(
        self,
        frame: np.ndarray,
        gripper_robot_xy: Tuple[float, float],
        gripper_z: Optional[float] = None,
        joints: Optional[np.ndarray] = None,
    ):
        """
        Record one frame during the approach.

        Args:
            frame: Full camera frame (BGR)
            gripper_robot_xy: Current EE position (x, y) in robot frame
            gripper_z: Current EE height (meters) — used for crop scale
            joints: Current joint positions (optional, for debugging)
        """
        if not self._active or self._target_xy is None:
            return

        cropper = get_cropper()

        try:
            crop, scale = cropper.compute_crop(
                frame, self._target_xy, gripper_robot_xy, gripper_z=gripper_z
            )
        except Exception as e:
            logger.warning(f"Crop computation failed: {e}")
            return

        # Save crop to disk
        idx = len(self._frames)
        crop_path = self._approach_dir / "frames" / f"{idx:03d}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Record metadata (labels computed after approach ends)
        self._frames.append({
            "idx": idx,
            "timestamp": round(time.time() - self._t_start, 3),
            "crop_scale": round(float(scale), 4),
            "gripper_x": round(float(gripper_robot_xy[0]), 5),
            "gripper_y": round(float(gripper_robot_xy[1]), 5),
            "joints": [round(float(j), 5) for j in joints] if joints is not None else None,
        })

    def end_approach(
        self,
        success: bool,
        final_gripper_xy: Tuple[float, float],
    ) -> dict:
        """
        End the approach and generate labels.

        For successful grasps, each frame gets labeled with the offset
        from its gripper position to the final grasp position.

        Args:
            success: Whether the grasp succeeded
            final_gripper_xy: Final EE (x, y) at moment of grasp

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
            dx = final_gripper_xy[0] - f["gripper_x"]
            dy = final_gripper_xy[1] - f["gripper_y"]
            labels.append({
                "idx": f["idx"],
                "crop_scale": f["crop_scale"],
                "dx": round(float(dx), 5),
                "dy": round(float(dy), 5),
                "gripper_x": f["gripper_x"],
                "gripper_y": f["gripper_y"],
            })

        # Save labels
        labels_path = self._approach_dir / "labels.json"
        with open(labels_path, "w") as fp:
            json.dump(labels, fp, indent=2)

        # Save metadata
        metadata = {
            "target_x": round(float(self._target_xy[0]), 5),
            "target_y": round(float(self._target_xy[1]), 5),
            "final_gripper_x": round(float(final_gripper_xy[0]), 5),
            "final_gripper_y": round(float(final_gripper_xy[1]), 5),
            "success": success,
            "num_frames": len(self._frames),
            "duration_s": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata_path = self._approach_dir / "metadata.json"
        with open(metadata_path, "w") as fp:
            json.dump(metadata, fp, indent=2)

        approach_name = self._approach_dir.name
        logger.info(
            f"SAWM {approach_name}: {'SUCCESS' if success else 'FAIL'}, "
            f"{len(self._frames)} frames, {duration:.1f}s"
        )

        # Update aggregate stats
        self._update_stats()

        # Reset
        self._approach_dir = None
        self._target_xy = None
        self._frames = []

        return {
            "approach": approach_name,
            "success": success,
            "frames": metadata["num_frames"],
            "duration_s": metadata["duration_s"],
        }

    def _update_stats(self):
        """Update aggregate stats file."""
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

        stats_path = self.data_dir / "stats.json"
        with open(stats_path, "w") as fp:
            json.dump(stats, fp, indent=2)

    def get_stats(self) -> dict:
        """Get current data collection statistics."""
        stats_path = self.data_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as fp:
                stats = json.load(fp)
        else:
            stats = {
                "total_approaches": 0,
                "successful": 0,
                "failed": 0,
                "total_frames": 0,
                "success_rate": 0,
            }

        stats["active"] = self._active
        stats["data_dir"] = str(self.data_dir)

        if self._active:
            stats["current_approach"] = {
                "target_xy": list(self._target_xy) if self._target_xy else None,
                "frames_so_far": len(self._frames),
                "duration_s": round(time.time() - self._t_start, 1),
            }

        return stats


# Singleton
_collector: Optional[SAWMDataCollector] = None


def get_collector() -> SAWMDataCollector:
    global _collector
    if _collector is None:
        _collector = SAWMDataCollector()
    return _collector
