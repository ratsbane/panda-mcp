"""
Skill-level data logger for Hi-Robot VLM training.

Records (camera_frame, task, skill_call, result) tuples during Claude's
normal pick-and-place operations. Each episode captures a sequence of
skill invocations with the camera frame taken BEFORE each skill executes
(this is what the VLM will see at inference time).

Episode format on disk:
    data/skill_episodes/
    ├── episode_0000/
    │   ├── episode.json          # metadata + skill sequence
    │   └── images/
    │       ├── step_000.jpg      # frame BEFORE skill 0
    │       ├── step_001.jpg      # frame BEFORE skill 1
    │       └── step_NNN.jpg      # final frame (for "done" step)
    └── dataset_info.json         # summary across all episodes
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SkillLogger:
    """Records skill-level episodes for VLM training data."""

    def __init__(self, data_dir: str = "./data/skill_episodes"):
        self.data_dir = Path(data_dir)
        self._camera = None
        self._camera_available = False

        # Per-episode state
        self._active = False
        self._episode_index: int = -1
        self._episode_dir: Optional[Path] = None
        self._task: str = ""
        self._steps: list[dict] = []
        self._start_time: float = 0.0

    def _get_next_episode_index(self) -> int:
        """Determine next episode index from existing directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.data_dir.glob("episode_*"))
        if not existing:
            return 0
        try:
            last = existing[-1].name
            return int(last.split("_")[1]) + 1
        except (ValueError, IndexError):
            return len(existing)

    def _init_camera(self):
        """Lazily initialize camera client."""
        if self._camera is not None:
            return

        try:
            from camera_daemon.client import CameraClient
            self._camera = CameraClient()
            if self._camera.connect():
                self._camera_available = True
                logger.info("SkillLogger: camera connected")
            else:
                logger.warning("SkillLogger: camera connection failed")
                self._camera_available = False
        except Exception as e:
            logger.warning(f"SkillLogger: camera init failed: {e}")
            self._camera_available = False

    def _capture_frame(self, save_path: Path) -> bool:
        """Capture a camera frame and save as JPEG."""
        self._init_camera()

        if not self._camera_available:
            logger.warning("SkillLogger: no camera, skipping frame capture")
            return False

        try:
            frame = self._camera.get_frame()
            if frame is None:
                logger.warning("SkillLogger: got None frame")
                return False
            cv2.imwrite(str(save_path), frame)
            return True
        except Exception as e:
            logger.warning(f"SkillLogger: frame capture failed: {e}")
            return False

    @property
    def active(self) -> bool:
        return self._active

    def start_episode(self, task: str) -> dict:
        """
        Start a new skill episode.

        Args:
            task: Natural language task description

        Returns:
            dict with episode_index and status
        """
        if self._active:
            return {"success": False, "error": "Episode already active"}

        self._episode_index = self._get_next_episode_index()
        self._episode_dir = self.data_dir / f"episode_{self._episode_index:04d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        (self._episode_dir / "images").mkdir(exist_ok=True)

        self._task = task
        self._steps = []
        self._start_time = time.time()
        self._active = True

        logger.info(f"SkillLogger: started episode {self._episode_index} — {task}")
        return {
            "success": True,
            "episode_index": self._episode_index,
            "task": task,
            "data_dir": str(self._episode_dir),
        }

    def log_skill(self, skill: str, params: dict, result: Optional[dict] = None) -> dict:
        """
        Log a single skill invocation.

        Captures a camera frame BEFORE the skill executes,
        then records skill name, params, and (optionally) result.

        Call with result=None before execution, then call update_last_result()
        after execution completes. Or pass result directly if available.

        Args:
            skill: Skill name (pick, place, move, open_gripper, grasp, home, done, etc.)
            params: Skill parameters dict
            result: Skill execution result (optional, can update later)

        Returns:
            dict with step_index and image status
        """
        if not self._active:
            return {"success": False, "error": "No active episode"}

        step_index = len(self._steps)
        image_name = f"step_{step_index:03d}.jpg"
        image_path = self._episode_dir / "images" / image_name
        timestamp = time.time() - self._start_time

        # Capture frame BEFORE skill execution
        has_image = self._capture_frame(image_path)

        step = {
            "step_index": step_index,
            "image_path": f"images/{image_name}",
            "has_image": has_image,
            "timestamp": round(timestamp, 3),
            "skill": skill,
            "params": params,
            "result": result,
        }
        self._steps.append(step)

        logger.info(f"SkillLogger: step {step_index} — {skill}({params})")
        return {
            "success": True,
            "step_index": step_index,
            "has_image": has_image,
        }

    def update_last_result(self, result: dict) -> None:
        """Update the result of the most recently logged skill."""
        if self._steps:
            self._steps[-1]["result"] = result

    def end_episode(self, success: bool) -> dict:
        """
        End the current episode.

        Captures a final frame and logs a "done" step, then writes
        episode.json and updates dataset_info.json.

        Args:
            success: Whether the episode was successful

        Returns:
            dict with episode stats
        """
        if not self._active:
            return {"success": False, "error": "No active episode"}

        # Log final "done" step with a fresh frame
        self.log_skill("done", {}, {"success": success})

        end_time = time.time()
        duration = end_time - self._start_time

        # Write episode.json
        episode_data = {
            "episode_index": self._episode_index,
            "task": self._task,
            "success": success,
            "num_steps": len(self._steps),
            "start_time": self._start_time,
            "end_time": end_time,
            "duration_s": round(duration, 1),
            "steps": self._steps,
        }

        episode_path = self._episode_dir / "episode.json"
        with open(episode_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        # Update dataset_info.json
        self._update_dataset_info()

        self._active = False
        logger.info(
            f"SkillLogger: ended episode {self._episode_index} — "
            f"{len(self._steps)} steps, {duration:.1f}s, success={success}"
        )

        return {
            "success": True,
            "episode_index": self._episode_index,
            "num_steps": len(self._steps),
            "duration_s": round(duration, 1),
            "episode_success": success,
            "path": str(episode_path),
        }

    def _update_dataset_info(self):
        """Update the dataset_info.json summary file."""
        info_path = self.data_dir / "dataset_info.json"

        # Count episodes and stats
        episodes = []
        total_steps = 0
        successful = 0

        for ep_dir in sorted(self.data_dir.glob("episode_*")):
            ep_json = ep_dir / "episode.json"
            if ep_json.exists():
                try:
                    with open(ep_json) as f:
                        ep = json.load(f)
                    episodes.append({
                        "index": ep["episode_index"],
                        "task": ep["task"],
                        "success": ep["success"],
                        "num_steps": ep["num_steps"],
                        "duration_s": ep.get("duration_s", 0),
                    })
                    total_steps += ep["num_steps"]
                    if ep["success"]:
                        successful += 1
                except Exception as e:
                    logger.warning(f"Failed to read {ep_json}: {e}")

        info = {
            "total_episodes": len(episodes),
            "successful_episodes": successful,
            "total_steps": total_steps,
            "episodes": episodes,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def get_status(self) -> dict:
        """Get current logging status."""
        result = {"active": self._active}

        if self._active:
            result["episode_index"] = self._episode_index
            result["task"] = self._task
            result["steps_logged"] = len(self._steps)
            result["elapsed_s"] = round(time.time() - self._start_time, 1)

        return result

    def list_episodes(self) -> dict:
        """List all collected skill episodes."""
        info_path = self.data_dir / "dataset_info.json"

        if info_path.exists():
            try:
                with open(info_path) as f:
                    return json.load(f)
            except Exception:
                pass

        # Rebuild from disk
        self._update_dataset_info()
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)

        return {"total_episodes": 0, "successful_episodes": 0, "total_steps": 0, "episodes": []}
