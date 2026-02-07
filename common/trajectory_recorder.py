"""
Trajectory recorder for VLA training data collection.

Records robot joint states + camera frames at ~30Hz during teleoperated
demonstrations. Saves to JSON + JPEG files for later conversion to
LeRobot dataset format.

Usage:
    recorder = TrajectoryRecorder()
    recorder.start_episode("pick up the red block", controller)
    # ... execute pick_at / place_at ...
    stats = recorder.stop_episode(success=True)
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """
    Records robot trajectories as training demonstrations.

    Background thread samples robot state + camera at target FPS.
    Data is buffered in memory during recording, then flushed to disk
    as JSON metadata + JPEG image files on stop.

    Episode format:
        data/recordings/
        ├── episode_0000/
        │   ├── episode.json    # metadata + per-frame state
        │   └── images/
        │       ├── 000000.jpg
        │       └── ...
        └── dataset_info.json   # summary across all episodes
    """

    def __init__(self, data_dir: str = "./data/recordings", fps: int = 30):
        self.data_dir = Path(data_dir)
        self.fps = fps

        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Camera client (lazy-init)
        self._camera = None
        self._camera_available = False

        # Per-episode state
        self._controller = None
        self._episode_frames: list[tuple[float, dict, Optional[bytes]]] = []
        self._episode_meta: dict = {}
        self._episode_index: int = 0

    def _get_next_episode_index(self) -> int:
        """Determine next episode index from existing directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.data_dir.glob("episode_*"))
        if not existing:
            return 0
        # Parse highest index
        try:
            last = existing[-1].name  # e.g. "episode_0042"
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
                logger.info("Camera connected for trajectory recording")
            else:
                self._camera_available = False
                logger.warning("Camera daemon not available - recording state only")
        except ImportError:
            self._camera_available = False
            logger.warning("camera_daemon not installed - recording state only")

    def _record_loop(self):
        """Background thread: sample state + camera at target FPS."""
        interval = 1.0 / self.fps

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            try:
                # Sample robot state
                state = self._controller.get_state()
                state_dict = {
                    "joint_positions": list(state.joint_positions),
                    "gripper_width": float(state.gripper_width),
                    "ee_position": list(state.ee_position),
                    "ee_orientation": list(state.ee_orientation),
                }

                # Sample camera
                jpeg_bytes = None
                if self._camera_available:
                    try:
                        jpeg_bytes = self._camera.get_frame_jpeg()
                    except Exception as e:
                        logger.debug(f"Camera frame failed: {e}")

                # Buffer the sample
                self._episode_frames.append(
                    (time.time(), state_dict, jpeg_bytes)
                )

            except Exception as e:
                logger.warning(f"Recording sample failed: {e}")

            # Sleep for remainder of interval
            elapsed = time.monotonic() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def start_episode(self, language_instruction: str, controller) -> dict:
        """
        Start background recording for a new episode.

        Args:
            language_instruction: Natural language task description
            controller: FrankaController instance (must have get_state())

        Returns:
            dict with episode_index and status
        """
        if self._recording:
            return {
                "success": False,
                "error": "Already recording. Stop current episode first.",
            }

        self._controller = controller
        self._init_camera()

        self._episode_index = self._get_next_episode_index()
        self._episode_frames = []
        self._episode_meta = {
            "language_instruction": language_instruction,
            "fps": self.fps,
            "start_time": time.time(),
            "robot_type": "franka",
            "camera_available": self._camera_available,
        }

        # Start recording thread
        self._stop_event.clear()
        self._recording = True
        self._thread = threading.Thread(
            target=self._record_loop, daemon=True, name="trajectory-recorder"
        )
        self._thread.start()

        logger.info(
            f"Recording episode {self._episode_index}: '{language_instruction}' "
            f"at {self.fps}fps (camera={'yes' if self._camera_available else 'no'})"
        )

        return {
            "success": True,
            "episode_index": self._episode_index,
            "fps": self.fps,
            "camera_available": self._camera_available,
        }

    def stop_episode(self, success: bool) -> dict:
        """
        Stop recording, flush buffer to disk, return stats.

        Args:
            success: Whether the demonstration was successful

        Returns:
            dict with episode stats and path
        """
        if not self._recording:
            return {"success": False, "error": "Not currently recording."}

        # Stop the recording thread
        self._stop_event.set()
        self._recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        end_time = time.time()
        num_frames = len(self._episode_frames)

        if num_frames == 0:
            return {
                "success": False,
                "error": "No frames were recorded.",
                "episode_index": self._episode_index,
            }

        # Create episode directory
        episode_dir = self.data_dir / f"episode_{self._episode_index:04d}"
        images_dir = episode_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build frames list and write images
        start_time = self._episode_meta["start_time"]
        frames = []
        images_saved = 0

        for i, (ts, state_dict, jpeg_bytes) in enumerate(self._episode_frames):
            image_path = None
            if jpeg_bytes is not None:
                image_filename = f"{i:06d}.jpg"
                (images_dir / image_filename).write_bytes(jpeg_bytes)
                image_path = f"images/{image_filename}"
                images_saved += 1

            frames.append({
                "timestamp": round(ts - start_time, 4),
                "frame_index": i,
                "image_path": image_path,
                **state_dict,
            })

        # Write episode.json
        episode_data = {
            "episode_index": self._episode_index,
            "language_instruction": self._episode_meta["language_instruction"],
            "success": success,
            "fps": self._episode_meta["fps"],
            "start_time": start_time,
            "end_time": end_time,
            "num_frames": num_frames,
            "num_images": images_saved,
            "robot_type": self._episode_meta["robot_type"],
            "frames": frames,
        }

        episode_json_path = episode_dir / "episode.json"
        with open(episode_json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        # Update dataset_info.json
        self._update_dataset_info()

        # Free the buffer
        self._episode_frames = []

        duration = end_time - start_time
        actual_fps = num_frames / duration if duration > 0 else 0

        logger.info(
            f"Episode {self._episode_index} saved: {num_frames} frames, "
            f"{duration:.1f}s, {actual_fps:.1f}fps actual, "
            f"{images_saved} images -> {episode_dir}"
        )

        return {
            "success": True,
            "episode_index": self._episode_index,
            "num_frames": num_frames,
            "num_images": images_saved,
            "duration": round(duration, 2),
            "actual_fps": round(actual_fps, 1),
            "path": str(episode_dir),
            "demonstration_success": success,
        }

    def _update_dataset_info(self):
        """Update the dataset_info.json summary file."""
        info_path = self.data_dir / "dataset_info.json"

        episodes = []
        for ep_dir in sorted(self.data_dir.glob("episode_*")):
            ep_json = ep_dir / "episode.json"
            if ep_json.exists():
                with open(ep_json) as f:
                    ep_data = json.load(f)
                episodes.append({
                    "episode_index": ep_data["episode_index"],
                    "language_instruction": ep_data["language_instruction"],
                    "success": ep_data["success"],
                    "num_frames": ep_data["num_frames"],
                    "duration": round(ep_data["end_time"] - ep_data["start_time"], 2),
                    "path": str(ep_dir),
                })

        info = {
            "num_episodes": len(episodes),
            "num_successful": sum(1 for e in episodes if e["success"]),
            "total_frames": sum(e["num_frames"] for e in episodes),
            "robot_type": "franka",
            "episodes": episodes,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def get_status(self) -> dict:
        """Return current recording state."""
        result = {
            "recording": self._recording,
            "data_dir": str(self.data_dir),
        }
        if self._recording:
            elapsed = time.time() - self._episode_meta.get("start_time", 0)
            result.update({
                "episode_index": self._episode_index,
                "frames_buffered": len(self._episode_frames),
                "elapsed_seconds": round(elapsed, 1),
                "language_instruction": self._episode_meta.get("language_instruction", ""),
                "camera_available": self._camera_available,
            })
        return result

    def list_episodes(self) -> dict:
        """List all recorded episodes with metadata."""
        info_path = self.data_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f)
        return {"num_episodes": 0, "episodes": []}


# Singleton
_recorder: Optional[TrajectoryRecorder] = None


def get_recorder(data_dir: str = "./data/recordings", fps: int = 30) -> TrajectoryRecorder:
    """Get or create the singleton TrajectoryRecorder."""
    global _recorder
    if _recorder is None:
        _recorder = TrajectoryRecorder(data_dir=data_dir, fps=fps)
    return _recorder
