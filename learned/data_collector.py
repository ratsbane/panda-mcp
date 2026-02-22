"""
Autonomous training data collection for Moondream fine-tuning.

Runs a pick-place-shuffle loop:
1. Detect all blocks via Moondream
2. Pick a random one
3. Place it at a random open position
4. Record (image, instruction, robot_coords) at each step
5. Repeat

Data is saved as individual episodes, one directory per pick-place cycle.
A conversion script can later transform this into Moondream fine-tuning format.
"""

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Block queries for Moondream detection
BLOCK_QUERIES = [
    "green block",
    "red block",
    "blue block",
    "wooden block",
    "orange block",
]

# Descriptions for natural language instructions
PICK_TEMPLATES = [
    "Pick up the {query}.",
    "Grab the {query}.",
    "Pick the {query} up.",
]

PLACE_TEMPLATES = [
    "Place the block down.",
    "Put the block down.",
    "Set the block down.",
]

# Workspace bounds for random placement
PLACE_BOUNDS = {
    "x_min": 0.35,
    "x_max": 0.52,
    "y_min": -0.18,
    "y_max": 0.18,
}

# Safe pick workspace (tighter than calibration.json — x<0.35 causes IK issues at table height)
PICK_WORKSPACE = {
    "x_min": 0.35,
    "x_max": 0.55,
    "y_min": -0.20,
    "y_max": 0.20,
}

# Minimum distance between blocks for placement (meters)
MIN_BLOCK_SPACING = 0.06


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    output_dir: str = "data/moondream_training"
    num_episodes: int = 10
    grasp_width: float = 0.03
    grasp_z: float = 0.018
    place_z: float = 0.015
    moondream_url: str = "http://spark:8091"
    block_queries: list = None  # defaults to BLOCK_QUERIES
    pause_between_episodes: float = 1.0  # seconds

    def __post_init__(self):
        if self.block_queries is None:
            self.block_queries = list(BLOCK_QUERIES)


@dataclass
class DetectedObject:
    """An object detected for collection."""
    query: str
    pixel_x: int
    pixel_y: int
    robot_x: float
    robot_y: float
    bbox_w: int
    bbox_h: int
    in_workspace: bool


def _detect_all_blocks(frame, H, workspace, config):
    """Detect all blocks using Moondream, return list of DetectedObject."""
    from learned.moondream_client import detect as md_detect
    from learned.block_detector import pixel_to_robot

    all_objects = []
    seen_positions = set()

    for query in config.block_queries:
        try:
            detections = md_detect(frame, query, server_url=config.moondream_url)
        except Exception as e:
            logger.warning(f"Moondream detect({query!r}) failed: {e}")
            continue

        for det in detections:
            # Deduplicate: skip if we already have something within 30px
            key = (det.center_x // 30, det.center_y // 30)
            if key in seen_positions:
                continue
            seen_positions.add(key)

            rx, ry = pixel_to_robot(det.center_x, det.center_y, H)
            rx, ry = float(rx), float(ry)
            in_ws = bool(
                workspace["x_min"] <= rx <= workspace["x_max"]
                and workspace["y_min"] <= ry <= workspace["y_max"]
            )

            all_objects.append(DetectedObject(
                query=query,
                pixel_x=int(det.center_x),
                pixel_y=int(det.center_y),
                robot_x=rx,
                robot_y=ry,
                bbox_w=int(det.bbox_w),
                bbox_h=int(det.bbox_h),
                in_workspace=in_ws,
            ))

    return all_objects


def _random_place_position(existing_objects, bounds=None):
    """Generate a random placement position that doesn't overlap existing blocks."""
    if bounds is None:
        bounds = PLACE_BOUNDS

    for _ in range(50):  # max attempts
        x = random.uniform(bounds["x_min"], bounds["x_max"])
        y = random.uniform(bounds["y_min"], bounds["y_max"])

        # Check distance from all existing objects
        too_close = False
        for obj in existing_objects:
            if not obj.in_workspace:
                continue
            dist = math.sqrt((x - obj.robot_x)**2 + (y - obj.robot_y)**2)
            if dist < MIN_BLOCK_SPACING:
                too_close = True
                break

        if not too_close:
            return x, y

    # Fallback: just pick a random position
    logger.warning("Could not find non-overlapping position, using random")
    return (
        random.uniform(bounds["x_min"], bounds["x_max"]),
        random.uniform(bounds["y_min"], bounds["y_max"]),
    )


def _save_step(episode_dir, step_name, frame, metadata):
    """Save a single step's image and metadata."""
    cv2.imwrite(str(episode_dir / f"{step_name}.jpg"), frame)
    with open(episode_dir / f"{step_name}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _check_and_recover(controller, stats):
    """Check robot error state and attempt recovery. Returns True if ok/recovered."""
    try:
        state = controller.get_state()
        if not state.has_error:
            # No error, just move home
            try:
                controller.open_gripper()
                controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
            except Exception:
                pass
            return True

        logger.warning(f"Robot in error state: {state.error_message}")
        stats.setdefault("robot_errors", 0)
        stats["robot_errors"] += 1

        # Attempt automatic recovery
        result = controller.recover()
        if result.get("success"):
            logger.info("Robot recovered from error")
            try:
                controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
            except Exception:
                pass
            return True
        else:
            logger.error(f"Recovery failed: {result.get('error')}")
            return False
    except Exception as e:
        logger.error(f"Error checking robot state: {e}")
        return False


def run_collection(controller, config: Optional[CollectionConfig] = None):
    """Run the autonomous data collection loop.

    Args:
        controller: FrankaController instance (must be connected)
        config: Collection configuration

    Returns:
        dict with summary statistics
    """
    if config is None:
        config = CollectionConfig()

    from learned.block_detector import load_homography
    from camera_daemon.client import CameraClient

    # Setup — use safe pick workspace, not calibration.json bounds
    H, _cal_workspace = load_homography()
    workspace = PICK_WORKSPACE
    cam = CameraClient()
    if not cam.connect():
        return {"error": "Failed to connect to camera"}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find next episode number
    existing = sorted(output_dir.glob("episode_*"))
    start_num = int(existing[-1].name.split("_")[1]) + 1 if existing else 0

    stats = {
        "episodes_attempted": 0,
        "episodes_completed": 0,
        "pick_successes": 0,
        "pick_failures": 0,
        "detection_failures": 0,
        "start_episode": start_num,
    }

    logger.info(f"Starting collection: {config.num_episodes} episodes, "
                f"output={config.output_dir}, starting at episode {start_num}")

    try:
        for i in range(config.num_episodes):
            episode_num = start_num + i
            episode_dir = output_dir / f"episode_{episode_num:04d}"
            episode_dir.mkdir(exist_ok=True)
            stats["episodes_attempted"] += 1

            logger.info(f"=== Episode {episode_num} ({i+1}/{config.num_episodes}) ===")

            # Step 1: Capture scene and detect blocks
            frame = cam.get_frame()
            if frame is None:
                logger.error("Failed to capture frame, skipping episode")
                stats["detection_failures"] += 1
                continue

            objects = _detect_all_blocks(frame, H, workspace, config)
            ws_objects = [o for o in objects if o.in_workspace]

            if not ws_objects:
                logger.warning("No objects in workspace, skipping episode")
                stats["detection_failures"] += 1
                # Save the empty scene anyway for debugging
                _save_step(episode_dir, "scene", frame, {
                    "status": "no_objects",
                    "all_detections": [asdict(o) for o in objects],
                })
                continue

            # Step 2: Choose a random block to pick (skip blocks too close to neighbors)
            MIN_NEIGHBOR_DIST = 0.05  # 5cm — gripper is ~8cm wide
            safe_targets = []
            for candidate in ws_objects:
                too_close = False
                for other in ws_objects:
                    if other is candidate:
                        continue
                    dist = math.sqrt((candidate.robot_x - other.robot_x)**2 +
                                     (candidate.robot_y - other.robot_y)**2)
                    if dist < MIN_NEIGHBOR_DIST:
                        too_close = True
                        break
                if not too_close:
                    safe_targets.append(candidate)

            if not safe_targets:
                logger.warning("All blocks too close to neighbors, picking random anyway")
                safe_targets = ws_objects

            target = random.choice(safe_targets)
            instruction = random.choice(PICK_TEMPLATES).format(query=target.query)

            logger.info(f"  Target: {target.query} at robot ({target.robot_x:.3f}, {target.robot_y:.3f})")

            # Step 3: Save pre-pick data
            _save_step(episode_dir, "pick", frame, {
                "type": "pick",
                "instruction": instruction,
                "query": target.query,
                "target_pixel": [target.pixel_x, target.pixel_y],
                "target_robot": [round(target.robot_x, 4), round(target.robot_y, 4)],
                "target_bbox": [target.bbox_w, target.bbox_h],
                "all_objects": [asdict(o) for o in objects],
                "ws_objects": [asdict(o) for o in ws_objects],
                "timestamp": time.time(),
            })

            # Step 4: Execute pick
            try:
                pick_result = controller.pick_at(
                    x=target.robot_x,
                    y=target.robot_y,
                    grasp_width=config.grasp_width,
                    z=config.grasp_z,
                )
            except Exception as e:
                logger.error(f"  Pick failed with exception: {e}")
                stats["pick_failures"] += 1
                _save_step(episode_dir, "pick_result", frame, {
                    "success": False, "error": str(e),
                })
                # Check if robot is in error state
                if _check_and_recover(controller, stats):
                    continue  # recovered, try next episode
                else:
                    logger.error("Robot in unrecoverable error state, aborting collection")
                    stats["abort_reason"] = "unrecoverable_robot_error"
                    break

            pick_success = bool(pick_result.get("gripper_width", 0.08) < 0.05)
            logger.info(f"  Pick: {'SUCCESS' if pick_success else 'FAIL'} "
                       f"(width={pick_result.get('gripper_width', '?')})")

            # Save pick result
            with open(episode_dir / "pick_result.json", "w") as f:
                json.dump({"success": pick_success, "gripper_width": pick_result.get("gripper_width")}, f)

            if not pick_success:
                stats["pick_failures"] += 1
                if not _check_and_recover(controller, stats):
                    logger.error("Robot in unrecoverable error state, aborting collection")
                    stats["abort_reason"] = "unrecoverable_robot_error"
                    break
                continue

            stats["pick_successes"] += 1

            # Step 5: Capture frame while holding, choose place position
            frame2 = cam.get_frame()
            other_objects = [o for o in ws_objects if o is not target]
            place_x, place_y = _random_place_position(other_objects)
            place_instruction = random.choice(PLACE_TEMPLATES)

            logger.info(f"  Placing at ({place_x:.3f}, {place_y:.3f})")

            # Step 6: Save pre-place data
            if frame2 is not None:
                _save_step(episode_dir, "place", frame2, {
                    "type": "place",
                    "instruction": place_instruction,
                    "target_robot": [round(place_x, 4), round(place_y, 4)],
                    "holding_query": target.query,
                    "timestamp": time.time(),
                })

            # Step 7: Execute place
            try:
                place_result = controller.place_at(
                    x=place_x,
                    y=place_y,
                    z=config.place_z,
                )
            except Exception as e:
                logger.error(f"  Place failed: {e}")
                if not _check_and_recover(controller, stats):
                    logger.error("Robot in unrecoverable error state, aborting collection")
                    stats["abort_reason"] = "unrecoverable_robot_error"
                    break
                continue

            # Step 8: Record completion
            with open(episode_dir / "episode_meta.json", "w") as f:
                json.dump({
                    "episode": episode_num,
                    "pick_query": target.query,
                    "pick_instruction": instruction,
                    "pick_robot": [round(target.robot_x, 4), round(target.robot_y, 4)],
                    "pick_pixel": [target.pixel_x, target.pixel_y],
                    "place_robot": [round(place_x, 4), round(place_y, 4)],
                    "place_instruction": place_instruction,
                    "success": True,
                    "timestamp": time.time(),
                }, f, indent=2)

            stats["episodes_completed"] += 1
            logger.info(f"  Episode {episode_num} complete")

            # Brief pause between episodes
            if config.pause_between_episodes > 0 and i < config.num_episodes - 1:
                time.sleep(config.pause_between_episodes)

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    finally:
        cam.disconnect()
        # Try to go home
        try:
            controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
        except Exception:
            pass

    stats["end_episode"] = start_num + stats["episodes_attempted"] - 1
    logger.info(f"Collection complete: {stats}")
    return stats
