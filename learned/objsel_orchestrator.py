"""
Object-Selection VLM Orchestrator.

Chains the fine-tuned Moondream VLM with Moondream's detect API to execute
pick-and-place tasks. The VLM decides WHICH object to act on; detect() finds WHERE.

Pipeline:
    1. Capture camera frame
    2. Detect all visible objects (Moondream detect)
    3. Build scene description ("Visible objects: red block, blue block")
    4. Query fine-tuned VLM with scene + instruction
       → {"skill": "pick", "object": "red block"}
    5. Use detect() to localize "red block" → pixel coords
    6. Homography → robot coords
    7. Execute pick_at(x, y)
    8. Re-capture, query VLM again → {"skill": "place"}
    9. Execute place_at(x, y)
    10. Re-capture, query VLM → {"skill": "done"}

This runs on the Pi, calling out to the Moondream server on Spark.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Block queries we scan for to build the scene description
SCAN_QUERIES = [
    "red block",
    "blue block",
    "green block",
    "wooden block",
    "orange block",
]

# Safe workspace for picking (slightly wider than data collection to account for
# detection noise at boundaries)
PICK_WORKSPACE = {
    "x_min": 0.33,
    "x_max": 0.57,
    "y_min": -0.28,
    "y_max": 0.28,
}

# Placement bounds
PLACE_BOUNDS = {
    "x_min": 0.35,
    "x_max": 0.52,
    "y_min": -0.25,
    "y_max": 0.25,
}

# Scene prompt templates (match training data format)
SCENE_PROMPT_TEMPLATES = [
    "You see: {objects}. {instruction}",
    "Objects on the table: {objects}. {instruction}",
    "Visible objects: {objects}. {instruction}",
]

PICK_INSTRUCTIONS = [
    "Pick up the {obj}.",
    "Pick the {obj} up.",
    "Grab the {obj}.",
    "Get the {obj}.",
    "Pick up the {obj} from the table.",
]

PLACE_INSTRUCTIONS = [
    "Put the block down.",
    "Place the block down.",
    "Set the block down.",
    "Release the block.",
    "Put it down.",
]


@dataclass
class SceneObject:
    """An object detected in the scene."""
    query: str  # e.g. "red block"
    pixel_x: int
    pixel_y: int
    robot_x: float
    robot_y: float
    bbox_w: int
    bbox_h: int
    in_workspace: bool


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    moondream_url: str = "http://spark:8091"
    grasp_width: float = 0.03
    grasp_z: float = 0.018
    place_z: float = 0.015
    scan_queries: list = None
    min_neighbor_dist: float = 0.05  # 5cm

    def __post_init__(self):
        if self.scan_queries is None:
            self.scan_queries = list(SCAN_QUERIES)


def _detect_scene(frame, H, config):
    """Detect all objects in the scene, return list of SceneObject."""
    from learned.moondream_client import detect as md_detect
    from learned.block_detector import pixel_to_robot

    objects = []
    seen_positions = set()

    for query in config.scan_queries:
        try:
            detections = md_detect(frame, query, server_url=config.moondream_url)
        except Exception as e:
            logger.warning(f"detect({query!r}) failed: {e}")
            continue

        for det in detections:
            # Deduplicate within 30px
            key = (det.center_x // 30, det.center_y // 30)
            if key in seen_positions:
                continue
            seen_positions.add(key)

            rx, ry = pixel_to_robot(det.center_x, det.center_y, H)
            rx, ry = float(rx), float(ry)
            in_ws = bool(
                PICK_WORKSPACE["x_min"] <= rx <= PICK_WORKSPACE["x_max"]
                and PICK_WORKSPACE["y_min"] <= ry <= PICK_WORKSPACE["y_max"]
            )

            objects.append(SceneObject(
                query=query,
                pixel_x=int(det.center_x),
                pixel_y=int(det.center_y),
                robot_x=rx,
                robot_y=ry,
                bbox_w=int(det.bbox_w),
                bbox_h=int(det.bbox_h),
                in_workspace=in_ws,
            ))

    return objects


def _build_scene_query(objects, instruction):
    """Build a scene-context prompt like training data.

    Lists ALL detected objects (not just workspace ones) so the VLM
    has full scene context for object selection.
    """
    obj_names = sorted(set(o.query for o in objects))
    if obj_names:
        objects_str = ", ".join(obj_names)
        template = random.choice(SCENE_PROMPT_TEMPLATES)
        return template.format(objects=objects_str, instruction=instruction)
    return instruction


def _parse_vlm_response(answer: str) -> Optional[dict]:
    """Parse VLM JSON response, handling common formatting issues.

    The fine-tuned model sometimes generates junk tokens before the real
    answer, e.g.: {" "": "pick", " ": "pick", "skill": "pick", "object": "red block"}
    We extract the "skill" and "object" fields via regex as a fallback.
    """
    import re
    answer = answer.strip()

    # Try direct JSON parse
    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from surrounding text
    idx = answer.find("{")
    if idx >= 0:
        end = answer.rfind("}")
        if end > idx:
            try:
                return json.loads(answer[idx:end + 1])
            except json.JSONDecodeError:
                pass

    # Regex fallback: extract "skill" and "object" values directly
    skill_match = re.search(r'"skill"\s*:\s*"([^"]+)"', answer)
    object_match = re.search(r'"object"\s*:\s*"([^"]+)"', answer)

    if skill_match:
        result = {"skill": skill_match.group(1)}
        if object_match:
            result["object"] = object_match.group(1)
        logger.info(f"Parsed via regex fallback: {result}")
        return result

    logger.warning(f"Could not parse VLM response: {answer!r}")
    return None


def _find_object_coords(target_query, scene_objects, frame, H, config):
    """Find robot coordinates for a specific object query.

    First checks if we already detected it in the scene scan.
    Falls back to a targeted detect() call.
    """
    from learned.moondream_client import detect as md_detect
    from learned.block_detector import pixel_to_robot

    # Check scene objects first — prefer closest to workspace center for reliability
    import math
    WS_CENTER_X = (PICK_WORKSPACE["x_min"] + PICK_WORKSPACE["x_max"]) / 2
    WS_CENTER_Y = (PICK_WORKSPACE["y_min"] + PICK_WORKSPACE["y_max"]) / 2
    candidates = []
    for obj in scene_objects:
        if obj.query == target_query and obj.in_workspace:
            dist = math.sqrt((obj.robot_x - WS_CENTER_X)**2 + (obj.robot_y - WS_CENTER_Y)**2)
            candidates.append((dist, obj))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, best = candidates[0]
        return best.robot_x, best.robot_y, best.pixel_x, best.pixel_y

    # Targeted detection as fallback — check ALL detections for one in workspace
    logger.info(f"Object {target_query!r} not in scene scan, running targeted detect")
    try:
        detections = md_detect(frame, target_query, server_url=config.moondream_url)
        for det in detections:
            rx, ry = pixel_to_robot(det.center_x, det.center_y, H)
            rx, ry = float(rx), float(ry)
            if (PICK_WORKSPACE["x_min"] <= rx <= PICK_WORKSPACE["x_max"]
                    and PICK_WORKSPACE["y_min"] <= ry <= PICK_WORKSPACE["y_max"]):
                return rx, ry, det.center_x, det.center_y
            else:
                logger.warning(f"Object {target_query!r} at ({rx:.3f}, {ry:.3f}) "
                               f"is outside safe workspace")
        if detections:
            logger.warning(f"No {target_query!r} detections in safe workspace "
                           f"({len(detections)} found but all out of bounds)")
    except Exception as e:
        logger.warning(f"Targeted detect({target_query!r}) failed: {e}")

    return None


def _random_place_position(existing_objects):
    """Generate a random placement position avoiding existing blocks."""
    import math
    bounds = PLACE_BOUNDS
    for _ in range(50):
        x = random.uniform(bounds["x_min"], bounds["x_max"])
        y = random.uniform(bounds["y_min"], bounds["y_max"])
        too_close = False
        for obj in existing_objects:
            if not obj.in_workspace:
                continue
            dist = math.sqrt((x - obj.robot_x)**2 + (y - obj.robot_y)**2)
            if dist < 0.06:
                too_close = True
                break
        if not too_close:
            return x, y
    return random.uniform(bounds["x_min"], bounds["x_max"]), \
           random.uniform(bounds["y_min"], bounds["y_max"])


def run_task(controller, instruction: str,
             config: Optional[OrchestratorConfig] = None,
             max_steps: int = 5) -> dict:
    """Execute a single task using the VLM→detect→execute pipeline.

    Args:
        controller: FrankaController instance
        instruction: Natural language task, e.g. "Pick up the red block."
        config: Orchestrator configuration
        max_steps: Maximum VLM query steps before giving up

    Returns:
        dict with execution results
    """
    if config is None:
        config = OrchestratorConfig()

    from learned.moondream_client import query as md_query
    from learned.block_detector import load_homography
    from camera_daemon.client import CameraClient

    H, _ = load_homography()
    cam = CameraClient()
    if not cam.connect():
        return {"success": False, "error": "Camera connection failed"}

    # Move arm out of the way before scene detection
    try:
        controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
        logger.info("Moved arm to home position")
    except Exception as e:
        logger.warning(f"Could not move to home: {e}")
        try:
            controller.recover()
            controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
        except Exception:
            pass

    results = {
        "instruction": instruction,
        "steps": [],
        "success": False,
        "start_time": time.time(),
    }
    holding_object = False  # Track if we're holding something

    try:
        for step_num in range(max_steps):
            logger.info(f"=== Step {step_num + 1}/{max_steps} ===")

            # 1. Capture frame
            frame = cam.get_frame()
            if frame is None:
                results["steps"].append({"error": "frame_capture_failed"})
                break

            # 2. Detect scene (skip full scan when placing — not needed)
            if not holding_object:
                scene_objects = _detect_scene(frame, H, config)
                ws_objects = [o for o in scene_objects if o.in_workspace]
                logger.info(f"Scene: {len(ws_objects)} objects in workspace: "
                            f"{[o.query for o in ws_objects]}")
            else:
                scene_objects = []
                ws_objects = []

            # 3. Build query
            if holding_object:
                # Place instruction — NO scene context (matches training data)
                question = random.choice(PLACE_INSTRUCTIONS)
            else:
                # Pick instruction — WITH scene context
                question = _build_scene_query(scene_objects, instruction)
            logger.info(f"VLM query: {question!r}")

            # 4. Query VLM
            vlm_result = md_query(frame, question, server_url=config.moondream_url)
            if vlm_result is None:
                results["steps"].append({"error": "vlm_query_failed"})
                break

            logger.info(f"VLM answer: {vlm_result.answer!r} ({vlm_result.latency_ms:.0f}ms)")

            # 5. Parse response
            parsed = _parse_vlm_response(vlm_result.answer)
            if parsed is None:
                results["steps"].append({
                    "error": "parse_failed",
                    "raw_answer": vlm_result.answer,
                })
                break

            skill = parsed.get("skill", "")
            step_result = {
                "step": step_num,
                "skill": skill,
                "vlm_answer": parsed,
                "vlm_latency_ms": vlm_result.latency_ms,
                "scene_objects": [o.query for o in ws_objects],
            }

            # 6. Execute skill
            if skill == "pick":
                target_query = parsed.get("object", "")
                if not target_query:
                    step_result["error"] = "no_object_specified"
                    results["steps"].append(step_result)
                    break

                coords = _find_object_coords(
                    target_query, scene_objects, frame, H, config
                )
                if coords is None:
                    step_result["error"] = f"object_not_found: {target_query}"
                    results["steps"].append(step_result)
                    break

                rx, ry, px, py = coords
                step_result["target"] = target_query
                step_result["robot_coords"] = [round(rx, 4), round(ry, 4)]
                step_result["pixel_coords"] = [px, py]

                logger.info(f"Picking {target_query!r} at robot ({rx:.3f}, {ry:.3f})")

                try:
                    pick_result = controller.pick_at(
                        x=rx, y=ry,
                        grasp_width=config.grasp_width,
                        z=config.grasp_z,
                    )
                    gripper_w = pick_result.get("gripper_width", 0.08)
                    # Check grasp: must be closed enough but not fully empty
                    GRIP_CLOSED_EMPTY = 0.002  # meters — fully closed, nothing held
                    pick_success = GRIP_CLOSED_EMPTY < gripper_w < 0.05
                    step_result["pick_success"] = pick_success
                    step_result["gripper_width"] = gripper_w

                    if not pick_success:
                        logger.warning(f"Pick failed (gripper_width={gripper_w:.4f})")
                        step_result["error"] = "grasp_failed"
                        results["steps"].append(step_result)
                        # Try to recover
                        try:
                            controller.open_gripper()
                            controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
                        except Exception:
                            pass
                        break

                    logger.info(f"Pick SUCCESS (gripper_width={gripper_w:.4f})")
                    holding_object = True

                except Exception as e:
                    step_result["error"] = f"pick_exception: {e}"
                    results["steps"].append(step_result)
                    try:
                        controller.recover()
                    except Exception:
                        pass
                    break

            elif skill == "place":
                # Place at a random safe position
                place_x, place_y = _random_place_position(ws_objects)
                step_result["robot_coords"] = [round(place_x, 4), round(place_y, 4)]
                logger.info(f"Placing at ({place_x:.3f}, {place_y:.3f})")

                try:
                    controller.place_at(
                        x=place_x, y=place_y,
                        z=config.place_z,
                    )
                    step_result["place_success"] = True
                    holding_object = False
                    results["success"] = True
                    results["steps"].append(step_result)
                    logger.info("Pick-and-place complete!")
                    break
                except Exception as e:
                    step_result["error"] = f"place_exception: {e}"
                    results["steps"].append(step_result)
                    try:
                        controller.recover()
                    except Exception:
                        pass
                    break

            elif skill == "done":
                step_result["task_complete"] = True
                results["steps"].append(step_result)
                results["success"] = True
                logger.info("Task complete!")
                break

            else:
                step_result["error"] = f"unknown_skill: {skill}"
                results["steps"].append(step_result)
                break

            results["steps"].append(step_result)

        else:
            logger.warning(f"Max steps ({max_steps}) reached without completion")

    finally:
        cam.disconnect()
        results["elapsed_s"] = round(time.time() - results["start_time"], 1)
        del results["start_time"]

    return results


def run_continuous(controller, instructions: list[str],
                   config: Optional[OrchestratorConfig] = None,
                   pause_between: float = 2.0) -> dict:
    """Run multiple tasks in sequence.

    Args:
        controller: FrankaController instance
        instructions: List of task instructions to execute
        config: Orchestrator configuration
        pause_between: Seconds to pause between tasks

    Returns:
        dict with overall results
    """
    if config is None:
        config = OrchestratorConfig()

    overall = {
        "tasks_attempted": 0,
        "tasks_succeeded": 0,
        "task_results": [],
    }

    for i, instruction in enumerate(instructions):
        logger.info(f"\n{'='*60}")
        logger.info(f"Task {i+1}/{len(instructions)}: {instruction}")
        logger.info(f"{'='*60}")

        overall["tasks_attempted"] += 1
        result = run_task(controller, instruction, config)
        overall["task_results"].append(result)

        if result.get("success"):
            overall["tasks_succeeded"] += 1

        if i < len(instructions) - 1:
            # Go home between tasks
            try:
                controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
            except Exception:
                pass
            time.sleep(pause_between)

    logger.info(f"\nDone: {overall['tasks_succeeded']}/{overall['tasks_attempted']} tasks succeeded")
    return overall
