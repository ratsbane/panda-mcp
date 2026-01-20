"""
High-level manipulation utilities.

Combines vision and arm control for pick-and-place tasks.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .vision import DetectedObject, find_largest_object, find_colored_blocks
from .calibration import get_transformer, CoordinateTransformer


@dataclass
class GraspPlan:
    """A planned grasp for an object."""
    target_position: tuple[float, float, float]  # x, y, z in robot frame
    approach_height: float  # Height to approach from
    grasp_width: float  # Gripper width for grasp
    grasp_force: float  # Force in Newtons
    confidence: float  # How confident we are in this plan

    def to_dict(self) -> dict:
        return {
            "target": {
                "x": self.target_position[0],
                "y": self.target_position[1],
                "z": self.target_position[2],
            },
            "approach_height": self.approach_height,
            "grasp_width": self.grasp_width,
            "grasp_force": self.grasp_force,
            "confidence": self.confidence,
        }


def plan_grasp_for_object(
    obj: DetectedObject,
    transformer: Optional[CoordinateTransformer] = None,
    object_height: float = 0.03,  # Estimated object height
    grasp_offset: float = 0.01,  # How far into object to grasp
    approach_clearance: float = 0.08,  # Height above object to approach from
) -> GraspPlan:
    """
    Plan a top-down grasp for a detected object.

    Args:
        obj: Detected object with bounding box
        transformer: Coordinate transformer (uses global if None)
        object_height: Estimated height of object in meters
        grasp_offset: How far into the object to close gripper
        approach_clearance: Height above object for approach

    Returns:
        GraspPlan with target coordinates and parameters
    """
    if transformer is None:
        transformer = get_transformer()

    # Get center of object in pixels
    cx, cy = obj.bbox.center

    # Transform to robot coordinates
    robot_x, robot_y, table_z = transformer.pixel_to_robot(cx, cy)

    # Grasp height is table + half object height
    grasp_z = table_z + object_height / 2

    # Estimate grasp width from bounding box
    # Use the smaller dimension of the bbox
    bbox = obj.bbox
    # Convert pixel width to approximate meters (rough estimate)
    # This is very approximate without proper calibration
    cal = transformer.calibration
    meters_per_pixel_x = (cal.workspace_y_max - cal.workspace_y_min) / cal.image_width
    meters_per_pixel_y = (cal.workspace_x_max - cal.workspace_x_min) / cal.image_height

    obj_width_m = min(bbox.width * meters_per_pixel_x, bbox.height * meters_per_pixel_y)

    # Grasp width should be slightly less than object width
    grasp_width = max(0.01, obj_width_m - grasp_offset)
    grasp_width = min(grasp_width, 0.08)  # Clamp to gripper max

    return GraspPlan(
        target_position=(robot_x, robot_y, grasp_z),
        approach_height=grasp_z + approach_clearance,
        grasp_width=grasp_width,
        grasp_force=20.0,  # Default gentle grasp
        confidence=obj.confidence,
    )


def execute_pick(
    controller,  # FrankaController
    plan: GraspPlan,
    speed_factor: float = 0.1,
) -> dict:
    """
    Execute a pick operation based on a grasp plan.

    Args:
        controller: FrankaController instance
        plan: GraspPlan to execute
        speed_factor: Movement speed (0.0 to 1.0)

    Returns:
        dict with success status and details
    """
    results = {"steps": []}

    try:
        x, y, z = plan.target_position

        # Step 1: Open gripper
        result = controller.gripper_move(0.08)
        results["steps"].append({"action": "open_gripper", "result": result})
        if not result.get("success"):
            return {"success": False, "error": "Failed to open gripper", "details": results}

        # Step 2: Move to approach position (above object)
        result = controller.move_cartesian(x, y, plan.approach_height, confirmed=True)
        results["steps"].append({"action": "approach", "result": result})
        if not result.get("success") and not result.get("requires_confirmation"):
            return {"success": False, "error": "Failed to approach", "details": results}

        # Step 3: Move down to grasp position
        result = controller.move_cartesian(x, y, z, confirmed=True)
        results["steps"].append({"action": "descend", "result": result})

        # Step 4: Close gripper to grasp
        result = controller.gripper_grasp(
            width=plan.grasp_width,
            force=plan.grasp_force,
        )
        results["steps"].append({"action": "grasp", "result": result})

        # Step 5: Lift object
        result = controller.move_cartesian(x, y, plan.approach_height, confirmed=True)
        results["steps"].append({"action": "lift", "result": result})

        return {"success": True, "details": results}

    except Exception as e:
        return {"success": False, "error": str(e), "details": results}


def execute_place(
    controller,  # FrankaController
    target_x: float,
    target_y: float,
    target_z: float,
    approach_height: float = 0.15,
    speed_factor: float = 0.1,
) -> dict:
    """
    Execute a place operation.

    Args:
        controller: FrankaController instance
        target_x, target_y, target_z: Target position
        approach_height: Height to approach from
        speed_factor: Movement speed

    Returns:
        dict with success status
    """
    results = {"steps": []}

    try:
        # Step 1: Move to above target
        result = controller.move_cartesian(target_x, target_y, approach_height, confirmed=True)
        results["steps"].append({"action": "approach", "result": result})

        # Step 2: Lower to place position
        result = controller.move_cartesian(target_x, target_y, target_z, confirmed=True)
        results["steps"].append({"action": "descend", "result": result})

        # Step 3: Open gripper to release
        result = controller.gripper_move(0.08)
        results["steps"].append({"action": "release", "result": result})

        # Step 4: Retreat upward
        result = controller.move_cartesian(target_x, target_y, approach_height, confirmed=True)
        results["steps"].append({"action": "retreat", "result": result})

        return {"success": True, "details": results}

    except Exception as e:
        return {"success": False, "error": str(e), "details": results}


def find_and_plan_grasp(
    image: np.ndarray,
    color: Optional[str] = None,
) -> Optional[tuple[DetectedObject, GraspPlan]]:
    """
    Find an object in the image and plan a grasp for it.

    Args:
        image: BGR image from camera
        color: Specific color to look for (or None for any object)

    Returns:
        Tuple of (detected_object, grasp_plan) or None if nothing found
    """
    if color:
        objects = find_colored_blocks(image, colors=[color])
        obj = objects[0] if objects else None
    else:
        obj = find_largest_object(image)

    if obj is None:
        return None

    plan = plan_grasp_for_object(obj)
    return (obj, plan)
