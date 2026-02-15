"""
Visual Servo Loop for SAWM.

Two-phase approach:
  Phase 1 (Coarse Servo): Gripper tilted forward so it's visible to USB camera.
    Iteratively capture frames, predict gripper-to-target offsets, move toward target.
    Progressive crop tightens as XY distance decreases.
  Phase 2 (Fine Grasp): Untilt to vertical, lower incrementally, grasp, lift.

In fallback mode (no model), the gripper moves toward the hint position in
fractional steps — still generates valid training data for self-supervised learning.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .cropper import get_cropper

logger = logging.getLogger(__name__)


@dataclass
class ServoConfig:
    """Configuration for visual servo loop."""
    servo_z: float = 0.10           # Height during servo (10cm — clears blocks even with tilt)
    servo_pitch: float = -0.40      # Forward tilt toward camera (~-23 deg) — gripper visible
    gain: float = 0.5               # Move half the predicted/estimated offset each step
    max_step: float = 0.05          # 50mm max per iteration (faster convergence)
    convergence_threshold: float = 0.015  # Stop when <15mm offset
    max_iterations: int = 20
    min_iterations: int = 3         # Always do at least this many steps (data diversity)
    step_delay: float = 0.3         # Seconds between servo steps (settle + camera)
    fallback_to_hint: bool = True   # Use hint-based motion if no model loaded


class VisualServoLoop:
    """
    Executes a two-phase visual servo pick.

    Phase 1: Coarse approach with tilted gripper (visible to camera)
    Phase 2: Vertical descent + grasp (reuses pick_at lowering logic)
    """

    def __init__(
        self,
        controller,
        config: Optional[ServoConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            controller: FrankaController instance
            config: ServoConfig (uses defaults if None)
            model_path: Path to ONNX model (None = fallback/data-collection mode)
        """
        self._controller = controller
        self._config = config or ServoConfig()
        self._cropper = get_cropper()
        self._model_path = model_path

        # ONNX model (lazy loaded)
        self._session = None
        if model_path:
            self._load_model(model_path)

        # ImageNet normalization
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_model(self, model_path: str):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"Servo ONNX model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load servo model: {e}")
            self._session = None

    @property
    def has_model(self) -> bool:
        return self._session is not None

    def execute(
        self,
        target_x_hint: float,
        target_y_hint: float,
        grasp_width: float = 0.03,
        grasp_force: float = 70.0,
        grasp_z: float = 0.013,
        approach_height: float = 0.15,
        collect_data: bool = False,
    ) -> dict:
        """
        Execute a two-phase visual servo pick.

        Args:
            target_x_hint: Rough X position hint (meters)
            target_y_hint: Rough Y position hint (meters)
            grasp_width: Expected object width for grasp (meters)
            grasp_force: Grasp force in Newtons
            grasp_z: Height to grasp at (meters, default table height)
            approach_height: Height to lift to after grasp (meters)
            collect_data: Whether to record frames for training

        Returns:
            dict with success, convergence info, phase details
        """
        cfg = self._config

        # Data collection
        collector = None
        if collect_data:
            from .servo_collector import get_servo_collector
            collector = get_servo_collector()
            collector.start_approach(target_hint_xy=(target_x_hint, target_y_hint))

        # Phase 1: Coarse visual servo
        phase1 = self._phase1_coarse_servo(
            target_x_hint, target_y_hint, collector
        )

        if not phase1["success"]:
            if collector and collector.active:
                collector.end_approach(success=False, final_gripper_xy=(0, 0))
            return {
                "success": False,
                "phase1": phase1,
                "phase2": None,
            }

        # Phase 2: Fine vertical grasp
        converged_x = phase1["final_x"]
        converged_y = phase1["final_y"]

        phase2 = self._phase2_fine_grasp(
            converged_x, converged_y,
            grasp_width=grasp_width,
            grasp_force=grasp_force,
            grasp_z=grasp_z,
            approach_height=approach_height,
        )

        # End data collection
        if collector and collector.active:
            if phase2["grasped"]:
                ee = self._controller.get_state().ee_position
                collector.end_approach(success=True, final_gripper_xy=(ee[0], ee[1]))
            else:
                collector.end_approach(success=False, final_gripper_xy=(0, 0))

        return {
            "success": phase2["grasped"],
            "converged": phase1["converged"],
            "iterations": phase1["iterations"],
            "phase1": phase1,
            "phase2": phase2,
        }

    def _phase1_coarse_servo(
        self,
        target_x_hint: float,
        target_y_hint: float,
        collector=None,
    ) -> dict:
        """
        Phase 1: Move gripper toward target with tilt, using visual feedback.

        Returns dict with success, converged, iterations, final_x, final_y.
        """
        cfg = self._config
        steps_log = []

        # Get camera client
        from camera_daemon.client import get_camera_client
        camera = get_camera_client()

        # Standard picking orientation with forward tilt
        pick_roll = math.pi
        pick_yaw = 0.0

        # Start position: offset from hint, at servo height, tilted
        start_x = target_x_hint
        start_y = target_y_hint

        # Open gripper
        self._controller.gripper_move(0.08)

        # Move to servo start position (tilted)
        result = self._controller.move_cartesian_ik(
            start_x, start_y, cfg.servo_z,
            roll=pick_roll, pitch=cfg.servo_pitch, yaw=pick_yaw,
            confirmed=True,
        )
        if not result.get("success"):
            return {
                "success": False,
                "error": f"Failed to reach servo start: {result}",
                "converged": False,
                "iterations": 0,
                "final_x": start_x,
                "final_y": start_y,
            }

        # Current estimate of where the target is
        target_estimate = [target_x_hint, target_y_hint]
        converged = False

        for i in range(cfg.max_iterations):
            # Small delay for robot to settle + camera to update
            time.sleep(cfg.step_delay)

            # Get current gripper position
            state = self._controller.get_state()
            gripper_x = state.ee_position[0]
            gripper_y = state.ee_position[1]
            gripper_z = state.ee_position[2]

            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning(f"Servo iter {i}: no camera frame")
                steps_log.append({"iter": i, "error": "no_frame"})
                continue

            # Predict offset (model or fallback)
            dx, dy, scale, pred_scale = self._predict_offset(
                frame, (gripper_x, gripper_y), tuple(target_estimate)
            )

            # Record frame for data collection
            if collector and collector.active:
                collector.record_frame(
                    frame=frame,
                    gripper_robot_xy=(gripper_x, gripper_y),
                    target_estimate_xy=tuple(target_estimate),
                    gripper_z=gripper_z,
                    pitch=cfg.servo_pitch,
                )

            magnitude = math.sqrt(dx * dx + dy * dy)

            step_info = {
                "iter": i,
                "gripper_x": round(gripper_x, 4),
                "gripper_y": round(gripper_y, 4),
                "dx_mm": round(dx * 1000, 1),
                "dy_mm": round(dy * 1000, 1),
                "magnitude_mm": round(magnitude * 1000, 1),
                "scale": round(scale, 3),
                "model_used": self.has_model,
            }
            if pred_scale is not None:
                step_info["pred_scale"] = round(pred_scale, 3)
            steps_log.append(step_info)

            logger.info(
                f"Servo iter {i}: dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm "
                f"mag={magnitude*1000:.1f}mm scale={scale:.3f}"
            )

            # Check convergence (but enforce min_iterations for data diversity)
            if magnitude < cfg.convergence_threshold and i >= cfg.min_iterations - 1:
                converged = True
                break

            # Apply movement (gain-damped, clamped)
            move_dx = max(-cfg.max_step, min(cfg.max_step, cfg.gain * dx))
            move_dy = max(-cfg.max_step, min(cfg.max_step, cfg.gain * dy))

            new_x = gripper_x + move_dx
            new_y = gripper_y + move_dy

            result = self._controller.move_cartesian_ik(
                new_x, new_y, cfg.servo_z,
                roll=pick_roll, pitch=cfg.servo_pitch, yaw=pick_yaw,
                confirmed=True,
            )

            if not result.get("success"):
                logger.warning(f"Servo move failed at iter {i}: {result}")
                # Don't break, try next iteration from current position

            # Update target estimate for next crop center
            target_estimate[0] = gripper_x + dx
            target_estimate[1] = gripper_y + dy

        # Final position
        state = self._controller.get_state()
        final_x = state.ee_position[0]
        final_y = state.ee_position[1]

        return {
            "success": True,
            "converged": converged,
            "iterations": len(steps_log),
            "final_x": round(final_x, 4),
            "final_y": round(final_y, 4),
            "steps": steps_log,
        }

    def _phase2_fine_grasp(
        self,
        x: float,
        y: float,
        grasp_width: float,
        grasp_force: float,
        grasp_z: float,
        approach_height: float,
    ) -> dict:
        """
        Phase 2: Untilt to vertical, lower, grasp, lift.

        Reuses the incremental lowering logic from pick_at().
        """
        steps = []
        pick_roll = math.pi
        pick_pitch = 0.0
        pick_yaw = 0.0

        # Untilt to vertical at current height
        state = self._controller.get_state()
        result = self._controller.move_cartesian_ik(
            x, y, state.ee_position[2],
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw,
            confirmed=True,
        )
        steps.append({"action": "untilt", "result": result})

        # Lower in 4cm increments (same pattern as pick_at)
        state = self._controller.get_state()
        current_z = state.ee_position[2]
        step_z = 0.04

        while current_z - step_z > grasp_z + step_z:
            intermediate_z = current_z - step_z
            result = self._controller.move_cartesian_ik(
                x, y, intermediate_z,
                roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw,
                confirmed=True,
            )
            steps.append({"action": "lower_step", "target_z": round(intermediate_z, 4), "result": result})
            state = self._controller.get_state()
            current_z = state.ee_position[2]

        # Final lower to grasp height
        result = self._controller.move_cartesian_ik(
            x, y, grasp_z,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw,
            confirmed=True,
        )
        steps.append({"action": "lower_to_grasp", "result": result})

        # Grasp
        result = self._controller.gripper_grasp(width=grasp_width, force=grasp_force)
        steps.append({"action": "grasp", "result": result})

        # Check grasp
        state = self._controller.get_state()
        actual_grip = state.gripper_width
        min_grip = 0.001
        grasped = actual_grip > min_grip and actual_grip < grasp_width + 0.005
        steps.append({"action": "check_grasp", "gripper_width": round(actual_grip, 4), "grasped": grasped})

        # Lift
        state = self._controller.get_state()
        result = self._controller.move_cartesian_ik(
            state.ee_position[0], state.ee_position[1],
            state.ee_position[2] + approach_height,
            roll=pick_roll, pitch=pick_pitch, yaw=pick_yaw,
            confirmed=True,
        )
        steps.append({"action": "lift", "result": result})

        state = self._controller.get_state()
        return {
            "grasped": grasped,
            "gripper_width": round(actual_grip, 4),
            "final_position": {
                "x": round(state.ee_position[0], 4),
                "y": round(state.ee_position[1], 4),
                "z": round(state.ee_position[2], 4),
            },
            "steps": steps,
        }

    def _predict_offset(
        self,
        frame: np.ndarray,
        gripper_xy: Tuple[float, float],
        target_estimate: Tuple[float, float],
    ) -> Tuple[float, float, float, Optional[float]]:
        """
        Predict gripper-to-target offset.

        If model is loaded, uses ONNX inference.
        Otherwise, falls back to hint-based estimate (target - gripper).

        Returns:
            (dx, dy, crop_scale, pred_scale_or_None)
        """
        # Compute crop for either mode
        crop, scale = self._cropper.compute_servo_crop(
            frame, target_estimate, gripper_xy
        )

        if self._session is not None:
            # Model inference
            image_input = self._preprocess(crop)
            scale_input = np.array([[scale]], dtype=np.float32)

            outputs = self._session.run(None, {
                "image": image_input,
                "crop_scale": scale_input,
            })

            dx, dy = outputs[0][0]
            pred_scale = float(outputs[1][0][0])
            return float(dx), float(dy), float(scale), pred_scale
        else:
            # Fallback: move toward target estimate
            dx = target_estimate[0] - gripper_xy[0]
            dy = target_estimate[1] - gripper_xy[1]
            return float(dx), float(dy), float(scale), None

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for ONNX inference (ImageNet normalization)."""
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img[np.newaxis]  # add batch dim
