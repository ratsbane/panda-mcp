"""
Safety constraints and validation for Franka Panda arm control.

These constraints are enforced BEFORE any command is sent to the arm.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class WorkspaceLimits:
    """Cartesian workspace bounding box (meters, relative to robot base)."""

    x_min: float = -0.4   # Allow positions behind/beside base
    x_max: float = 0.75   # Forward reach limit
    y_min: float = -0.5   # Left (from robot's perspective)
    y_max: float = 0.5    # Right
    z_min: float = 0.05   # Don't hit the table
    z_max: float = 0.7    # Upper limit
    
    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within the workspace."""
        return (
            self.x_min <= x <= self.x_max and
            self.y_min <= y <= self.y_max and
            self.z_min <= z <= self.z_max
        )
    
    def clamp(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Clamp a point to within the workspace."""
        return (
            np.clip(x, self.x_min, self.x_max),
            np.clip(y, self.y_min, self.y_max),
            np.clip(z, self.z_min, self.z_max),
        )
    
    def to_dict(self) -> dict:
        return {
            "x": {"min": self.x_min, "max": self.x_max},
            "y": {"min": self.y_min, "max": self.y_max},
            "z": {"min": self.z_min, "max": self.z_max},
        }


@dataclass
class VelocityLimits:
    """Maximum velocities for safe operation."""
    
    # Cartesian velocity (m/s)
    max_translation_velocity: float = 0.1
    max_rotation_velocity: float = 0.5  # rad/s
    
    # Joint velocities (rad/s) - Panda has 7 joints
    max_joint_velocities: list[float] = field(default_factory=lambda: [
        2.0,  # Joint 1
        2.0,  # Joint 2  
        2.0,  # Joint 3
        2.0,  # Joint 4
        2.5,  # Joint 5
        2.5,  # Joint 6
        2.5,  # Joint 7
    ])
    
    # These are already conservative vs Panda's actual limits
    # (actual max is ~2.6 rad/s for joints 1-4, ~3.0 for 5-7)
    
    def to_dict(self) -> dict:
        return {
            "translation_m_s": self.max_translation_velocity,
            "rotation_rad_s": self.max_rotation_velocity,
            "joints_rad_s": self.max_joint_velocities,
        }


@dataclass 
class SafetyConfig:
    """Complete safety configuration."""
    
    workspace: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    velocity: VelocityLimits = field(default_factory=VelocityLimits)
    
    # Require explicit confirmation for certain operations
    require_confirmation_for_large_moves: bool = True
    large_move_threshold_m: float = 0.2  # 20cm
    
    # Dry run mode - report what would happen without executing
    dry_run: bool = False
    
    def to_dict(self) -> dict:
        return {
            "workspace": self.workspace.to_dict(),
            "velocity": self.velocity.to_dict(),
            "require_confirmation_for_large_moves": self.require_confirmation_for_large_moves,
            "large_move_threshold_m": self.large_move_threshold_m,
            "dry_run": self.dry_run,
        }


class SafetyValidator:
    """Validates commands against safety constraints."""
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
    
    def validate_cartesian_target(
        self, 
        x: float, 
        y: float, 
        z: float,
        current_position: Optional[tuple[float, float, float]] = None,
    ) -> dict:
        """
        Validate a Cartesian target position.
        
        Returns:
            dict with keys:
                - valid: bool
                - warnings: list[str]
                - errors: list[str]
                - requires_confirmation: bool
                - clamped_position: Optional[tuple] if position was adjusted
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "requires_confirmation": False,
            "clamped_position": None,
        }
        
        # Check workspace bounds
        if not self.config.workspace.contains(x, y, z):
            clamped = self.config.workspace.clamp(x, y, z)
            result["warnings"].append(
                f"Target ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace. "
                f"Clamped to ({clamped[0]:.3f}, {clamped[1]:.3f}, {clamped[2]:.3f})"
            )
            result["clamped_position"] = clamped
            x, y, z = clamped
        
        # Check move distance if current position known
        if current_position is not None:
            distance = np.sqrt(
                (x - current_position[0])**2 +
                (y - current_position[1])**2 +
                (z - current_position[2])**2
            )
            
            if distance > self.config.large_move_threshold_m:
                if self.config.require_confirmation_for_large_moves:
                    result["requires_confirmation"] = True
                    result["warnings"].append(
                        f"Large move detected: {distance:.3f}m. Confirmation required."
                    )
        
        return result
    
    def validate_joint_target(
        self,
        joints: list[float],
        current_joints: Optional[list[float]] = None,
    ) -> dict:
        """Validate joint configuration target."""
        
        # Panda joint limits (radians)
        joint_limits = [
            (-2.8973, 2.8973),   # Joint 1
            (-1.7628, 1.7628),   # Joint 2
            (-2.8973, 2.8973),   # Joint 3
            (-3.0718, -0.0698),  # Joint 4
            (-2.8973, 2.8973),   # Joint 5
            (-0.0175, 3.7525),   # Joint 6
            (-2.8973, 2.8973),   # Joint 7
        ]
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "requires_confirmation": False,
        }
        
        if len(joints) != 7:
            result["valid"] = False
            result["errors"].append(f"Expected 7 joints, got {len(joints)}")
            return result
        
        for i, (q, (q_min, q_max)) in enumerate(zip(joints, joint_limits)):
            if not (q_min <= q <= q_max):
                result["valid"] = False
                result["errors"].append(
                    f"Joint {i+1} value {q:.4f} outside limits [{q_min:.4f}, {q_max:.4f}]"
                )
        
        return result
    
    def validate_gripper_command(
        self,
        width: float,
        force: Optional[float] = None,
    ) -> dict:
        """Validate gripper command."""
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Panda gripper: 0 to 0.08m width
        if not (0.0 <= width <= 0.08):
            result["valid"] = False
            result["errors"].append(
                f"Gripper width {width:.4f}m outside range [0, 0.08]"
            )
        
        # Force: 0.01 to 70N for grasping
        if force is not None and not (0.01 <= force <= 70):
            result["valid"] = False  
            result["errors"].append(
                f"Gripper force {force:.2f}N outside range [0.01, 70]"
            )
        
        return result


# Singleton default config
_default_safety_config = SafetyConfig()


def get_safety_config() -> SafetyConfig:
    return _default_safety_config


def update_safety_config(
    workspace: Optional[dict] = None,
    velocity: Optional[dict] = None,
    dry_run: Optional[bool] = None,
) -> SafetyConfig:
    """Update the global safety configuration."""
    global _default_safety_config
    
    if workspace:
        for key, value in workspace.items():
            if hasattr(_default_safety_config.workspace, key):
                setattr(_default_safety_config.workspace, key, value)
    
    if velocity:
        if "translation_m_s" in velocity:
            _default_safety_config.velocity.max_translation_velocity = velocity["translation_m_s"]
        if "rotation_rad_s" in velocity:
            _default_safety_config.velocity.max_rotation_velocity = velocity["rotation_rad_s"]
    
    if dry_run is not None:
        _default_safety_config.dry_run = dry_run
    
    return _default_safety_config
