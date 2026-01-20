"""Tests for safety constraints."""

import pytest
from common.safety import (
    SafetyConfig,
    SafetyValidator,
    WorkspaceLimits,
    VelocityLimits,
)


class TestWorkspaceLimits:
    def test_default_limits(self):
        limits = WorkspaceLimits()
        assert limits.x_min == 0.2
        assert limits.x_max == 0.75
        assert limits.z_min == 0.05
    
    def test_contains_inside(self):
        limits = WorkspaceLimits()
        assert limits.contains(0.4, 0.0, 0.3)
    
    def test_contains_outside_x(self):
        limits = WorkspaceLimits()
        assert not limits.contains(0.1, 0.0, 0.3)  # Too close to base
        assert not limits.contains(0.9, 0.0, 0.3)  # Too far
    
    def test_contains_outside_z(self):
        limits = WorkspaceLimits()
        assert not limits.contains(0.4, 0.0, 0.01)  # Too low (would hit table)
        assert not limits.contains(0.4, 0.0, 0.8)   # Too high
    
    def test_clamp(self):
        limits = WorkspaceLimits()
        x, y, z = limits.clamp(0.1, 0.0, 0.01)
        assert x == 0.2   # Clamped to x_min
        assert y == 0.0   # Unchanged
        assert z == 0.05  # Clamped to z_min


class TestSafetyValidator:
    def test_valid_cartesian_target(self):
        validator = SafetyValidator()
        result = validator.validate_cartesian_target(0.4, 0.0, 0.3)
        assert result["valid"]
        assert len(result["errors"]) == 0
    
    def test_cartesian_target_clamped(self):
        validator = SafetyValidator()
        result = validator.validate_cartesian_target(0.1, 0.0, 0.01)
        assert result["valid"]
        assert result["clamped_position"] is not None
        assert len(result["warnings"]) > 0
    
    def test_large_move_requires_confirmation(self):
        validator = SafetyValidator()
        # Large move from origin
        result = validator.validate_cartesian_target(
            0.4, 0.0, 0.5,
            current_position=(0.4, 0.0, 0.2)  # 30cm move in Z
        )
        assert result["requires_confirmation"]
    
    def test_valid_joint_target(self):
        validator = SafetyValidator()
        joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        result = validator.validate_joint_target(joints)
        assert result["valid"]
    
    def test_invalid_joint_target_wrong_count(self):
        validator = SafetyValidator()
        joints = [0.0, 0.0, 0.0]  # Only 3 joints
        result = validator.validate_joint_target(joints)
        assert not result["valid"]
    
    def test_invalid_joint_target_out_of_range(self):
        validator = SafetyValidator()
        joints = [5.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0]  # Joint 1 out of range
        result = validator.validate_joint_target(joints)
        assert not result["valid"]
    
    def test_valid_gripper_command(self):
        validator = SafetyValidator()
        result = validator.validate_gripper_command(0.04)
        assert result["valid"]
    
    def test_invalid_gripper_width(self):
        validator = SafetyValidator()
        result = validator.validate_gripper_command(0.1)  # Too wide
        assert not result["valid"]


class TestSafetyConfig:
    def test_default_config(self):
        config = SafetyConfig()
        assert config.dry_run == False
        assert config.require_confirmation_for_large_moves == True
    
    def test_to_dict(self):
        config = SafetyConfig()
        d = config.to_dict()
        assert "workspace" in d
        assert "velocity" in d
        assert "dry_run" in d
