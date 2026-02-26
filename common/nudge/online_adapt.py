"""
Online adaptation layer for NUDGE -- learns from each pick attempt in real time.

Maintains a rolling buffer of recent pick outcomes (target vs actual position)
and computes a running correction offset. This catches systematic errors like
calibration drift, homography bias, and consistent IK offsets without retraining.

The simplest form of online learning: exponential moving average of error vectors.
Can be upgraded to gradient-based adaptation once the signal proves useful.

Usage:
    adapter = OnlineAdapter()
    adapter.enable()

    # Before each pick:
    dx, dy = adapter.get_correction(target_x, target_y)
    actual_target_x = target_x + dx
    actual_target_y = target_y + dy

    # After each pick:
    adapter.record_outcome(target_x, target_y, actual_x, actual_y, success)
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "online_adapt"


@dataclass
class OnlineAdaptConfig:
    """Configuration for online adaptation."""
    # Rolling buffer size (number of recent outcomes to consider)
    buffer_size: int = 10

    # Exponential moving average decay (0 = no memory, 1 = infinite memory)
    # 0.7 means ~70% weight on history, ~30% on new observation
    ema_alpha: float = 0.7

    # Maximum correction magnitude (meters) -- safety clamp
    max_correction_m: float = 0.025  # 25mm

    # Minimum number of successful picks before applying corrections
    min_samples: int = 3

    # Only use successful picks for adaptation (failed picks may have bad position data)
    success_only: bool = True


class OnlineAdapter:
    """
    Learns a position correction offset from each pick attempt.

    After each pick, records the error between target position and actual grasp
    position. Maintains an exponential moving average of these errors and applies
    the correction to future picks.

    This is the simplest possible online learning: a running calibration offset.
    It catches systematic biases (homography error, IK drift) without any model
    training. Think of it as "auto-calibration from experience."
    """

    def __init__(self, config: OnlineAdaptConfig = OnlineAdaptConfig()):
        self.config = config
        self._enabled = False

        # Rolling buffer of recent outcomes
        self._history = deque(maxlen=config.buffer_size)

        # Current EMA correction (x, y in meters)
        self._ema_dx = 0.0
        self._ema_dy = 0.0

        # Stats
        self._total_picks = 0
        self._total_successful = 0
        self._total_corrections_applied = 0
        self._session_start = None

        # Persistent log
        self._data_dir = DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._data_dir / "adaptation_log.jsonl"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> dict:
        """Enable online adaptation."""
        self._enabled = True
        self._session_start = time.time()
        logger.info(
            f"Online adaptation enabled: buffer={self.config.buffer_size}, "
            f"ema_alpha={self.config.ema_alpha}, max_correction={self.config.max_correction_m*1000:.0f}mm"
        )
        return self.get_status()

    def disable(self) -> dict:
        """Disable online adaptation."""
        status = self.get_status()
        self._enabled = False
        logger.info(
            f"Online adaptation disabled: {self._total_picks} picks, "
            f"final offset: dx={self._ema_dx*1000:.1f}mm dy={self._ema_dy*1000:.1f}mm"
        )
        return status

    def reset(self):
        """Clear all history and reset EMA."""
        self._history.clear()
        self._ema_dx = 0.0
        self._ema_dy = 0.0
        self._total_picks = 0
        self._total_successful = 0
        self._total_corrections_applied = 0
        logger.info("Online adaptation reset")

    def get_correction(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Get the correction offset to apply before a pick.

        Returns (dx, dy) in meters to ADD to the target position.
        Returns (0, 0) if adaptation is disabled or insufficient data.
        """
        if not self._enabled:
            return (0.0, 0.0)

        # Need minimum samples before applying corrections
        n_usable = sum(1 for h in self._history if h["success"]) if self.config.success_only else len(self._history)
        if n_usable < self.config.min_samples:
            return (0.0, 0.0)

        # Safety clamp
        dx = max(-self.config.max_correction_m, min(self.config.max_correction_m, self._ema_dx))
        dy = max(-self.config.max_correction_m, min(self.config.max_correction_m, self._ema_dy))

        if abs(dx) > 0.001 or abs(dy) > 0.001:
            self._total_corrections_applied += 1
            logger.info(
                f"Online adapt correction: dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm "
                f"(from {n_usable} samples)"
            )

        return (dx, dy)

    def record_outcome(
        self,
        target_x: float,
        target_y: float,
        actual_x: float,
        actual_y: float,
        success: bool,
        gripper_width: float = 0.0,
        target_z: float = 0.013,
        actual_z: float = 0.0,
    ):
        """
        Record the outcome of a pick attempt.

        The error is (actual - target): if the robot consistently lands 5mm to the
        right of where we aimed, error_y = +5mm, and we should correct by -5mm.

        Args:
            target_x, target_y: Where we aimed (robot frame, meters)
            actual_x, actual_y: Where we actually grasped (from FK)
            success: Whether the grasp succeeded
            gripper_width: Final gripper width
            target_z, actual_z: Z positions (for logging, not used in correction yet)
        """
        error_x = actual_x - target_x
        error_y = actual_y - target_y

        outcome = {
            "timestamp": time.time(),
            "target_x": round(target_x, 5),
            "target_y": round(target_y, 5),
            "actual_x": round(actual_x, 5),
            "actual_y": round(actual_y, 5),
            "error_x": round(error_x, 5),
            "error_y": round(error_y, 5),
            "target_z": round(target_z, 5),
            "actual_z": round(actual_z, 5),
            "success": success,
            "gripper_width": round(gripper_width, 5),
        }

        self._history.append(outcome)
        self._total_picks += 1
        if success:
            self._total_successful += 1

        # Update EMA (only from successful picks if configured)
        if success or not self.config.success_only:
            alpha = self.config.ema_alpha
            # EMA: new = alpha * old + (1 - alpha) * observation
            # The correction should NEGATE the error: if we land too far right,
            # correct by going left next time
            self._ema_dx = alpha * self._ema_dx + (1 - alpha) * (-error_x)
            self._ema_dy = alpha * self._ema_dy + (1 - alpha) * (-error_y)

        # Log to file
        self._log_outcome(outcome)

        logger.info(
            f"Online adapt: {'OK' if success else 'FAIL'} "
            f"error=({error_x*1000:.1f}, {error_y*1000:.1f})mm "
            f"ema_correction=({self._ema_dx*1000:.1f}, {self._ema_dy*1000:.1f})mm "
            f"[{self._total_successful}/{self._total_picks} successful]"
        )

    def _log_outcome(self, outcome: dict):
        """Append outcome to persistent JSONL log."""
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(outcome) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log outcome: {e}")

    def get_status(self) -> dict:
        """Get current adaptation status."""
        # Compute stats from buffer
        errors_x = [h["error_x"] for h in self._history if h["success"]]
        errors_y = [h["error_y"] for h in self._history if h["success"]]

        stats = {
            "enabled": self._enabled,
            "total_picks": self._total_picks,
            "total_successful": self._total_successful,
            "corrections_applied": self._total_corrections_applied,
            "buffer_size": len(self._history),
            "buffer_capacity": self.config.buffer_size,
            "current_correction_mm": {
                "dx": round(self._ema_dx * 1000, 1),
                "dy": round(self._ema_dy * 1000, 1),
            },
            "config": {
                "ema_alpha": self.config.ema_alpha,
                "max_correction_mm": self.config.max_correction_m * 1000,
                "min_samples": self.config.min_samples,
                "success_only": self.config.success_only,
            },
        }

        if errors_x:
            stats["recent_error_stats_mm"] = {
                "mean_x": round(np.mean(errors_x) * 1000, 1),
                "mean_y": round(np.mean(errors_y) * 1000, 1),
                "std_x": round(np.std(errors_x) * 1000, 1),
                "std_y": round(np.std(errors_y) * 1000, 1),
                "n_samples": len(errors_x),
            }

        if self._session_start:
            stats["session_duration_s"] = round(time.time() - self._session_start, 0)

        return stats


# Singleton
_online_adapter: Optional[OnlineAdapter] = None


def get_online_adapter() -> OnlineAdapter:
    global _online_adapter
    if _online_adapter is None:
        _online_adapter = OnlineAdapter()
    return _online_adapter
