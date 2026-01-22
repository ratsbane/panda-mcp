"""
MCP-integrated data collector.

Allows Claude to collect training data while interacting with the robot.
Can be used alongside normal MCP tool calls.
"""

import time
from pathlib import Path
from typing import Optional

from .data_collection import (
    DataCollector,
    RobotState,
    create_robot_state_from_status,
)


class MCPDataCollector:
    """
    Singleton data collector for use during MCP sessions.

    Usage:
        collector = get_mcp_collector()
        collector.start_session("exploration_session")
        ...
        # During normal operation, call record() after moves
        collector.record(image, robot_status)
        ...
        collector.end_session()
    """

    _instance = None

    def __init__(self):
        self.collector: Optional[DataCollector] = None
        self.is_recording = False
        self.auto_record = False  # If True, record every move automatically

    @classmethod
    def get_instance(cls) -> "MCPDataCollector":
        if cls._instance is None:
            cls._instance = MCPDataCollector()
        return cls._instance

    def start_session(
        self,
        session_name: Optional[str] = None,
        output_dir: str = "./training_data",
        note: str = "",
    ) -> dict:
        """Start a new data collection session."""
        if self.collector is not None and self.is_recording:
            return {
                "success": False,
                "error": "Session already in progress. End it first.",
            }

        self.collector = DataCollector(output_dir, session_name)
        self.is_recording = True

        if note:
            self.collector.add_note(note)

        return {
            "success": True,
            "session_id": self.collector.session_id,
            "session_dir": str(self.collector.session_dir),
        }

    def end_session(self) -> dict:
        """End current session and save data."""
        if self.collector is None or not self.is_recording:
            return {"success": False, "error": "No session in progress"}

        self.collector.save_session()
        sample_count = self.collector.sample_count
        session_dir = str(self.collector.session_dir)

        self.collector = None
        self.is_recording = False

        return {
            "success": True,
            "samples_collected": sample_count,
            "session_dir": session_dir,
        }

    def record(self, image, robot_status: dict) -> dict:
        """Record a single sample."""
        if self.collector is None or not self.is_recording:
            return {"success": False, "error": "No session in progress"}

        robot_state = create_robot_state_from_status(robot_status)
        sample = self.collector.record_sample(image, robot_state)

        return {
            "success": True,
            "sample_id": sample.sample_id,
            "total_samples": self.collector.sample_count,
            "camera_pose_confidence": sample.camera_pose.confidence if sample.camera_pose else None,
        }

    def add_marker(self, marker_id: int, position: tuple, rotation: tuple = (0, 0, 0)) -> dict:
        """Register an ArUco marker position."""
        if self.collector is None:
            return {"success": False, "error": "No session in progress"}

        self.collector.add_marker(marker_id, position, rotation)
        return {"success": True, "marker_id": marker_id}

    def get_status(self) -> dict:
        """Get current collection status."""
        if self.collector is None or not self.is_recording:
            return {
                "recording": False,
                "session_id": None,
                "sample_count": 0,
            }

        return {
            "recording": True,
            "session_id": self.collector.session_id,
            "sample_count": self.collector.sample_count,
            "session_dir": str(self.collector.session_dir),
            "auto_record": self.auto_record,
        }

    def set_auto_record(self, enabled: bool) -> dict:
        """Enable/disable automatic recording after every move."""
        self.auto_record = enabled
        return {"success": True, "auto_record": enabled}

    def add_note(self, note: str) -> dict:
        """Add a note to the current session."""
        if self.collector is None:
            return {"success": False, "error": "No session in progress"}

        self.collector.add_note(note)
        return {"success": True}


def get_mcp_collector() -> MCPDataCollector:
    """Get the singleton MCP data collector."""
    return MCPDataCollector.get_instance()


# Example of how to integrate with MCP tool calls
"""
# In franka_mcp/server.py, after a successful move:

from common.mcp_data_collector import get_mcp_collector

collector = get_mcp_collector()
if collector.is_recording and collector.auto_record:
    # Capture image
    camera = get_camera_controller()
    image = camera.capture_raw()

    # Get robot state
    robot_status = controller.get_status()

    # Record sample
    collector.record(image, robot_status)
"""
