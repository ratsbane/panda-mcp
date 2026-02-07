"""
MCP Server for Franka Panda robot arm control.

Exposes tools for Claude to control the arm safely.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .controller import get_controller, FrankaController
from common.safety import get_safety_config, update_safety_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("franka-mcp")


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for robot control."""
    return [
        Tool(
            name="connect",
            description="Connect to the Franka Panda robot. Must be called before other commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "robot_ip": {
                        "type": "string",
                        "description": "Robot IP address (default: 192.168.0.1)",
                    }
                },
            },
        ),
        Tool(
            name="get_status",
            description="Get current robot state including joint positions, end effector pose, gripper width, and error status.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="move_cartesian",
            description="Move end effector to a Cartesian position. Coordinates in meters relative to robot base. "
                       "X is forward, Y is left/right, Z is up. Orientation in radians (roll, pitch, yaw).",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X position (meters, forward from base)"},
                    "y": {"type": "number", "description": "Y position (meters, left/right)"},
                    "z": {"type": "number", "description": "Z position (meters, height)"},
                    "roll": {"type": "number", "description": "Roll angle (radians, optional)"},
                    "pitch": {"type": "number", "description": "Pitch angle (radians, optional)"},
                    "yaw": {"type": "number", "description": "Yaw angle (radians, optional)"},
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set to true to confirm large moves (>20cm)",
                        "default": False,
                    },
                },
                "required": ["x", "y", "z"],
            },
        ),
        Tool(
            name="move_relative",
            description="Move end effector relative to current position. Deltas in meters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dx": {"type": "number", "description": "X delta (meters)", "default": 0},
                    "dy": {"type": "number", "description": "Y delta (meters)", "default": 0},
                    "dz": {"type": "number", "description": "Z delta (meters)", "default": 0},
                    "confirmed": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="move_joints",
            description="Move to a specific joint configuration. Panda has 7 joints.",
            inputSchema={
                "type": "object",
                "properties": {
                    "joints": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 7,
                        "maxItems": 7,
                        "description": "Joint angles in radians [q1, q2, q3, q4, q5, q6, q7]",
                    },
                    "confirmed": {"type": "boolean", "default": False},
                },
                "required": ["joints"],
            },
        ),
        Tool(
            name="gripper_move",
            description="Move gripper to specified width. Range: 0 (closed) to 0.08m (fully open).",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "number",
                        "description": "Target width in meters (0 to 0.08)",
                        "minimum": 0,
                        "maximum": 0.08,
                    },
                },
                "required": ["width"],
            },
        ),
        Tool(
            name="gripper_grasp",
            description="Grasp an object with specified width and force.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "number",
                        "description": "Target grasp width (meters)",
                    },
                    "force": {
                        "type": "number",
                        "description": "Grasp force in Newtons (0.01 to 70)",
                        "default": 20,
                    },
                    "speed": {
                        "type": "number",
                        "description": "Closing speed (m/s)",
                        "default": 0.1,
                    },
                },
                "required": ["width"],
            },
        ),
        Tool(
            name="move_cartesian_sequence",
            description="Execute a sequence of Cartesian waypoints as smooth continuous motion. "
                       "Eliminates gaps between movements for fluid trajectories like waving or gestures.",
            inputSchema={
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number", "description": "X position (meters)"},
                                "y": {"type": "number", "description": "Y position (meters)"},
                                "z": {"type": "number", "description": "Z position (meters)"},
                                "roll": {"type": "number", "description": "Roll (radians, optional)"},
                                "pitch": {"type": "number", "description": "Pitch (radians, optional)"},
                                "yaw": {"type": "number", "description": "Yaw (radians, optional)"},
                            },
                            "required": ["x", "y", "z"],
                        },
                        "description": "List of waypoints to traverse smoothly",
                    },
                    "speed_factor": {
                        "type": "number",
                        "description": "Motion speed (0.0 to 1.0, default 0.1)",
                        "default": 0.1,
                        "minimum": 0.01,
                        "maximum": 1.0,
                    },
                },
                "required": ["waypoints"],
            },
        ),
        Tool(
            name="move_joint_sequence",
            description="Execute a sequence of joint configurations as smooth motion. "
                       "Each configuration is 7 joint angles in radians.",
            inputSchema={
                "type": "object",
                "properties": {
                    "configurations": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 7,
                            "maxItems": 7,
                        },
                        "description": "List of joint configurations [q1..q7] to traverse",
                    },
                    "speed_factor": {
                        "type": "number",
                        "description": "Motion speed (0.0 to 1.0, default 0.2)",
                        "default": 0.2,
                        "minimum": 0.01,
                        "maximum": 1.0,
                    },
                },
                "required": ["configurations"],
            },
        ),
        Tool(
            name="pick_at",
            description="Pick an object at the given robot coordinates. "
                       "Opens gripper, approaches from above, lowers to grasp height, grasps, and lifts. "
                       "Use with describe_scene robot_coords to automate pick-and-place.",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X position in robot frame (meters)"},
                    "y": {"type": "number", "description": "Y position in robot frame (meters)"},
                    "z": {"type": "number", "description": "Grasp height (meters, default: 0.013 = table)", "default": 0.013},
                    "grasp_width": {"type": "number", "description": "Expected object width (meters, default: 0.03)", "default": 0.03},
                    "grasp_force": {"type": "number", "description": "Grasp force in Newtons (default: 70)", "default": 70},
                    "x_offset": {"type": "number", "description": "X offset to compensate calibration error (meters, default: 0.04)", "default": 0.04},
                    "approach_height": {"type": "number", "description": "Height to approach from (meters, default: 0.15)", "default": 0.15},
                },
                "required": ["x", "y"],
            },
        ),
        Tool(
            name="place_at",
            description="Place a held object at the given robot coordinates. "
                       "Moves above target, lowers, releases gripper, and retreats upward.",
            inputSchema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X position in robot frame (meters)"},
                    "y": {"type": "number", "description": "Y position in robot frame (meters)"},
                    "z": {"type": "number", "description": "Place height (meters, default: 0.08)", "default": 0.08},
                    "approach_height": {"type": "number", "description": "Height to approach/retreat from (meters, default: 0.15)", "default": 0.15},
                },
                "required": ["x", "y"],
            },
        ),
        Tool(
            name="stop",
            description="Immediately stop any current motion. Use this if something looks wrong.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="recover",
            description="Recover from error state. Call this if the robot has stopped due to an error.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="teaching_mode",
            description="Enable/disable teaching mode (gravity compensation). "
                       "When active, the arm goes compliant and can be physically moved by hand. "
                       "Useful for recovering from joint limit issues or manual positioning.",
            inputSchema={
                "type": "object",
                "properties": {
                    "active": {
                        "type": "boolean",
                        "description": "True to enable, False to disable",
                    },
                },
                "required": ["active"],
            },
        ),
        Tool(
            name="start_recording",
            description="Begin recording a trajectory demonstration for VLA training. "
                       "Records robot joint states + camera frames at target FPS in background. "
                       "Call this BEFORE executing pick_at/place_at, then stop_recording when done.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language task description (e.g. 'pick up the red block and place it to the right')",
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Recording frame rate (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["instruction"],
            },
        ),
        Tool(
            name="stop_recording",
            description="Stop trajectory recording and save the episode to disk. "
                       "Returns episode stats (frames, duration, path).",
            inputSchema={
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the demonstration was successful",
                    },
                },
                "required": ["success"],
            },
        ),
        Tool(
            name="get_recording_status",
            description="Check if trajectory recording is active and get current stats.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_episodes",
            description="List all recorded trajectory episodes with metadata.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_safety_limits",
            description="Get current safety limits (workspace bounds, velocity limits).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="set_safety_limits",
            description="Update safety limits. Use with caution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "object",
                        "description": "Workspace bounds: {x_min, x_max, y_min, y_max, z_min, z_max}",
                    },
                    "velocity": {
                        "type": "object",
                        "description": "Velocity limits: {translation_m_s, rotation_rad_s}",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Enable dry run mode (no actual movement)",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    controller = get_controller()
    
    try:
        if name == "connect":
            robot_ip = arguments.get("robot_ip")
            if robot_ip:
                controller.robot_ip = robot_ip
            result = controller.connect()
            return json_response(result)
        
        elif name == "get_status":
            if not controller.connected:
                return json_response({"error": "Not connected. Call 'connect' first."})
            state = controller.get_state()
            return json_response(state.to_dict())
        
        elif name == "move_cartesian":
            result = controller.move_cartesian(
                x=arguments["x"],
                y=arguments["y"],
                z=arguments["z"],
                roll=arguments.get("roll"),
                pitch=arguments.get("pitch"),
                yaw=arguments.get("yaw"),
                confirmed=arguments.get("confirmed", False),
            )
            return json_response(result)
        
        elif name == "move_relative":
            result = controller.move_relative(
                dx=arguments.get("dx", 0),
                dy=arguments.get("dy", 0),
                dz=arguments.get("dz", 0),
                confirmed=arguments.get("confirmed", False),
            )
            return json_response(result)
        
        elif name == "move_joints":
            result = controller.move_joints(
                joints=arguments["joints"],
                confirmed=arguments.get("confirmed", False),
            )
            return json_response(result)
        
        elif name == "move_cartesian_sequence":
            result = controller.move_cartesian_sequence(
                waypoints=arguments["waypoints"],
                speed_factor=arguments.get("speed_factor", 0.1),
            )
            return json_response(result)

        elif name == "move_joint_sequence":
            result = controller.move_joint_sequence(
                configurations=arguments["configurations"],
                speed_factor=arguments.get("speed_factor", 0.2),
            )
            return json_response(result)

        elif name == "gripper_move":
            result = controller.gripper_move(width=arguments["width"])
            return json_response(result)
        
        elif name == "gripper_grasp":
            result = controller.gripper_grasp(
                width=arguments["width"],
                force=arguments.get("force", 20),
                speed=arguments.get("speed", 0.1),
            )
            return json_response(result)
        
        elif name == "pick_at":
            result = controller.pick_at(
                x=arguments["x"],
                y=arguments["y"],
                z=arguments.get("z", 0.013),
                grasp_width=arguments.get("grasp_width", 0.03),
                grasp_force=arguments.get("grasp_force", 70),
                x_offset=arguments.get("x_offset", 0.04),
                approach_height=arguments.get("approach_height", 0.15),
            )
            return json_response(result)

        elif name == "place_at":
            result = controller.place_at(
                x=arguments["x"],
                y=arguments["y"],
                z=arguments.get("z", 0.08),
                approach_height=arguments.get("approach_height", 0.15),
            )
            return json_response(result)

        elif name == "stop":
            result = controller.stop()
            return json_response(result)
        
        elif name == "recover":
            result = controller.recover()
            return json_response(result)

        elif name == "teaching_mode":
            result = controller.teaching_mode(active=arguments["active"])
            return json_response(result)

        elif name == "start_recording":
            result = controller.start_recording(
                language_instruction=arguments["instruction"],
                fps=arguments.get("fps", 30),
            )
            return json_response(result)

        elif name == "stop_recording":
            result = controller.stop_recording(
                success=arguments["success"],
            )
            return json_response(result)

        elif name == "get_recording_status":
            result = controller.get_recording_status()
            return json_response(result)

        elif name == "list_episodes":
            result = controller.list_episodes()
            return json_response(result)

        elif name == "get_safety_limits":
            config = get_safety_config()
            return json_response(config.to_dict())
        
        elif name == "set_safety_limits":
            config = update_safety_config(
                workspace=arguments.get("workspace"),
                velocity=arguments.get("velocity"),
                dry_run=arguments.get("dry_run"),
            )
            return json_response({
                "success": True,
                "updated_config": config.to_dict(),
            })
        
        else:
            return json_response({"error": f"Unknown tool: {name}"})
    
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    logger.info("Starting Franka MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_entry():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()
