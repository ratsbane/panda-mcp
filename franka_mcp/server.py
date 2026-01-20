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
        
        elif name == "stop":
            result = controller.stop()
            return json_response(result)
        
        elif name == "recover":
            result = controller.recover()
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


if __name__ == "__main__":
    asyncio.run(main())
