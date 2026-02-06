"""
MCP Server for SO-ARM100 robot arm control.

Exposes tools for Claude to control the LeRobot SO-ARM100 arm.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .controller import get_controller, SO100Controller, MOTOR_NAMES, CENTER_POSITION, discover_ports

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("so100-mcp")


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for SO-ARM100 control."""
    return [
        Tool(
            name="connect",
            description="Connect to the SO-ARM100 robot arm. Must be called before other commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "port": {
                        "type": "string",
                        "description": "Serial port (default: /dev/ttyACM0)",
                        "default": "/dev/ttyACM0",
                    }
                },
            },
        ),
        Tool(
            name="get_status",
            description="Get current robot state including joint positions (ticks and degrees), voltages, temperatures, and torque status.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="move_joint",
            description="Move a single joint to a position. Position in ticks (0-4096, center=2048) or degrees from center.",
            inputSchema={
                "type": "object",
                "properties": {
                    "joint": {
                        "type": "string",
                        "description": "Joint name: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, or gripper",
                        "enum": MOTOR_NAMES,
                    },
                    "position": {
                        "type": "number",
                        "description": "Target position in ticks (0-4096, center=2048)",
                    },
                    "degrees": {
                        "type": "number",
                        "description": "Target position in degrees from center (alternative to position)",
                    },
                },
                "required": ["joint"],
            },
        ),
        Tool(
            name="move_joints",
            description="Move multiple joints simultaneously. Positions in ticks (0-4096, center=2048).",
            inputSchema={
                "type": "object",
                "properties": {
                    "positions": {
                        "type": "object",
                        "description": "Dict of joint names to target positions in ticks",
                        "additionalProperties": {"type": "number"},
                    },
                },
                "required": ["positions"],
            },
        ),
        Tool(
            name="gripper_open",
            description="Open the gripper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Target width in ticks (default: 2300 for open)",
                        "default": 2300,
                    },
                },
            },
        ),
        Tool(
            name="gripper_close",
            description="Close the gripper.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Target width in ticks (default: 1700 for closed)",
                        "default": 1700,
                    },
                },
            },
        ),
        Tool(
            name="home",
            description="Move all joints to center position (2048 ticks).",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="wave",
            description="Perform a friendly wave gesture.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="enable_torque",
            description="Enable torque on motors (required before moving).",
            inputSchema={
                "type": "object",
                "properties": {
                    "joints": {
                        "type": "array",
                        "items": {"type": "string", "enum": MOTOR_NAMES},
                        "description": "List of joints to enable (default: all)",
                    },
                },
            },
        ),
        Tool(
            name="disable_torque",
            description="Disable torque on motors (arm will go limp).",
            inputSchema={
                "type": "object",
                "properties": {
                    "joints": {
                        "type": "array",
                        "items": {"type": "string", "enum": MOTOR_NAMES},
                        "description": "List of joints to disable (default: all)",
                    },
                },
            },
        ),
        Tool(
            name="stop",
            description="Emergency stop - disable all torque immediately.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="discover_ports",
            description="Scan serial ports to find connected SO-ARM100 arms. Returns port paths, motor counts, and whether each port has a valid SO-ARM100.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="diagnose",
            description="Run diagnostics on all motors. Checks for disconnected motors, voltage issues (missing power supply), overheating, communication reliability, and stall conditions.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="calibrate_joint",
            description="Calibrate a joint's range of motion by slowly sweeping to find physical limits. Position other joints first so they don't block the test joint. Saves results to calibration file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "joint": {
                        "type": "string",
                        "description": "Joint to calibrate",
                        "enum": MOTOR_NAMES,
                    },
                },
                "required": ["joint"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    controller = get_controller()

    try:
        if name == "connect":
            port = arguments.get("port", "/dev/ttyACM0")
            result = controller.connect(port=port)
            return json_response(result)

        elif name == "get_status":
            if not controller.connected:
                return json_response({"error": "Not connected. Call connect first."})

            state = controller.get_state()
            return json_response({
                "connected": True,
                "port": controller.port_name,
                "joints": {
                    name: {
                        "position_ticks": state.positions[name],
                        "position_degrees": round(state.positions_deg[name], 1),
                        "voltage": state.voltages[name],
                        "temperature_c": state.temperatures[name],
                        "torque_enabled": state.torque_enabled[name],
                        "moving": state.moving[name],
                    }
                    for name in MOTOR_NAMES
                },
            })

        elif name == "move_joint":
            joint = arguments["joint"]
            if "degrees" in arguments:
                from .controller import degrees_to_ticks
                position = degrees_to_ticks(arguments["degrees"])
            elif "position" in arguments:
                position = int(arguments["position"])
            else:
                return json_response({"error": "Must specify either 'position' or 'degrees'"})

            result = controller.move_joint(joint, position)
            return json_response(result)

        elif name == "move_joints":
            positions = {k: int(v) for k, v in arguments["positions"].items()}
            result = controller.move_joints(positions)
            return json_response(result)

        elif name == "gripper_open":
            width = arguments.get("width", 2300)
            result = controller.gripper_open(width)
            return json_response(result)

        elif name == "gripper_close":
            width = arguments.get("width", 1700)
            result = controller.gripper_close(width)
            return json_response(result)

        elif name == "home":
            result = controller.home()
            return json_response(result)

        elif name == "wave":
            result = controller.wave()
            return json_response(result)

        elif name == "enable_torque":
            joints = arguments.get("joints")
            controller.enable_torque(joints)
            return json_response({"torque_enabled": joints or "all"})

        elif name == "disable_torque":
            joints = arguments.get("joints")
            controller.disable_torque(joints)
            return json_response({"torque_disabled": joints or "all"})

        elif name == "stop":
            result = controller.stop()
            return json_response(result)

        elif name == "discover_ports":
            result = discover_ports()
            return json_response(result)

        elif name == "diagnose":
            if not controller.connected:
                return json_response({"error": "Not connected. Call connect first."})
            result = controller.diagnose()
            return json_response(result)

        elif name == "calibrate_joint":
            if not controller.connected:
                return json_response({"error": "Not connected. Call connect first."})
            joint = arguments["joint"]
            result = controller.calibrate_joint(joint)
            return json_response(result)

        else:
            return json_response({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main_entry():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()
