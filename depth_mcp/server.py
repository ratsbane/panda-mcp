"""
MCP Server for PhotoNeo 3D depth camera.

Wraps SSH calls to phoxi_grab on tuppy for depth sensing.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .phoxi_client import get_phoxi_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("depth-mcp")


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="connect",
            description="Connect to the PhotoNeo depth camera via tuppy. Must be called before other commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Remote host running PhoXi Control (default: tuppy)",
                        "default": "tuppy",
                    },
                },
            },
        ),
        Tool(
            name="capture_depth",
            description=(
                "Trigger a PhotoNeo 3D scan and retrieve depth, pointcloud, and texture. "
                "Returns scan statistics. Data is cached for get_depth_at queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "save_path": {
                        "type": "string",
                        "description": "Auto-save scan to this path (default: /tmp/phoxi_scan.npz). Set to empty string to skip saving.",
                        "default": "/tmp/phoxi_scan.npz",
                    },
                },
            },
        ),
        Tool(
            name="get_depth_at",
            description=(
                "Get depth and 3D position (in camera frame, mm) at a specific pixel coordinate. "
                "Call capture_depth first. Use radius > 0 for median filtering over a patch."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": "X pixel coordinate (0-1679)",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "Y pixel coordinate (0-1199)",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Patch radius for median filtering (0 = single pixel, default: 5)",
                        "default": 5,
                    },
                },
                "required": ["pixel_x", "pixel_y"],
            },
        ),
        Tool(
            name="get_robot_coords_at",
            description=(
                "Get 3D position at a pixel in both camera frame (mm) and robot frame (m). "
                "Requires depth calibration (run scripts/calibrate_depth.py). "
                "Call capture_depth first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": "X pixel coordinate (0-1679)",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "Y pixel coordinate (0-1199)",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Patch radius for median filtering (0 = single pixel, default: 5)",
                        "default": 5,
                    },
                },
                "required": ["pixel_x", "pixel_y"],
            },
        ),
        Tool(
            name="save_scan",
            description="Save current scan data as compressed NPZ file for external use.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output path (default: /tmp/phoxi_scan.npz)",
                        "default": "/tmp/phoxi_scan.npz",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        client = get_phoxi_client()

        if name == "connect":
            host = arguments.get("host", "tuppy")
            client.host = host
            result = await client.connect()
            return json_response(result)

        elif name == "capture_depth":
            result = await client.capture()
            save_path = arguments.get("save_path", "/tmp/phoxi_scan.npz")
            if save_path and result.get("success"):
                save_result = client.save_npz(save_path)
                result["saved"] = save_path
                result["save_size_bytes"] = save_result.get("size_bytes", 0)
            return json_response(result)

        elif name == "get_depth_at":
            px = arguments["pixel_x"]
            py = arguments["pixel_y"]
            radius = arguments.get("radius", 5)
            if radius > 0:
                result = client.get_depth_patch(px, py, radius)
            else:
                result = client.get_depth_at(px, py)
            return json_response(result)

        elif name == "get_robot_coords_at":
            px = arguments["pixel_x"]
            py = arguments["pixel_y"]
            radius = arguments.get("radius", 5)
            if radius > 0:
                patch = client.get_depth_patch(px, py, radius)
            else:
                patch = client.get_depth_at(px, py)
            if not patch.get("valid"):
                return json_response(patch)
            # Add robot frame coordinates
            pos = patch.get("position_mm", {})
            robot_result = client.camera_to_robot(pos["x"], pos["y"], pos["z"])
            patch.update(robot_result)
            return json_response(patch)

        elif name == "save_scan":
            path = arguments.get("path", "/tmp/phoxi_scan.npz")
            result = client.save_npz(path)
            return json_response(result)

        else:
            return json_response({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    logger.info("Starting Depth MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_entry():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()
