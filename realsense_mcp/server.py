"""
MCP Server for Intel RealSense D405/D435 depth cameras.

Provides RGB + depth capture, 3D point queries, pointcloud generation,
and robot-frame coordinate transforms via calibration.
"""

import asyncio
import base64
import json
import logging
from typing import Any

import cv2
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .controller import get_realsense_controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("realsense-mcp")


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_devices",
            description="List all connected Intel RealSense cameras. Use this to find device serial numbers before connecting.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="connect",
            description=(
                "Connect to an Intel RealSense camera (D405 or D435). "
                "Must be called before capture commands. Auto-detects device type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "serial": {
                        "type": "string",
                        "description": "Device serial number (optional, connects to first device if omitted)",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to connect to: D435, D405 (optional, useful with multiple cameras)",
                        "enum": ["D435", "D405", "D435i"],
                    },
                    "width": {
                        "type": "integer",
                        "description": "Capture width (default: 640)",
                        "default": 640,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Capture height (default: 480)",
                        "default": 480,
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Frame rate (default: 30)",
                        "default": 30,
                    },
                },
            },
        ),
        Tool(
            name="disconnect",
            description="Disconnect from the RealSense camera and release resources.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="capture",
            description=(
                "Capture aligned RGB + depth frames. Returns frame statistics "
                "(dimensions, depth range, coverage). Data is cached for subsequent queries."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="capture_color",
            description="Capture and return the RGB image as JPEG. Calls capture internally.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Resize width (optional, returns full resolution if omitted)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Resize height (optional)",
                    },
                },
            },
        ),
        Tool(
            name="capture_depth_image",
            description=(
                "Capture and return a colorized depth image as JPEG. "
                "Useful for visualizing depth coverage and range."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_mm": {
                        "type": "integer",
                        "description": "Min depth in mm for colormap (default: 100)",
                        "default": 100,
                    },
                    "max_mm": {
                        "type": "integer",
                        "description": "Max depth in mm for colormap (default: 1000)",
                        "default": 1000,
                    },
                },
            },
        ),
        Tool(
            name="get_depth_at",
            description=(
                "Get depth and 3D position (camera frame, meters) at a pixel coordinate. "
                "Call capture first. Use radius > 0 for median filtering."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": "X pixel coordinate",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "Y pixel coordinate",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Patch radius for median filtering (0=single pixel, default: 5)",
                        "default": 5,
                    },
                },
                "required": ["pixel_x", "pixel_y"],
            },
        ),
        Tool(
            name="get_robot_coords_at",
            description=(
                "Get 3D position at a pixel in both camera frame (m) and robot frame (m). "
                "Requires calibration. Call capture first. "
                "For wrist cameras (D405): pass ee_position and ee_rpy from get_status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pixel_x": {
                        "type": "integer",
                        "description": "X pixel coordinate",
                    },
                    "pixel_y": {
                        "type": "integer",
                        "description": "Y pixel coordinate",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Patch radius for median filtering (default: 5)",
                        "default": 5,
                    },
                    "ee_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE position [x,y,z] in meters (required for wrist camera)",
                    },
                    "ee_rpy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE orientation [roll,pitch,yaw] in radians (required for wrist camera)",
                    },
                },
                "required": ["pixel_x", "pixel_y"],
            },
        ),
        Tool(
            name="get_pointcloud_stats",
            description=(
                "Get 3D pointcloud statistics from current depth frame. "
                "Returns bounds, point count, and centroid. Does not return raw points."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "downsample": {
                        "type": "integer",
                        "description": "Take every Nth pixel (default: 4)",
                        "default": 4,
                    },
                },
            },
        ),
        Tool(
            name="save_scan",
            description="Save current RGB + depth frames as compressed NPZ file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Output path (default: /tmp/realsense_scan.npz)",
                        "default": "/tmp/realsense_scan.npz",
                    },
                },
            },
        ),
        Tool(
            name="get_camera_info",
            description="Get info about the connected camera (model, serial, resolution, calibration status).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="detect_objects",
            description=(
                "Detect colored objects and return their 3D robot-frame positions. "
                "Captures fresh aligned color+depth, runs HSV color detection, "
                "and resolves each object to calibrated robot (x, y, z) coordinates. "
                "Returns positions ready for pick_at. Requires calibration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["red", "green", "blue", "orange"],
                        },
                        "description": "Colors to detect (default: all). Options: red, green, blue, orange.",
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum blob area in pixels (default: 500)",
                        "default": 500,
                    },
                    "method": {
                        "type": "string",
                        "enum": ["hsv", "clahe", "lab", "auto"],
                        "description": "Detection method. auto (default) tries CLAHE+HSV, then LAB, then raw HSV.",
                        "default": "auto",
                    },
                    "debug": {
                        "type": "boolean",
                        "description": "Save debug images (masks, annotated) to /tmp/detect_debug/",
                        "default": False,
                    },
                    "ee_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE position [x,y,z] in meters (required for wrist camera)",
                    },
                    "ee_rpy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE orientation [roll,pitch,yaw] in radians (required for wrist camera)",
                    },
                },
            },
        ),
        Tool(
            name="ground_object",
            description=(
                "Find an object in the camera image using natural language. "
                "Uses Qwen2.5-VL-3B on Spark for visual grounding. Returns bounding box "
                "in pixel coordinates and 3D robot coordinates via depth. "
                "Examples: 'the red block', 'the block behind the green one'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the object to find",
                    },
                    "server_url": {
                        "type": "string",
                        "description": "URL of the grounding server (default: http://spark:8090)",
                        "default": "http://spark:8090",
                    },
                    "ee_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE position [x,y,z] in meters (required for wrist camera)",
                    },
                    "ee_rpy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "EE orientation [roll,pitch,yaw] in radians (required for wrist camera)",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    try:
        ctrl = get_realsense_controller()

        if name == "list_devices":
            return json_response(ctrl.list_devices())

        elif name == "connect":
            serial = arguments.get("serial")
            model = arguments.get("model")
            width = arguments.get("width", 640)
            height = arguments.get("height", 480)
            fps = arguments.get("fps", 30)
            ctrl.config.width = width
            ctrl.config.height = height
            ctrl.config.fps = fps
            result = ctrl.connect(serial=serial, model=model)
            return json_response(result)

        elif name == "disconnect":
            return json_response(ctrl.disconnect())

        elif name == "capture":
            result = ctrl.capture()
            return json_response(result)

        elif name == "capture_color":
            # Capture fresh frames
            cap_result = ctrl.capture()
            if not cap_result.get("success"):
                return json_response(cap_result)

            img = ctrl.get_color_image()
            if img is None:
                return json_response({"success": False, "error": "No color frame"})

            # Optional resize
            rw = arguments.get("width")
            rh = arguments.get("height")
            if rw and rh:
                img = cv2.resize(img, (rw, rh))

            # Encode as JPEG
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            return [
                TextContent(type="text", text=json.dumps({
                    "success": True,
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "size_bytes": len(buf),
                })),
                ImageContent(type="image", data=b64, mimeType="image/jpeg"),
            ]

        elif name == "capture_depth_image":
            cap_result = ctrl.capture()
            if not cap_result.get("success"):
                return json_response(cap_result)

            depth = ctrl.get_depth_image()
            if depth is None:
                return json_response({"success": False, "error": "No depth frame"})

            # Colorize depth
            min_mm = arguments.get("min_mm", 100)
            max_mm = arguments.get("max_mm", 1000)
            depth_clipped = np.clip(depth, min_mm, max_mm)
            depth_norm = ((depth_clipped - min_mm) / (max_mm - min_mm) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            # Black out invalid pixels
            depth_color[depth == 0] = 0

            _, buf = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            return [
                TextContent(type="text", text=json.dumps({
                    "success": True,
                    "width": depth.shape[1],
                    "height": depth.shape[0],
                    "min_mm": min_mm,
                    "max_mm": max_mm,
                    **cap_result,
                })),
                ImageContent(type="image", data=b64, mimeType="image/jpeg"),
            ]

        elif name == "get_depth_at":
            px = arguments["pixel_x"]
            py = arguments["pixel_y"]
            radius = arguments.get("radius", 5)
            return json_response(ctrl.get_depth_at(px, py, radius))

        elif name == "get_robot_coords_at":
            px = arguments["pixel_x"]
            py = arguments["pixel_y"]
            radius = arguments.get("radius", 5)
            ee_pos = arguments.get("ee_position")
            ee_rpy = arguments.get("ee_rpy")
            if ee_pos and ee_rpy:
                ctrl.set_ee_pose(ee_pos, ee_rpy)
            return json_response(ctrl.get_robot_coords_at(px, py, radius))

        elif name == "get_pointcloud_stats":
            downsample = arguments.get("downsample", 4)
            return json_response(ctrl.get_pointcloud(downsample))

        elif name == "save_scan":
            path = arguments.get("path", "/tmp/realsense_scan.npz")
            return json_response(ctrl.save_scan(path))

        elif name == "get_camera_info":
            if not ctrl.connected:
                return json_response({"connected": False, "error": "Not connected"})
            info = ctrl._camera_info
            return json_response({
                "connected": True,
                "backend": ctrl.backend,
                "device": info.name,
                "model": info.product_line,
                "serial": info.serial,
                "firmware": info.firmware,
                "usb_type": info.usb_type,
                "resolution": f"{ctrl.config.width}x{ctrl.config.height}",
                "fps": ctrl.config.fps,
                "depth_scale_m": ctrl._depth_scale,
                "has_frames": ctrl.has_frames,
                "calibrated": ctrl._calibration_matrix is not None,
            })

        elif name == "detect_objects":
            if not ctrl.connected:
                return json_response({"error": "Not connected. Call 'connect' first."})
            ee_pos = arguments.get("ee_position")
            ee_rpy = arguments.get("ee_rpy")
            if ee_pos and ee_rpy:
                ctrl.set_ee_pose(ee_pos, ee_rpy)
            colors = arguments.get("colors")
            min_area = arguments.get("min_area", 500)
            method = arguments.get("method", "auto")
            debug = arguments.get("debug", False)
            return json_response(ctrl.detect_objects(
                colors=colors, min_area=min_area,
                method=method, debug=debug))

        elif name == "ground_object":
            if not ctrl.connected:
                return json_response({"error": "Not connected. Call 'connect' first."})
            ee_pos = arguments.get("ee_position")
            ee_rpy = arguments.get("ee_rpy")
            if ee_pos and ee_rpy:
                ctrl.set_ee_pose(ee_pos, ee_rpy)
            query = arguments.get("query", "")
            server_url = arguments.get("server_url", "http://spark:8090")
            return json_response(ctrl.ground_object(
                query=query, server_url=server_url))

        else:
            return json_response({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    logger.info("Starting RealSense MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_entry():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()
