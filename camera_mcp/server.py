"""
MCP Server for camera/vision.

Provides workspace observation for Claude.
"""

import asyncio
import json
import logging
from typing import Any

import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .controller import get_camera_controller
from common.scene_interpreter import interpret_scene, annotate_scene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("camera-mcp")


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available camera tools."""
    return [
        Tool(
            name="connect",
            description="Connect to the USB camera. Other tools auto-connect, so this is only needed to force a reconnect or change device.",
            inputSchema={
                "type": "object",
                "properties": {
                    "device": {
                        "type": "integer",
                        "description": "Camera device number (default: 0 for /dev/video0)",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="capture_frame",
            description="Capture a single frame from the camera. Returns base64-encoded JPEG image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {
                        "type": "integer",
                        "description": "Target width for resize (optional)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Target height for resize (optional)",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["jpeg", "png"],
                        "default": "jpeg",
                        "description": "Output image format",
                    },
                },
            },
        ),
        Tool(
            name="capture_burst",
            description="Capture multiple frames in sequence for motion analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of frames (1-30)",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 5,
                    },
                    "interval_ms": {
                        "type": "integer",
                        "description": "Milliseconds between frames",
                        "default": 100,
                    },
                    "width": {
                        "type": "integer",
                        "description": "Target width for resize (optional)",
                    },
                    "height": {
                        "type": "integer",
                        "description": "Target height for resize (optional)",
                    },
                },
            },
        ),
        Tool(
            name="get_camera_info",
            description="Get information about the connected camera.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="set_resolution",
            description="Change the camera capture resolution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "width": {"type": "integer", "description": "Target width"},
                    "height": {"type": "integer", "description": "Target height"},
                },
                "required": ["width", "height"],
            },
        ),
        Tool(
            name="describe_scene",
            description="Capture an image and analyze the scene. Returns detected objects, their positions, colors, spatial relationships, and a natural language summary.",
            inputSchema={
                "type": "object",
                "properties": {
                    "use_color_detection": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use color-based object detection",
                    },
                    "use_contour_detection": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use edge/contour-based detection",
                    },
                    "use_segmentation": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use MobileSAM for instance segmentation (slower but more accurate)",
                    },
                    "use_yolo_detection": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use Hailo-accelerated YOLOv8 for real object class labels (e.g. person, cup, bottle)",
                    },
                    "min_area": {
                        "type": "integer",
                        "default": 500,
                        "description": "Minimum object area in pixels",
                    },
                    "include_annotated_image": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include an annotated image showing detected objects",
                    },
                },
            },
        ),
        Tool(
            name="describe_scene_3d",
            description=(
                "Fuse Hailo YOLOv8 detection (class labels) with PhotoNeo depth pointcloud "
                "(3D positions) to produce a labeled, 3D-positioned scene graph. "
                "Call depth-mcp capture_depth + save_scan first to provide fresh depth data. "
                "Returns objects with COCO labels, robot-frame 3D positions, dimensions, "
                "grasp widths, and colors."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "depth_npz_path": {
                        "type": "string",
                        "default": "/tmp/phoxi_scan.npz",
                        "description": "Path to saved depth scan NPZ from depth-mcp save_scan",
                    },
                    "max_age_seconds": {
                        "type": "number",
                        "default": 60,
                        "description": "Reject depth scans older than this many seconds",
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "default": 0.3,
                        "description": "YOLO confidence threshold (0-1)",
                    },
                },
            },
        ),
        Tool(
            name="ground_object",
            description=(
                "Find an object in the camera image using natural language. "
                "Uses Qwen2.5-VL-3B on Spark for visual grounding. "
                "Returns bounding box in pixel coordinates and approximate robot coordinates "
                "(via USB camera homography). For precise 3D positioning, use the depth camera. "
                "Examples: 'the red block', 'the block behind the green one', 'the tiger toy'."
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
                        "default": "http://spark:8090",
                        "description": "URL of the grounding server",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    
    controller = get_camera_controller()
    
    try:
        # Auto-connect for any tool that needs the camera
        if name != "connect" and not controller.connected:
            conn_result = controller.connect()
            if not conn_result.get("connected"):
                return json_response({"error": f"Auto-connect failed: {conn_result.get('error', 'unknown')}"})
            logger.info("Camera auto-connected")

        if name == "connect":
            device = arguments.get("device", 0)
            controller.config.device = device
            # Disconnect first to force a fresh connection
            controller.disconnect()
            result = controller.connect()
            return json_response(result)

        elif name == "capture_frame":
            result = controller.capture_frame(
                width=arguments.get("width"),
                height=arguments.get("height"),
                format=arguments.get("format", "jpeg"),
            )
            
            if result["success"]:
                # Return as image content for Claude to see
                return [
                    ImageContent(
                        type="image",
                        data=result["data"],
                        mimeType=result.get("mime_type", "image/jpeg"),
                    ),
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "width": result["width"],
                            "height": result["height"],
                            "size_bytes": result["size_bytes"],
                            "mock": result.get("mock", False),
                        }, indent=2),
                    ),
                ]
            else:
                return json_response(result)
        
        elif name == "capture_burst":
            result = controller.capture_burst(
                count=arguments.get("count", 5),
                interval_ms=arguments.get("interval_ms", 100),
                width=arguments.get("width"),
                height=arguments.get("height"),
            )
            
            if result["success"]:
                # Return multiple images plus metadata
                responses: list[TextContent | ImageContent] = []
                
                for frame in result["frames"]:
                    responses.append(ImageContent(
                        type="image",
                        data=frame["data"],
                        mimeType="image/jpeg",
                    ))
                
                responses.append(TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "frame_count": result["frame_count"],
                    }, indent=2),
                ))
                
                return responses
            else:
                return json_response(result)
        
        elif name == "get_camera_info":
            info = controller.get_info()
            return json_response(info.to_dict())
        
        elif name == "set_resolution":
            result = controller.set_resolution(
                width=arguments["width"],
                height=arguments["height"],
            )
            return json_response(result)

        elif name == "describe_scene":
            # Capture raw frame
            frame = controller.capture_raw()
            if frame is None:
                return json_response({"error": "Failed to capture frame"})

            # Interpret scene
            scene = interpret_scene(
                frame,
                use_color_detection=arguments.get("use_color_detection", True),
                use_contour_detection=arguments.get("use_contour_detection", True),
                use_segmentation=arguments.get("use_segmentation", False),
                use_yolo_detection=arguments.get("use_yolo_detection", False),
                min_area=arguments.get("min_area", 500),
            )

            responses: list[TextContent | ImageContent] = []

            # Include annotated image if requested
            if arguments.get("include_annotated_image", False):
                import cv2
                from PIL import Image
                import io
                import base64

                annotated = annotate_scene(frame, scene, show_labels=True, show_relationships=True)
                # Convert BGR to RGB
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated_rgb)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                image_data = buffer.getvalue()

                responses.append(ImageContent(
                    type="image",
                    data=base64.b64encode(image_data).decode("utf-8"),
                    mimeType="image/jpeg",
                ))

            # Add scene description as JSON
            responses.append(TextContent(
                type="text",
                text=json.dumps(scene.to_dict(), indent=2),
            ))

            return responses

        elif name == "ground_object":
            query = arguments.get("query", "")
            if not query:
                return json_response({"error": "query is required"})

            server_url = arguments.get("server_url", "http://spark:8090")

            # Get JPEG frame from camera
            if controller._using_daemon and controller._daemon_client:
                jpeg_data = controller._daemon_client.get_frame_jpeg()
                if jpeg_data is None:
                    return json_response({"error": "Failed to capture frame"})
                info = controller._daemon_client.get_info()
                img_w, img_h = info.width, info.height
            else:
                frame = controller.capture_raw()
                if frame is None:
                    return json_response({"error": "Failed to capture frame"})
                import cv2
                _, jpeg_buf = cv2.imencode(".jpg", frame)
                jpeg_data = jpeg_buf.tobytes()
                img_h, img_w = frame.shape[:2]

            # Call grounding server
            from common.grounding_client import GroundingClient
            client = GroundingClient(server_url)
            result = client.ground(jpeg_data, query, width=img_w, height=img_h)

            if not result.get("success"):
                return json_response({
                    "error": result.get("error", "Grounding failed"),
                    "raw_output": result.get("raw_output", ""),
                })

            # Convert pixel center to approximate robot coordinates
            pixel_cx, pixel_cy = result["center"]
            robot_coords = None
            try:
                import cv2
                cal_path = "/home/doug/panda-mcp/calibration/aruco_calibration.npz"
                cal_data = np.load(cal_path)
                H = cal_data["H"]
                table_z = float(cal_data["table_z"])
                point = np.array([[[float(pixel_cx), float(pixel_cy)]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, H)
                rx, ry = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
                robot_coords = {"x": round(rx, 4), "y": round(ry, 4), "z": round(table_z, 4)}
            except Exception as e:
                logger.warning(f"Could not convert to robot coords: {e}")

            return json_response({
                "success": True,
                "query": query,
                "bbox": result["bbox"],
                "center_pixel": result["center"],
                "robot_coords_approx": robot_coords,
                "inference_ms": result.get("inference_ms", 0),
                "image_size": [img_w, img_h],
                "note": "robot_coords_approx uses USB camera homography (~2-4cm error). Use depth camera for precise positioning.",
            })

        elif name == "describe_scene_3d":
            # Capture raw frame from USB camera
            frame = controller.capture_raw()
            if frame is None:
                return json_response({"error": "Failed to capture frame"})

            from common.depth_fusion import build_scene_graph

            depth_npz_path = arguments.get("depth_npz_path", "/tmp/phoxi_scan.npz")
            max_age = arguments.get("max_age_seconds", 60)
            confidence = arguments.get("confidence_threshold", 0.3)

            scene = build_scene_graph(
                frame,
                depth_npz_path=depth_npz_path,
                max_age_seconds=max_age,
                confidence_threshold=confidence,
            )

            return json_response(scene.to_dict())

        else:
            return json_response({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    logger.info("Starting Camera MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main_entry():
    """Sync entry point for console_scripts."""
    asyncio.run(main())


if __name__ == "__main__":
    main_entry()
