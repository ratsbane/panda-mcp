"""
MCP Server for camera/vision.

Provides workspace observation for Claude.
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .controller import get_camera_controller

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
            description="Connect to the USB camera. Call this before capturing frames.",
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Handle tool calls."""
    
    controller = get_camera_controller()
    
    try:
        if name == "connect":
            device = arguments.get("device", 0)
            controller.config.device = device
            result = controller.connect()
            return json_response(result)
        
        elif name == "capture_frame":
            if not controller.connected:
                return json_response({"error": "Camera not connected. Call 'connect' first."})
            
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
            if not controller.connected:
                return json_response({"error": "Camera not connected. Call 'connect' first."})
            
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
            if not controller.connected:
                return json_response({"error": "Camera not connected"})
            
            info = controller.get_info()
            return json_response(info.to_dict())
        
        elif name == "set_resolution":
            result = controller.set_resolution(
                width=arguments["width"],
                height=arguments["height"],
            )
            return json_response(result)
        
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


if __name__ == "__main__":
    asyncio.run(main())
