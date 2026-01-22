"""
MCP Server for text-to-speech.

Gives Claude a voice using Piper TTS.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("voice-mcp")

# Default configuration
PIPER_PATH = "/home/doug/panda-mcp/venv/bin/piper"
VOICES_DIR = Path("/home/doug/panda-mcp/voices")
DEFAULT_VOICE = "en_US-lessac-medium"
AUDIO_DEVICE = "plughw:3,0"

# Current settings
current_voice = DEFAULT_VOICE


def json_response(data: Any) -> list[TextContent]:
    """Format response as JSON text content."""
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def get_voice_path(voice_name: str) -> Path:
    """Get the path to a voice model file."""
    return VOICES_DIR / f"{voice_name}.onnx"


def list_available_voices() -> list[str]:
    """List all available voice models."""
    voices = []
    if VOICES_DIR.exists():
        for f in VOICES_DIR.glob("*.onnx"):
            voices.append(f.stem)
    return sorted(voices)


async def speak_text(text: str, voice: str = None) -> dict:
    """Speak text using Piper TTS."""
    voice = voice or current_voice
    voice_path = get_voice_path(voice)

    if not voice_path.exists():
        return {
            "success": False,
            "error": f"Voice model not found: {voice_path}",
            "available_voices": list_available_voices(),
        }

    # Prepend a brief pause to prevent clipping at the start
    # The ellipsis creates a natural pause before the actual content
    padded_text = "... " + text

    # Build the command pipeline
    cmd = (
        f'echo {json.dumps(padded_text)} | '
        f'{PIPER_PATH} --model {voice_path} --output-raw 2>/dev/null | '
        f'aplay -D {AUDIO_DEVICE} -r 22050 -f S16_LE -t raw - 2>/dev/null'
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            return {
                "success": True,
                "text": text,
                "voice": voice,
                "characters": len(text),
            }
        else:
            return {
                "success": False,
                "error": f"Speech failed with code {proc.returncode}",
                "stderr": stderr.decode() if stderr else None,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available voice tools."""
    return [
        Tool(
            name="speak",
            description="Speak text aloud using text-to-speech. Use this to communicate verbally with the user.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak aloud",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="list_voices",
            description="List available voice models.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="set_voice",
            description="Change the voice model used for speech.",
            inputSchema={
                "type": "object",
                "properties": {
                    "voice": {
                        "type": "string",
                        "description": "Name of the voice model (without .onnx extension)",
                    },
                },
                "required": ["voice"],
            },
        ),
        Tool(
            name="get_voice_status",
            description="Get current voice settings and status.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    global current_voice

    try:
        if name == "speak":
            text = arguments.get("text", "")
            if not text:
                return json_response({"error": "No text provided"})

            result = await speak_text(text)
            return json_response(result)

        elif name == "list_voices":
            voices = list_available_voices()
            return json_response({
                "voices": voices,
                "current_voice": current_voice,
                "voices_directory": str(VOICES_DIR),
            })

        elif name == "set_voice":
            voice = arguments.get("voice", "")
            voice_path = get_voice_path(voice)

            if not voice_path.exists():
                return json_response({
                    "success": False,
                    "error": f"Voice not found: {voice}",
                    "available_voices": list_available_voices(),
                })

            current_voice = voice
            return json_response({
                "success": True,
                "voice": current_voice,
            })

        elif name == "get_voice_status":
            return json_response({
                "current_voice": current_voice,
                "voice_path": str(get_voice_path(current_voice)),
                "available_voices": list_available_voices(),
                "piper_path": PIPER_PATH,
                "audio_device": AUDIO_DEVICE,
            })

        else:
            return json_response({"error": f"Unknown tool: {name}"})

    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return json_response({"error": str(e)})


async def main():
    """Run the MCP server."""
    logger.info("Starting Voice MCP server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
