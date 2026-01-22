"""
MCP Server for text-to-speech.

Gives Claude a voice using Piper TTS.
"""

import asyncio
import json
import logging
import tempfile
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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    import re
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


async def generate_audio(text: str, voice_path: Path) -> tuple[str, list[str]]:
    """Generate audio file from text. Returns (final_wav_path, temp_files_to_cleanup)."""
    sentences = split_sentences(text)
    temp_files = []

    # Generate audio for each sentence separately
    for i, sentence in enumerate(sentences):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_wav = f.name
            temp_files.append(temp_wav)

        gen_cmd = (
            f'echo {json.dumps(sentence)} | '
            f'{PIPER_PATH} --model {voice_path} --output_file {temp_wav} 2>/dev/null'
        )

        proc = await asyncio.create_subprocess_shell(
            gen_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"TTS generation failed for sentence {i+1}")

    # Create final output file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        final_wav = f.name

    # Generate a quiet primer tone to wake up the USB speaker
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        primer_wav = f.name

    primer_cmd = (
        f'ffmpeg -y -f lavfi -i "sine=frequency=100:duration=0.2" '
        f'-af "volume=0.01" -ar 22050 {primer_wav} 2>/dev/null'
    )
    await (await asyncio.create_subprocess_shell(primer_cmd)).communicate()

    # Concatenate primer + all sentences
    all_inputs = [primer_wav] + temp_files
    inputs = ' '.join(f'-i {f}' for f in all_inputs)
    n = len(all_inputs)
    filter_concat = ''.join(f'[{i}:a]' for i in range(n))
    concat_cmd = (
        f'ffmpeg -y {inputs} -filter_complex "{filter_concat}concat=n={n}:v=0:a=1[out]" '
        f'-map "[out]" {final_wav} 2>/dev/null'
    )

    temp_files.append(primer_wav)

    proc = await asyncio.create_subprocess_shell(
        concat_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    play_file = final_wav if proc.returncode == 0 else temp_files[0]
    return play_file, temp_files + [final_wav]


async def play_and_cleanup(play_file: str, temp_files: list[str]):
    """Play audio file and clean up temp files."""
    play_cmd = f'aplay -D {AUDIO_DEVICE} {play_file} 2>/dev/null'

    proc = await asyncio.create_subprocess_shell(
        play_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    # Clean up temp files
    for f in temp_files:
        try:
            Path(f).unlink()
        except:
            pass


async def speak_text(text: str, voice: str = None, blocking: bool = True) -> dict:
    """Speak text using Piper TTS.

    Args:
        text: Text to speak
        voice: Voice model to use
        blocking: If True, wait for speech to complete. If False, return immediately.
    """
    voice = voice or current_voice
    voice_path = get_voice_path(voice)

    if not voice_path.exists():
        return {
            "success": False,
            "error": f"Voice model not found: {voice_path}",
            "available_voices": list_available_voices(),
        }

    try:
        # Generate the audio (always wait for this)
        play_file, temp_files = await generate_audio(text, voice_path)

        if blocking:
            # Wait for playback to complete
            await play_and_cleanup(play_file, temp_files)
        else:
            # Fire and forget - schedule playback as background task
            asyncio.create_task(play_and_cleanup(play_file, temp_files))

        return {
            "success": True,
            "text": text,
            "voice": voice,
            "characters": len(text),
            "blocking": blocking,
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
                    "blocking": {
                        "type": "boolean",
                        "description": "If true (default), wait for speech to complete. If false, return immediately while audio plays in background.",
                        "default": True,
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

            blocking = arguments.get("blocking", True)
            result = await speak_text(text, blocking=blocking)
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
