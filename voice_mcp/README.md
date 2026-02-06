# robot-mcp-voice

MCP server that gives [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (or any MCP-compatible agent) a voice using [Piper TTS](https://github.com/rhasspy/piper).

## Requirements

- **Piper TTS** installed and on `PATH` (or set `PIPER_PATH`)
- **ffmpeg** for audio concatenation
- **aplay** (ALSA) for audio playback
- At least one Piper voice model (`.onnx` + `.onnx.json`)

## Installation

```bash
pip install robot-mcp-voice
```

### Installing Piper

```bash
pip install piper-tts
# Download a voice model:
mkdir -p ~/.local/share/piper-voices
cd ~/.local/share/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPER_PATH` | auto-detect via `which piper` | Path to piper binary |
| `VOICES_DIR` | `~/.local/share/piper-voices/` | Directory containing voice models |
| `PIPER_VOICE` | `en_US-lessac-medium` | Default voice model name |
| `AUDIO_DEVICE` | `default` | ALSA audio device (e.g. `plughw:3,0`) |

## Quick Start

Add to your Claude Code MCP config (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "voice-mcp": {
      "command": "voice-mcp",
      "args": [],
      "env": {
        "AUDIO_DEVICE": "default"
      }
    }
  }
}
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `speak` | Speak text aloud (blocking or non-blocking) |
| `list_voices` | List available voice models |
| `set_voice` | Change the active voice model |
| `get_voice_status` | Current voice settings and status |

## License

MIT
