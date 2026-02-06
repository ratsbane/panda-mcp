# robot-mcp-camera

MCP server for USB camera capture and scene analysis via [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or any MCP-compatible agent.

## Hardware Requirements

- USB camera (or any V4L2-compatible device)
- Optional: GPU for MobileSAM instance segmentation

## Installation

```bash
pip install robot-mcp-camera
```

## Quick Start

Add to your Claude Code MCP config (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "camera-mcp": {
      "command": "camera-mcp",
      "args": []
    }
  }
}
```

Or run directly:

```bash
camera-mcp            # via entry point
python -m camera_mcp  # via module
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to a camera device |
| `capture_frame` | Capture a single frame (returns base64 JPEG/PNG) |
| `capture_burst` | Capture multiple frames for motion analysis |
| `get_camera_info` | Resolution, FPS, and device info |
| `set_resolution` | Change capture resolution |
| `describe_scene` | Analyze scene: detect objects, colors, spatial relationships |

## Scene Analysis

`describe_scene` supports three detection methods:
- **Color detection** - finds objects by HSV color ranges
- **Contour detection** - finds objects by edges and shapes
- **MobileSAM segmentation** - instance segmentation (slower, more accurate)

## License

MIT
