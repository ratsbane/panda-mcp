# Panda MCP - Claude-Controlled Robot Arm

MCP (Model Context Protocol) servers for controlling a Franka Emika Panda robot arm, USB camera, and text-to-speech from Claude.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Claude                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Protocol (stdio)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  franka-mcp   │ │  camera-mcp   │ │  voice-mcp    │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        │ panda-py        │ OpenCV          │ Piper TTS
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Franka Panda │ │  USB Camera   │ │  USB Speaker  │
│ (172.16.0.2)  │ │ (/dev/video0) │ │ (plughw:3,0)  │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Hardware Setup

### Raspberry Pi Configuration
- Raspberry Pi 5 with 16GB RAM
- Real-time kernel (6.12.62+rpt-rpi-v8-rt) for reliable robot control
- Connected to Panda arm directly via Ethernet
- USB camera for workspace observation
- USB speaker for voice output

### Network Configuration
The Pi connects directly to the Panda arm's control interface (not the shop floor controller):

```bash
# Configure static IP on eth0
sudo nmcli con add con-name panda type ethernet ifname eth0 \
    ipv4.addresses 172.16.0.1/24 ipv4.method manual
```

- **Panda arm control interface:** `172.16.0.2`
- **Raspberry Pi:** `172.16.0.1`

### Real-Time Kernel Setup
For reliable FCI communication, install the RT kernel:

```bash
sudo apt install linux-image-rpi-v8-rt
sudo reboot
```

Verify with: `uname -r` (should show `-rt` suffix)

## Installation

### Prerequisites

```bash
# System dependencies
sudo apt update
sudo apt install -y build-essential cmake git libpoco-dev libeigen3-dev

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Piper TTS (for voice-mcp)

```bash
pip install piper-tts
mkdir -p voices
# Download a voice model
wget -O voices/en_US-lessac-medium.onnx \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O voices/en_US-lessac-medium.onnx.json \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Running the Servers

The MCP servers are configured in `~/.claude.json` and start automatically when Claude Code launches.

### Manual startup:
```bash
# Franka MCP server
python -m franka_mcp.server

# Camera MCP server
python -m camera_mcp.server

# Voice MCP server
python -m voice_mcp.server
```

## MCP Tool Reference

### franka-mcp Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to the robot (must be called first) |
| `get_status` | Current joint positions, EE pose, gripper state, errors |
| `move_cartesian` | Move end effector to [x, y, z, roll, pitch, yaw] |
| `move_joints` | Move to joint configuration [q1..q7] |
| `move_relative` | Move EE relative to current position |
| `gripper_move` | Move gripper to width (0.0 - 0.08m) |
| `gripper_grasp` | Grasp with specified width and force |
| `stop` | Immediate stop |
| `recover` | Recover from error state |
| `get_safety_limits` | Current workspace and velocity limits |
| `set_safety_limits` | Update safety boundaries |

### camera-mcp Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to the camera |
| `capture_frame` | Capture single frame, returns base64 JPEG |
| `capture_burst` | Capture N frames for motion analysis |
| `describe_scene` | Analyze scene with object detection |
| `get_camera_info` | Resolution, FPS, device info |
| `set_resolution` | Change capture resolution |

### voice-mcp Tools

| Tool | Description |
|------|-------------|
| `speak` | Speak text aloud using Piper TTS |
| `list_voices` | List available voice models |
| `set_voice` | Change the voice model |
| `get_voice_status` | Current voice settings |

## Data Collection for Visuomotor Learning

This project includes infrastructure for collecting training data to teach Claude to estimate gripper position from camera images.

### Concept
Train a model on (image, robot_state) pairs to predict gripper position, enabling vision-based manipulation without explicit depth sensing. Similar to how humans can pick up objects with one eye closed.

### Data Collection Scripts

```bash
# Collect random samples
python scripts/collect_data.py --mode random --samples 500

# Collect on a grid pattern
python scripts/collect_data.py --mode grid --grid-resolution 5

# Manual collection (press 's' to save)
python scripts/collect_data.py --mode manual
```

### ArUco Marker Tracking
For robust training with variable camera positions, use ArUco markers to track camera pose:

```bash
# Generate printable markers (5cm recommended)
python -c "from common.data_collection import generate_aruco_markers; generate_aruco_markers()"
```

Print markers from `aruco_markers/` and place in the workspace with known positions.

### Visualize Collected Data

```bash
python scripts/visualize_dataset.py path/to/dataset/
```

### Train Gripper Localizer

```bash
python scripts/train_localizer.py \
    --data-dir path/to/dataset/ \
    --epochs 50 \
    --batch-size 32 \
    --export-onnx  # Export for Pi deployment
```

### Model Architecture
- **Backbone:** MobileNetV3-Small (efficient for Pi inference)
- **Output:** (x, y, z) gripper position
- **Optional:** Camera pose conditioning for variable viewpoints

See `models/README.md` for detailed architecture documentation.

## Common Utilities

### Scene Interpretation (`common/scene_interpreter.py`)
Natural language scene understanding for manipulation tasks.

### Manipulation Utilities (`common/manipulation.py`)
Helper functions for common manipulation tasks.

### Calibration (`common/calibration.py`)
Camera-robot calibration utilities.

## Safety Features

### Workspace Limits
Default bounding box (adjustable via `set_safety_limits`):
- X: -0.4m to 0.75m (forward from base)
- Y: -0.5m to 0.5m (left/right)
- Z: 0.05m to 0.7m (height)

### Velocity Limits
- Max Cartesian velocity: 0.1 m/s
- Max rotation velocity: 0.5 rad/s

### Error Recovery
If the robot enters an error state (red light), use:
```
mcp__franka-mcp__recover
```

## Claude Code Configuration

Add to `~/.claude.json` under the project path:

```json
{
  "projects": {
    "/home/doug/panda-mcp": {
      "mcpServers": {
        "franka-mcp": {
          "command": "/home/doug/panda-mcp/venv/bin/python",
          "args": ["-m", "franka_mcp.server"],
          "cwd": "/home/doug/panda-mcp"
        },
        "camera-mcp": {
          "command": "/home/doug/panda-mcp/venv/bin/python",
          "args": ["-m", "camera_mcp.server"],
          "cwd": "/home/doug/panda-mcp"
        },
        "voice-mcp": {
          "command": "/home/doug/panda-mcp/venv/bin/python",
          "args": ["-m", "voice_mcp.server"],
          "cwd": "/home/doug/panda-mcp"
        }
      }
    }
  }
}
```

## Development

### Running in simulation mode (no hardware):
```bash
FRANKA_MOCK=1 python -m franka_mcp.server
CAMERA_MOCK=1 python -m camera_mcp.server
```

### Project Structure
```
panda-mcp/
├── franka_mcp/          # Robot arm MCP server
├── camera_mcp/          # Camera MCP server
├── voice_mcp/           # Text-to-speech MCP server
├── common/              # Shared utilities
│   ├── data_collection.py    # Training data collection
│   ├── mcp_data_collector.py # MCP-integrated collector
│   ├── scene_interpreter.py  # Scene understanding
│   ├── manipulation.py       # Manipulation helpers
│   └── calibration.py        # Calibration utilities
├── models/              # Neural network models
│   └── gripper_localizer.py  # Vision-based position estimation
├── scripts/             # Utility scripts
│   ├── collect_data.py       # Data collection runner
│   ├── train_localizer.py    # Model training
│   └── visualize_dataset.py  # Dataset visualization
├── aruco_markers/       # Generated ArUco markers
└── voices/              # Piper TTS voice models
```

## License

MIT

## Acknowledgments

- [panda-py](https://github.com/JeanElsworthy/panda-py)
- [libfranka](https://github.com/frankaemika/libfranka)
- [MCP Protocol](https://modelcontextprotocol.io)
- [Piper TTS](https://github.com/rhasspy/piper)
- [LeRobot](https://github.com/huggingface/lerobot) - Inspiration for visuomotor learning approach
