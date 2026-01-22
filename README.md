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
        │ panda-py        │ ZeroMQ          │ Piper TTS
        │                 ▼                 │
        │         ┌───────────────┐         │
        │         │ camera-daemon │         │
        │         │  (systemd)    │         │
        │         └───────┬───────┘         │
        │                 │ OpenCV          │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Franka Panda │ │  USB Camera   │ │  USB Speaker  │
│(192.168.0.253)│ │ (/dev/video0) │ │ (plughw:3,0)  │
└───────────────┘ └───────────────┘ └───────────────┘
```

The camera daemon enables multiple clients (camera-mcp, scene_viewer) to share the camera simultaneously via ZeroMQ pub/sub.

## Hardware Setup

### Raspberry Pi Configuration
- Raspberry Pi 5 with 16GB RAM
- Real-time kernel (6.12.62+rpt-rpi-v8-rt) for reliable robot control
- Connected to Panda arm directly via Ethernet
- USB camera for workspace observation
- USB speaker for voice output

### Network Configuration
The Pi connects to the Panda's shop floor controller via Ethernet. The controller expects a DHCP server, so configure the Pi to share its connection using NetworkManager's "shared" method, which automatically runs a DHCP server (dnsmasq):

```bash
# Create a shared connection on eth0 (runs DHCP server automatically)
sudo nmcli con add con-name franka-direct type ethernet ifname eth0 \
    ipv4.addresses 192.168.0.2/24 ipv4.method shared

# Activate the connection
sudo nmcli con up franka-direct
```

This configures:
- **Raspberry Pi:** `192.168.0.2` (static)
- **DHCP range:** `192.168.0.11 - 192.168.0.254` (served by dnsmasq)
- **Panda shop floor controller:** Receives address via DHCP (typically `192.168.0.253`)

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

## Camera Daemon

The camera daemon captures frames from the USB camera and publishes them over a ZeroMQ pub/sub bus. This architecture solves a key problem: only one process can open a V4L2 camera device at a time, but multiple clients need camera access (the MCP server for Claude, the scene viewer for monitoring, data collection scripts, etc.).

### ZeroMQ Architecture
```
┌─────────────────┐
│  USB Camera     │
│  /dev/video0    │
└────────┬────────┘
         │ OpenCV VideoCapture
         ▼
┌─────────────────┐
│  camera-daemon  │  Captures frames, publishes to ZMQ
└────────┬────────┘
         │ ZeroMQ PUB socket
         │ ipc:///tmp/camera-daemon.sock
         ▼
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
┌───────┐ ┌───────┐ ┌──────────────┐
│camera │ │scene  │ │ data         │
│ -mcp  │ │viewer │ │ collection   │
└───────┘ └───────┘ └──────────────┘
   ZMQ SUB clients (can subscribe/unsubscribe freely)
```

### Setup as systemd user service:
```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp systemd/camera-daemon.service ~/.config/systemd/user/

# Enable and start
systemctl --user daemon-reload
systemctl --user enable camera-daemon
systemctl --user start camera-daemon

# Check status
systemctl --user status camera-daemon
journalctl --user -u camera-daemon -f

# Enable linger so service runs even when not logged in
sudo loginctl enable-linger $USER
```

The service runs as a user service (not system-wide) so it has access to the user's Python virtual environment. With linger enabled, the daemon starts at boot and continues running after logout.

### Manual startup:
```bash
python -m camera_daemon.server
```

### Configuration
The daemon publishes frames on:
- **IPC:** `ipc:///tmp/camera-daemon.sock` (preferred, faster)
- **TCP:** `tcp://127.0.0.1:5555` (fallback)

Environment variables:
- `CAMERA_DEVICE`: Camera device number (default: 0)
- `CAMERA_WIDTH`: Frame width (default: 640)
- `CAMERA_HEIGHT`: Frame height (default: 480)
- `CAMERA_FPS`: Target FPS (default: 30)

## Scene Viewer

Interactive viewer with live object detection overlays. Useful for monitoring the workspace on a connected display.

```bash
# Run with daemon (default)
python -m common.scene_viewer

# Run with direct camera access
python -m common.scene_viewer --direct

# With custom settings
python -m common.scene_viewer --width 1280 --height 720 --min-area 500
```

### Keyboard controls:
- `q` - Quit
- `s` - Save current frame and scene description
- `c` - Toggle color detection
- `e` - Toggle edge/contour detection
- `r` - Toggle relationship lines
- `Space` - Pause/resume

## MCP Tool Reference

### franka-mcp Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to the robot (must be called first) |
| `get_status` | Current joint positions, EE pose, gripper state, errors |
| `move_cartesian` | Move end effector to [x, y, z, roll, pitch, yaw] |
| `move_joints` | Move to joint configuration [q1..q7] |
| `move_relative` | Move EE relative to current position |
| `move_cartesian_sequence` | Execute multiple waypoints as smooth continuous motion |
| `move_joint_sequence` | Execute multiple joint configurations as smooth motion |
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
| `speak` | Speak text aloud using Piper TTS (supports blocking/non-blocking) |
| `list_voices` | List available voice models |
| `set_voice` | Change the voice model |
| `get_voice_status` | Current voice settings |

The `speak` tool supports a `blocking` parameter (default: true). Set to false to continue execution while audio plays in the background.

#### Voice Server Features
- **Piper TTS**: Uses the fast, local Piper text-to-speech engine
- **Sentence splitting**: Long text is split into sentences for natural pauses
- **Audio primer**: A brief low-frequency tone primes the USB speaker to avoid clipping the first syllable
- **Background playback**: Non-blocking mode allows Claude to continue working while speaking
- **Multiple voices**: Download additional Piper voice models to `voices/` directory

Audio output is configured for USB speaker at `plughw:3,0`. Adjust `AUDIO_DEVICE` in `voice_mcp/server.py` if your speaker is on a different device.

## Data Collection for Visuomotor Learning

This project includes infrastructure for collecting training data to teach Claude to estimate gripper position from camera images.

### Concept
Train a model on (image, robot_state) pairs to predict gripper position, enabling vision-based manipulation without explicit depth sensing. Similar to how humans can pick up objects with one eye closed.

### Data Collection Scripts

The `scripts/collect_data.py` script captures synchronized (image, robot_state) pairs for training:

```bash
# Collect random samples - robot moves to random positions in workspace
python scripts/collect_data.py --mode random --samples 500

# Collect on a grid pattern - systematic coverage of workspace
python scripts/collect_data.py --mode grid --grid-resolution 5

# Manual collection - you control the robot, press 's' to save samples
python scripts/collect_data.py --mode manual

# Specify output directory
python scripts/collect_data.py --mode random --samples 100 --output-dir ./my_dataset
```

Each sample saves:
- JPEG image from the workspace camera
- JSON metadata with joint positions, end-effector pose, gripper width, and timestamp

#### Collection Modes
- **random**: Robot moves to random positions within safe workspace bounds. Good for diverse coverage.
- **grid**: Systematic grid pattern across the workspace. Good for uniform spatial coverage.
- **manual**: Human teleoperates or guides the robot. Press 's' to save current state. Good for task-specific demonstrations.

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
├── camera_daemon/       # ZeroMQ camera frame publisher
├── voice_mcp/           # Text-to-speech MCP server
├── common/              # Shared utilities
│   ├── data_collection.py    # Training data collection
│   ├── mcp_data_collector.py # MCP-integrated collector
│   ├── scene_interpreter.py  # Scene understanding
│   ├── scene_viewer.py       # Interactive scene viewer
│   ├── manipulation.py       # Manipulation helpers
│   └── calibration.py        # Calibration utilities
├── models/              # Neural network models
│   └── gripper_localizer.py  # Vision-based position estimation
├── scripts/             # Utility scripts
│   ├── collect_data.py       # Data collection runner
│   ├── train_localizer.py    # Model training
│   └── visualize_dataset.py  # Dataset visualization
├── systemd/             # Systemd service files
│   └── camera-daemon.service # Camera daemon user service
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
