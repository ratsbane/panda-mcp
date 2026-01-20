# Panda MCP - Claude-Controlled Robot Arm

MCP (Model Context Protocol) servers for controlling a Franka Emika Panda robot arm and USB camera from Claude.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Claude                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Protocol (stdio/SSE)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│   franka-mcp      │               │   camera-mcp      │
│   (Port 8401)     │               │   (Port 8402)     │
└────────┬──────────┘               └────────┬──────────┘
         │                                   │
         │ frankx/libfranka                  │ OpenCV
         │                                   │
         ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│   Franka Panda    │               │   USB Camera      │
│   (192.168.0.1)   │               │   (/dev/video0)   │
└───────────────────┘               └───────────────────┘
```

## Hardware Setup

### Raspberry Pi Configuration
- Raspberry Pi 5 with 16GB RAM
- Connected to Panda arm directly via Ethernet (static IP: 192.168.0.2)
- USB camera connected for workspace observation

### Network Configuration
```bash
# Configure via nmcli (Raspberry Pi OS)
sudo nmcli con add con-name panda type ethernet ifname eth0 \
    ipv4.addresses 192.168.0.2/24 ipv4.method manual
```

The Panda arm's IP is `192.168.0.1` (direct arm connection, not via controller).

## Installation

### Prerequisites

```bash
# System dependencies
sudo apt update
sudo apt install -y build-essential cmake git libpoco-dev libeigen3-dev

# Real-time kernel (recommended but not required for basic operation)
# See: https://frankaemika.github.io/docs/installation_linux.html

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### libfranka Installation

```bash
git clone --recursive https://github.com/frankaemika/libfranka.git
cd libfranka
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

### frankx Installation (High-level Python wrapper)

```bash
pip install frankx
```

## Running the Servers

### Start both servers:
```bash
./run.sh
```

### Or individually:
```bash
# Franka MCP server
python -m franka_mcp.server

# Camera MCP server  
python -m camera_mcp.server
```

## MCP Tool Reference

### franka-mcp Tools

| Tool | Description |
|------|-------------|
| `get_status` | Current joint positions, EE pose, gripper state, errors |
| `move_cartesian` | Move end effector to [x, y, z, roll, pitch, yaw] |
| `move_joints` | Move to joint configuration [q1..q7] |
| `move_relative` | Move EE relative to current position |
| `gripper_move` | Move gripper to width (0.0 - 0.08m) |
| `gripper_grasp` | Grasp with specified width and force |
| `stop` | Immediate stop (non-emergency) |
| `recover` | Recover from error state |
| `get_safety_limits` | Current workspace and velocity limits |
| `set_safety_limits` | Update safety boundaries |

### camera-mcp Tools

| Tool | Description |
|------|-------------|
| `capture_frame` | Capture single frame, returns base64 JPEG |
| `capture_burst` | Capture N frames for motion analysis |
| `get_camera_info` | Resolution, FPS, device info |
| `set_resolution` | Change capture resolution |

## Safety Features

### Workspace Limits
Default bounding box (adjustable):
- X: 0.2m to 0.8m (forward from base)
- Y: -0.4m to 0.4m (left/right)
- Z: 0.05m to 0.6m (height)

### Velocity Limits
- Max Cartesian velocity: 0.1 m/s (default, adjustable to 0.3)
- Max joint velocity: 0.5 rad/s

### Dry Run Mode
Enable to preview commands without execution:
```python
{"dry_run": true}
```

## Claude Code Configuration

Add to your MCP settings (`~/.config/claude-code/settings.json`):

```json
{
  "mcpServers": {
    "franka": {
      "command": "ssh",
      "args": ["pi@panda-pi.local", "cd /home/pi/panda-mcp && ./venv/bin/python -m franka_mcp.server"],
      "env": {}
    },
    "camera": {
      "command": "ssh", 
      "args": ["pi@panda-pi.local", "cd /home/pi/panda-mcp && ./venv/bin/python -m camera_mcp.server"],
      "env": {}
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

### Running tests:
```bash
pytest tests/
```

## License

MIT

## Acknowledgments

- [libfranka](https://github.com/frankaemika/libfranka)
- [frankx](https://github.com/pantor/frankx)
- [MCP Protocol](https://modelcontextprotocol.io)
