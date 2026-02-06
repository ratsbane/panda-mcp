# robot-mcp-so100

MCP server for controlling the [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robot arm via [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or any MCP-compatible agent.

## Hardware Requirements

- **SO-ARM100** robot arm (6x Feetech STS3215 servos, IDs 1-6)
- **12V power supply** connected to the servo daisy chain (USB alone only powers comms)
- USB serial connection (typically `/dev/ttyACM0` on Linux)

## Installation

```bash
pip install robot-mcp-so100
```

## Quick Start

Add to your Claude Code MCP config (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "so100-mcp": {
      "command": "so100-mcp",
      "args": []
    }
  }
}
```

Or run directly:

```bash
so100-mcp           # via entry point
python -m so100_mcp  # via module
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to the arm on a serial port |
| `get_status` | Joint positions, voltages, temperatures, torque status |
| `move_joint` | Move a single joint (ticks or degrees) |
| `move_joints` | Move multiple joints simultaneously |
| `gripper_open` | Open the gripper |
| `gripper_close` | Close the gripper |
| `home` | Move all joints to center (2048 ticks) |
| `wave` | Perform a friendly wave gesture |
| `enable_torque` | Enable torque on selected motors |
| `disable_torque` | Disable torque (arm goes limp) |
| `stop` | Emergency stop - disable all torque |
| `discover_ports` | Scan serial ports for connected arms |
| `diagnose` | Motor diagnostics (voltage, temp, comms, stall) |
| `calibrate_joint` | Sweep a joint to find its physical limits |

## Calibration

Calibration data is stored in `~/.config/robot-mcp-so100/calibration.json`. Run `calibrate_joint` for each joint to map its physical range of motion. The server auto-positions other joints out of the way before sweeping.

## License

MIT
