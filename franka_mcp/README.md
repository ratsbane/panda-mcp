# robot-mcp-franka

MCP server for controlling a [Franka Emika Panda](https://franka.de/) robot arm via [Claude Code](https://docs.anthropic.com/en/docs/claude-code) or any MCP-compatible agent.

## Hardware Requirements

- **Franka Emika Panda** (or Franka Research 3) with FCI enabled
- Network connection to the robot (default: `192.168.0.1`)
- FCI must be activated via Franka Desk web interface

## Installation

```bash
pip install robot-mcp-franka
```

> **Note:** Requires `libfranka` system library. See [panda-python docs](https://github.com/JeanElsworthy/panda-python) for installation.

## Quick Start

Add to your Claude Code MCP config (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "franka-mcp": {
      "command": "franka-mcp",
      "args": []
    }
  }
}
```

Or run directly:

```bash
franka-mcp            # via entry point
python -m franka_mcp  # via module
```

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `connect` | Connect to the Panda at a given IP |
| `get_status` | Joint positions, end-effector pose, gripper width, errors |
| `move_cartesian` | Move end-effector to an absolute Cartesian pose |
| `move_relative` | Move end-effector by a relative delta |
| `move_joints` | Move to a 7-DOF joint configuration |
| `move_cartesian_sequence` | Smooth multi-waypoint Cartesian trajectory |
| `move_joint_sequence` | Smooth multi-waypoint joint trajectory |
| `gripper_move` | Open/close gripper to a width |
| `gripper_grasp` | Grasp with specified width and force |
| `stop` | Emergency stop - halt all motion |
| `recover` | Recover from error state |
| `get_safety_limits` | View workspace bounds and velocity limits |
| `set_safety_limits` | Update safety constraints |

## Safety

All commands pass through a safety validator that enforces workspace bounds and velocity limits. Large moves (>20cm) require explicit confirmation.

## License

MIT
