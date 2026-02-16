# Panda MCP: An LLM-Controlled Robot Arm

**Claude Code autonomously picks up objects using a 7-DOF Franka Panda arm, 3D depth camera, AI-accelerated computer vision, and analytical inverse kinematics -- all orchestrated through the Model Context Protocol (MCP).**

<div align="center">
  <a href="https://www.youtube.com/watch?v=Zig-_-1gK1Y">
    <img src="https://img.youtube.com/vi/Zig-_-1gK1Y/maxresdefault.jpg" alt="Demo Video" width="700">
  </a>
  <p><strong>&#9654; Watch the demo video</strong></p>
</div>

## What This Is

This project gives an LLM physical embodiment. [Claude Code](https://github.com/anthropics/claude-code) connects to five MCP servers that provide tools for robot control, computer vision, 3D depth sensing, and speech. Claude can see the workspace, reason about what to do, move a robot arm, grasp objects, and speak aloud -- all through natural language interaction.

```
User: "Pick up the green block and put it on the right side"

Claude: *captures 3D depth scan* → *runs YOLOv8 object detection* →
        *fuses labels with depth pointcloud* → *identifies green block at (0.45, -0.04, 0.04)* →
        *moves arm above target* → *lowers with analytical IK* → *grasps with 70N force* →
        *lifts and places* → "Done! Green block moved to the right."
```

## Key Capabilities

### Autonomous Pick-and-Place
Claude picks up blocks without human intervention using a perception-to-action pipeline:
1. **3D depth scan** via PhotoNeo MotionCam-3D (structured light, 1680x1200)
2. **Object detection** via YOLOv8 on Hailo-10H AI accelerator (27ms inference, 40 TOPS)
3. **3D scene fusion** -- YOLO class labels matched to depth pointcloud clusters in robot coordinates
4. **Analytical IK** for precise table-height grasping (1-4mm accuracy)
5. **Incremental lowering** in 4cm steps to avoid joint reflexes
6. **Force-controlled grasp** with width matching (critical: grasp width must match object)

### Visual Grounding
The `ground_object` tool uses **Qwen2.5-VL-3B** (running on DGX Spark) to find objects from natural language descriptions. Ask for "the red block behind the green one" or "the tiger toy" and get bounding boxes and approximate robot coordinates. Useful when the depth camera can't see an object or for disambiguating between similar items.

### Skill Sequencing
The `execute_plan` tool accepts a sequence of high-level skills (pick, place, move, grasp, open_gripper, home, wait) and executes them back-to-back with zero inter-step latency. Claude plans the full sequence, the robot executes it all at once. Three blocks picked, placed, and homed in 96 seconds.

### 3D Scene Understanding
The `describe_scene_3d` tool fuses two sensing modalities:
- **USB camera + Hailo YOLOv8**: Object class labels and 2D bounding boxes
- **PhotoNeo depth camera**: Accurate 3D pointcloud transformed to robot frame via Kabsch/SVD calibration

Output: A labeled 3D scene graph with object positions, dimensions, colors, and suggested grasp widths.

### Web-Based Remote Control
A browser-based jog interface provides virtual joysticks, d-pad for wrist orientation, and real-time feedback:
- Live camera stream with position overlay
- Joint position visualization with limit indicators (color-coded: blue/orange/red)
- IK BLOCKED warning when the arm can't reach a requested position
- Works on phones and tablets via WebSocket (port 8766)

### Gamepad Teleoperation
Xbox 360 controller for manual arm positioning via IK-based jogging at up to 50mm/step, 20Hz. Fine mode (LB) drops to 3mm steps. Used for calibration, data collection, and interactive demos.

### Voice Narration
Piper TTS announces actions during demos ("Scanning the workspace...", "Picking up the green block...").

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   Claude Code (Opus 4.6)                       │
│              Reasoning, Planning, Error Recovery               │
└──────┬──────────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │          │
       │ MCP      │ MCP      │ MCP      │ MCP      │ MCP
       │ (stdio)  │ (stdio)  │ (stdio)  │ (stdio)  │ (stdio)
       ▼          ▼          ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│franka-mcp││camera-mcp││depth-mcp ││voice-mcp ││so100-mcp │
│          ││+Hailo    ││          ││          ││          │
│ panda-py ││ YOLOv8   ││ PhotoNeo ││ Piper    ││ Feetech  │
│ IK/FK    ││ Qwen VL  ││ SSH+SCP  ││ TTS      ││ STS3215  │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
     │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼
  Franka     USB Cam +    PhotoNeo     HDMI       SO-ARM100
  Panda      Hailo-10H   MotionCam    Audio      (6-DOF)
               │
               └──→ DGX Spark (Qwen2.5-VL-3B grounding server)
```

**All real-time computation runs on a Raspberry Pi 5** (16GB, RT kernel) -- the AI hat provides 40 TOPS for neural network inference. The DGX Spark provides GPU compute for visual grounding and model training.

## Hardware

| Component | Details |
|-----------|---------|
| **Controller** | Raspberry Pi 5, 16GB RAM, RT kernel |
| **Robot Arm** | Franka Emika Panda (7-DOF, 1kg payload) |
| **AI Accelerator** | Hailo-10H AI Hat+ 2 (40 TOPS, 8GB LPDDR4X, PCIe) |
| **3D Camera** | PhotoNeo MotionCam-3D (structured light, via remote PhoXi Control) |
| **RGB Camera** | USB webcam via ZeroMQ daemon |
| **Audio** | HDMI audio output, Piper TTS |
| **Second Arm** | SO-ARM100 (6-DOF, Feetech servos) -- for future bimanual tasks |
| **Training GPU** | NVIDIA DGX Spark (Grace Blackwell, 128GB unified) -- VLM training + inference |

## Technical Details

### Analytical IK Beats Cartesian Planning
The Franka's built-in Cartesian planner (`move_to_pose`) exhibited 3-7cm cross-axis drift -- moving in X would couple into Y and Z. Replacing all motion with **analytical IK** (`panda_py.ik` with current joints as seed) achieved 1-4mm accuracy, enabling reliable autonomous grasping.

The IK solver tries the current q7 value first, then nearby values, to minimize unnecessary joint movement. Cost is pure joint travel with a small quartic penalty near joint limits. This produces smooth, predictable motions during both autonomous picking and interactive jogging.

### Grasp Width Matters
panda-py's `gripper_grasp(width)` checks if the final finger width is within &epsilon; of the target. Setting `width=0.001` for a 30mm block causes the grasp to "fail" -- the force is not maintained and the block slips. Setting `width=0.03` (matching the actual block) maintains 70N force and holds the block securely.

### Depth Fusion Pipeline
Fusing 2D object detection (YOLO on USB camera) with 3D depth pointcloud (PhotoNeo) produces labeled, 3D-positioned scene graphs. Grid-based BFS clustering (no scipy dependency) with dimensional filtering removes noise from the ~47% coverage depth scans.

### Incremental Lowering
Moving from home position directly to table height (z=0.013m) in one step causes large joint changes that trigger the Franka's joint reflex safety. Instead, `pick_at` lowers in 4cm incremental steps, re-solving IK at each step with the current joint configuration as seed. This produces smooth, safe descents.

## The Bigger Picture: Skill VLM

This project is evolving toward training a small **Vision-Language Model** to output parameterized skill calls:

```
Current:   Camera → Claude (LLM reasoning) → skill calls → Robot
Future:    Camera → SmolVLM2-500M (local VLM) → skill calls → Robot
```

The key insight: dense trajectory VLAs (predicting joint deltas at 10Hz) are a poor fit for pick-and-place. **Parameterized skill calls** (`pick(x, y)`, `place(x, y)`, `done`) are vastly more effective. The pipeline:

1. **Data collection** (built): `skill_episode_start` → `execute_plan` → `skill_episode_stop` captures camera frames before each skill call
2. **Dataset conversion** (built): Episodes converted to JSONL with HuggingFace chat template for VLM fine-tuning
3. **Training** (next): Fine-tune SmolVLM2-500M on DGX Spark with TRL SFTTrainer
4. **Inference** (next): Local VLM predicts one skill at a time, re-captures image after each execution

The depth camera is a training-time scaffold -- the learned skill model only needs a cheap webcam.

## Quick Start

### Prerequisites
- Raspberry Pi 5 with RT kernel
- Franka Panda with FCI enabled
- Python 3.11+

### Installation
```bash
git clone https://github.com/ratsbane/panda-mcp.git
cd panda-mcp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure Claude Code
Add to `.claude/settings.local.json`:
```json
{
  "mcpServers": {
    "franka-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "franka_mcp.server"],
      "cwd": "/path/to/panda-mcp"
    },
    "camera-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "camera_mcp.server"],
      "cwd": "/path/to/panda-mcp"
    },
    "depth-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "depth_mcp.server"],
      "cwd": "/path/to/panda-mcp"
    },
    "voice-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "voice_mcp.server"],
      "cwd": "/path/to/panda-mcp"
    },
    "so100-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "so100_mcp.server"],
      "cwd": "/path/to/panda-mcp"
    }
  }
}
```

### Run
```bash
claude  # Start Claude Code in the panda-mcp directory
```

Then just ask: *"Connect to the robot and pick up the red block."*

### Web Viewer
```bash
python -m camera_viewer.app  # Serves on http://localhost:8080
```

Navigate to `/jog` for the remote arm control interface.

## MCP Tools Reference

### franka-mcp (30 tools)
**Motion**: `connect`, `get_status`, `move_cartesian`, `move_relative`, `move_joints`, `move_cartesian_sequence`, `move_joint_sequence`, `gripper_move`, `gripper_grasp`, `pick_at`, `place_at`, `execute_plan`, `stop`, `recover`, `teaching_mode`

**Jog**: `jog_enable`, `jog_disable`, `jog_status`

**VLA**: `vla_enable`, `vla_disable`, `vla_status`, `start_recording`, `stop_recording`, `get_recording_status`, `list_episodes`

**SAWM**: `sawm_enable`, `sawm_disable`, `sawm_status`, `sawm_collect_enable`, `sawm_collect_disable`, `sawm_collect_stats`

**Skill logging**: `skill_episode_start`, `skill_episode_stop`, `skill_episode_list`

**Safety**: `get_safety_limits`, `set_safety_limits`

### camera-mcp (7 tools)
`connect`, `capture_frame`, `capture_burst`, `get_camera_info`, `set_resolution`, `describe_scene`, `describe_scene_3d`, `ground_object`

### depth-mcp (5 tools)
`connect`, `capture_depth`, `get_depth_at`, `get_robot_coords_at`, `save_scan`

### voice-mcp (4 tools)
`speak`, `list_voices`, `set_voice`, `get_voice_status`

### so100-mcp (14 tools)
`connect`, `get_status`, `move_joint`, `move_joints`, `gripper_open`, `gripper_close`, `home`, `wave`, `enable_torque`, `disable_torque`, `stop`, `discover_ports`, `diagnose`, `calibrate_joint`

## Project Structure
```
panda-mcp/
├── franka_mcp/        # Panda arm control, IK solver, gamepad jog, plan executor
├── camera_mcp/        # USB camera, Hailo YOLOv8, MobileSAM, visual grounding
├── depth_mcp/         # PhotoNeo 3D camera (SSH to PhoXi Control host)
├── voice_mcp/         # Piper text-to-speech
├── so100_mcp/         # SO-ARM100 servo arm control
├── camera_daemon/     # ZeroMQ camera frame publisher
├── camera_viewer/     # Web UI: camera stream, fusion overlay, remote jog control
├── common/            # Shared: scene interpretation, depth fusion, calibration,
│                      #   grounding client, skill logger, SAWM
├── calibration/       # Saved calibration data (ArUco homography, depth transform)
├── scripts/           # Calibration, dataset conversion, training
├── systemd/           # Service files for daemons
└── aruco_markers/     # Printable calibration markers
```

## Calibration

Two calibration systems transform between camera pixels and robot coordinates:

- **2D (USB camera)**: ArUco marker homography -- 4+ markers with known robot positions, ~2-4cm accuracy
- **3D (PhotoNeo)**: Kabsch/SVD rigid transform (SE(3)) using 4+ ArUco markers, ~9mm RMSE, sub-1mm Z accuracy

Both use ArUco 4X4_50 markers taped to the desk. Robot positions are recorded via teaching mode or gamepad jog. Scripts in `scripts/` automate the calibration process.

## License

MIT

## Acknowledgments

- [Claude Code](https://github.com/anthropics/claude-code) by Anthropic
- [panda-py](https://github.com/JeanElsner/panda-py) by Jean Elsner
- [Hailo](https://hailo.ai/) for the AI Hat+ 2
- [PhotoNeo](https://www.photoneo.com/) for the MotionCam-3D
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Piper TTS](https://github.com/rhasspy/piper)
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) by Alibaba Cloud
