# Panda MCP: An LLM-Controlled Robot Arm

**Claude Code autonomously picks up objects using a 7-DOF Franka Panda arm, 3D depth camera, AI-accelerated computer vision, and analytical inverse kinematics -- all orchestrated through the Model Context Protocol (MCP).**

<div align="center">
  <a href="https://www.youtube.com/watch?v=Zig-_-1gK1Y">
    <img src="https://img.youtube.com/vi/Zig-_-1gK1Y/maxresdefault.jpg" alt="Demo Video" width="700">
  </a>
  <p><strong>▶ Watch the demo video</strong></p>
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

### 3D Scene Understanding
The `describe_scene_3d` tool fuses two sensing modalities:
- **USB camera + Hailo YOLOv8**: Object class labels and 2D bounding boxes
- **PhotoNeo depth camera**: Accurate 3D pointcloud transformed to robot frame via Kabsch/SVD calibration

Output: A labeled 3D scene graph with object positions, dimensions, and suggested grasp widths.

### Gamepad Teleoperation
Xbox 360 controller integrated for manual arm positioning via IK-based jogging at up to 50mm/step, 20Hz. Used for calibration, ground-truth positioning, and interactive demos.

### Voice Narration
Piper TTS announces actions during demos ("Scanning the workspace...", "Picking up the green block...").

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Claude Code (Opus 4)                         │
│              Reasoning, Planning, Error Recovery                │
└──────┬──────────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │          │
       │ MCP      │ MCP      │ MCP      │ MCP      │ MCP
       │ (stdio)  │ (stdio)  │ (stdio)  │ (stdio)  │ (stdio)
       ▼          ▼          ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│franka-mcp││camera-mcp││depth-mcp ││voice-mcp ││so100-mcp │
│          ││+Hailo    ││          ││          ││          │
│ panda-py ││ YOLOv8   ││ PhotoNeo ││ Piper    ││ Feetech  │
│ IK/FK    ││ MobileSAM││ SSH+SCP  ││ TTS      ││ STS3215  │
└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘└────┬─────┘
     │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼
  Franka     USB Cam +    PhotoNeo     HDMI       SO-ARM100
  Panda      Hailo-10H   MotionCam    Audio      (6-DOF)
```

**All computation runs on a Raspberry Pi 5** (16GB, RT kernel) -- the AI hat provides 40 TOPS for real-time neural network inference.

## Hardware

| Component | Details |
|-----------|---------|
| **Controller** | Raspberry Pi 5, 16GB RAM, RT kernel |
| **Robot Arm** | Franka Emika Panda (7-DOF, 1kg payload) |
| **AI Accelerator** | Hailo-10H AI Hat (40 TOPS, PCIe) |
| **3D Camera** | PhotoNeo MotionCam-3D (structured light, via remote PhoXi Control) |
| **RGB Camera** | USB webcam via ZeroMQ daemon |
| **Audio** | HDMI audio output, Piper TTS |
| **Second Arm** | SO-ARM100 (6-DOF, Feetech servos) -- for future bimanual tasks |
| **Training GPU** | NVIDIA DGX Spark (Grace Blackwell, 128GB) -- for VLA fine-tuning |

## Technical Breakthroughs

### Analytical IK Beats Cartesian Planning
The Franka's built-in Cartesian planner (`move_to_pose`) exhibited 3-7cm cross-axis drift -- moving in X would couple into Y and Z. Replacing all motion with **analytical IK** (`panda_py.ik_full` + `move_to_joint_position`) achieved 1-4mm accuracy, enabling reliable autonomous grasping.

### Wrist Orientation Control
IK solutions that minimize joint travel from the current configuration can cause the wrist to drift to bad orientations over successive moves. We enforce a **straight-down picking orientation** (roll=&pi;, pitch=0, yaw=0) and add a **q7 preference bias** in the cost function to keep the wrist stable.

### Grasp Width Matters
panda-py's `gripper_grasp(width)` checks if the final finger width is within &epsilon; of the target. Setting `width=0.001` for a 30mm block causes the grasp to "fail" -- the force is not maintained and the block slips. Setting `width=0.03` (matching the actual block) maintains 70N force and holds the block securely.

### Depth Fusion Pipeline
Fusing 2D object detection (YOLO on USB camera) with 3D depth pointcloud (PhotoNeo) produces labeled, 3D-positioned scene graphs. Grid-based BFS clustering (no scipy dependency) with dimensional filtering removes noise from the ~42% coverage depth scans.

## The Bigger Picture: VLA Skill Learning

This project is scaffolding for training **Vision-Language-Action (VLA)** models:

1. **Data collection**: Claude teleoperates the arm via MCP + depth camera (privileged sensing)
2. **Training**: Fine-tune a VLA model on the DGX Spark using LeRobot dataset format
3. **Deployment**: VLA runs on the Hailo-10H using only RGB webcam input (no depth camera needed at inference)

The depth camera is a training-time scaffold -- the learned policy only needs a cheap webcam.

```
Training:  Claude (LLM reasoning) + Depth Camera → Demonstrations → VLA Model
Inference: RGB Camera → VLA on Hailo-10H → Joint Commands → Robot
```

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
    "voice-mcp": {
      "command": "./venv/bin/python",
      "args": ["-m", "voice_mcp.server"],
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

## MCP Tools Reference

### franka-mcp (17 tools)
`connect`, `get_status`, `move_cartesian`, `move_relative`, `move_joints`, `move_cartesian_sequence`, `move_joint_sequence`, `gripper_move`, `gripper_grasp`, `pick_at`, `place_at`, `stop`, `recover`, `teaching_mode`, `jog_enable`, `jog_disable`, `jog_status`

### camera-mcp (5 tools)
`connect`, `capture_frame`, `capture_burst`, `describe_scene`, `describe_scene_3d`

### depth-mcp (4 tools)
`connect`, `capture_depth`, `get_depth_at`, `get_robot_coords_at`, `save_scan`

### voice-mcp (4 tools)
`speak`, `list_voices`, `set_voice`, `get_voice_status`

### so100-mcp (12 tools)
`connect`, `get_status`, `move_joint`, `move_joints`, `gripper_open`, `gripper_close`, `home`, `wave`, `enable_torque`, `disable_torque`, `discover_ports`, `diagnose`

## Project Structure
```
panda-mcp/
├── franka_mcp/        # Panda arm control, IK solver, gamepad jog
├── camera_mcp/        # USB camera, Hailo YOLOv8, MobileSAM
├── depth_mcp/         # PhotoNeo 3D camera (SSH to PhoXi Control host)
├── voice_mcp/         # Piper text-to-speech
├── so100_mcp/         # SO-ARM100 servo arm control
├── camera_daemon/     # ZeroMQ camera frame publisher
├── camera_viewer/     # Flask web viewer with fusion overlay
├── common/            # Shared: vision, depth fusion, calibration, recording
├── scripts/           # Calibration, data collection, training
├── systemd/           # Service files for daemons
└── aruco_markers/     # Calibration markers
```

## Calibration

Two calibration systems transform between camera pixels and robot coordinates:

- **2D (USB camera)**: ArUco marker homography -- 4 markers with known robot positions, ~2cm accuracy
- **3D (PhotoNeo)**: Kabsch/SVD rigid transform (SE(3)) using 5+ ArUco markers, 9mm RMSE, sub-1mm Z accuracy

## License

MIT

## Acknowledgments

- [Claude Code](https://github.com/anthropics/claude-code) by Anthropic
- [panda-py](https://github.com/JeanElsworthy/panda-py) by Jean Elsworthy
- [Hailo](https://hailo.ai/) for the AI Hat
- [PhotoNeo](https://www.photoneo.com/) for the MotionCam-3D
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Piper TTS](https://github.com/rhasspy/piper)
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
