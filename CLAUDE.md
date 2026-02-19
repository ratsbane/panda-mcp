# panda-mcp

Claude Code controls a Franka Emika Panda 7-DOF robot arm via MCP. The system runs on a Raspberry Pi 5 with a Hailo-10H AI accelerator hat and connects to the robot over direct Ethernet.

## Architecture

```
User → Claude Code → MCP Servers → Hardware
         ↓
   (reasoning, planning)
         ↓
   Vision → Scene Understanding → Motion Planning → Execution
```

Five MCP servers, each independent:
- **franka-mcp** — Arm + gripper control (panda-py, analytical IK)
- **camera-mcp** — USB RGB camera, Hailo YOLOv8, MobileSAM, Qwen2.5-VL grounding
- **depth-mcp** — PhotoNeo MotionCam-3D (SSH to remote host `tuppy`)
- **voice-mcp** — Piper TTS narration
- **so100-mcp** — SO-ARM100 secondary 6-DOF servo arm

## Key Design Decisions

### Analytical IK over Cartesian planning
Franka's built-in Cartesian planner (`move_to_pose`) had 3-7cm cross-axis drift. All motion now uses `panda_py.ik()` with q7 candidate search and cost-based selection. This gives 1-4mm accuracy.

### Incremental lowering
Moving from home height (z=0.3m) to table (z=0.013m) in one go triggers Franka's joint reflex safety. Solution: descend in 4cm steps, re-seeding IK at each step.

### Timeout protection
All blocking panda-py/libfranka calls wrapped in ThreadPoolExecutor with explicit timeouts (MOTION_TIMEOUT_S=15, GRIPPER_TIMEOUT_S=10, STATE_TIMEOUT_S=5). This was added after discovering executor.shutdown(wait=True) defeated the timeout mechanism.

### Waypoint interpolation
Large joint moves are interpolated through intermediate configurations to prevent jerky motion.

### Parameterized skills over dense trajectories
Pick/place as discrete skill calls (`pick(x,y)`, `place(x,y)`) rather than joint deltas at 10Hz. More learnable, more robust.

## Hardware

| Component | Details |
|-----------|---------|
| CPU | Raspberry Pi 5, 16GB RAM, RT kernel |
| Robot | Franka Emika Panda, direct Ethernet at 192.168.0.253 |
| AI Accelerator | Hailo-10H Hat+ 2 (40 TOPS) |
| 3D Camera | PhotoNeo MotionCam-3D via SSH to `tuppy` |
| RGB Camera | USB webcam (ZeroMQ daemon for shared access) |
| Secondary Arm | SO-ARM100 (6-DOF, Feetech STS3215 servos) |
| Training GPU | NVIDIA DGX Spark (Grace Blackwell, 128GB unified) |

## Network

- Raspberry Pi: 192.168.0.x (local)
- Franka Panda: 192.168.0.253 (direct Ethernet)
- PhotoNeo host: `tuppy` (SSH)
- DGX Spark: `spark` (Qwen grounding at :8090, VLA inference at :8085)

## Vision Pipeline

Two-level perception:
1. **USB camera** — Fast semantic detection via Hailo YOLOv8 (27ms/frame, COCO 80 classes)
2. **PhotoNeo depth** — Accurate 3D positions via structured light (1680x1200)

These fuse in `describe_scene_3d`: YOLO 2D detections matched to depth pointcloud clusters via BFS grid clustering + XY proximity matching.

Visual grounding via Qwen2.5-VL-3B on Spark for natural language queries ("the red block behind the green one").

## Coordinate Transforms

- USB camera pixel → robot XY: homography matrix in `calibration.json`
- Depth camera 3D → robot frame: Kabsch/SVD rigid transform in `calibration/depth_calibration.npz`

## Data Collection & Training Path

The system is designed to evolve from Claude-based reasoning to a local VLM:

```
Current:  Camera → Claude (LLM) → skill calls → Robot
Target:   Camera → SmolVLM2-500M (local) → skill calls → Robot
```

Three data pipelines:
1. **Skill episodes** (`data/skill_episodes/`) — (image, skill_call, result) tuples for VLM training
2. **Trajectory recordings** (`data/recordings/`) — Joint states + frames at 30FPS for VLA training
3. **SAWM servo data** (`data/sawm_servo_approaches/`) — Progressive crops for self-supervised servo learning

Conversion scripts in `scripts/`: `convert_to_skill_dataset.py` (HuggingFace chat format), `convert_to_lerobot.py`.

## Code Conventions

- **Mock mode**: `FRANKA_MOCK=1` env var enables development without hardware
- **Lazy singletons**: Vision components (Hailo, MobileSAM) use lazy init with checked flag
- **Dataclasses**: `RobotState`, `DetectedObject`, `SceneGraph3D` for structured data
- **JSON responses**: All MCP tools return via `json_response()` helper
- **Timeout wrapping**: `_call_with_timeout(fn, timeout, label, recover)` pattern

## Key Files

- `franka_mcp/controller.py` — Core arm control, IK solver, pick/place logic (~1100 lines)
- `franka_mcp/server.py` — MCP tool definitions (30 tools)
- `camera_mcp/server.py` — Vision MCP tools
- `common/hailo_detector.py` — YOLOv8 on Hailo-10H
- `common/depth_fusion.py` — 3D scene fusion (YOLO + depth pointcloud)
- `common/sawm/` — Self-supervised servo learning (5 modules)
- `common/skill_logger.py` — Skill episode recording for VLM training
- `calibration.json` — USB camera homography + workspace bounds

## IK Solver Tuning (current values)

- `max_single_joint_change`: 2.5 rad
- Joint limit margin: 0.10 rad
- FK verification: position error < 3mm
- q7 candidates: current → nearby offsets → reference → fixed values
- Cost: travel + limit_penalty + large_change_penalty

## Grasp Detection

Force/torque monitoring during grasp:
- `GRIP_CLOSED_EMPTY = 0.0015m` (fully closed, nothing held)
- `GRIP_STILL_OPEN = 0.075m` (didn't close)
- `GRIP_SLIP_TOLERANCE = 0.008m` (width change during transport)
- `FORCE_DROP_THRESHOLD = 2.0 N` (Fz change suggesting drop)
- Monitors `O_F_ext_hat_K` (6D EE force/torque) and `tau_ext_hat_filtered` (joint torques)
