# Project Page Outline: Panda MCP

## Project Title and Tagline

**Title:** Panda MCP - Giving AI Physical Embodiment

**Tagline Options:**
- "Claude with hands: AI-controlled robot manipulation through natural conversation"
- "See. Move. Speak. An LLM-controlled robot arm that closes the perception-action loop."
- "MCP servers that give any AI agent eyes, hands, and a voice"

---

## 1. Problem Statement / Motivation

### The Gap Between AI Intelligence and Physical Action
- Modern LLMs can reason, plan, and communicate, but lack physical presence
- Robotics traditionally requires specialized programming, not natural language
- Bridging this gap would make robots accessible to anyone who can describe a task

### Why MCP (Model Context Protocol)?
- Open standard for connecting AI to external tools
- Any MCP-compatible agent (Claude Code, Codex, custom agents) can control the robot
- Modular design: swap cameras, arms, or AI models independently
- No robot-specific code in the AI - just tool calls

### The Vision
> "Pick up the green block and put it in the basket"
>
> The AI captures an image, identifies objects, plans movements, executes them, and verifies success - all through natural language understanding.

---

## 2. Architecture Overview

### System Diagram
```
                    Claude / Codex / Any MCP Agent
                              |
                    MCP Protocol (stdio/JSON-RPC)
                              |
        +---------+-----------+-----------+
        |         |           |           |
   franka-mcp  camera-mcp  voice-mcp   (future: depth-mcp)
        |         |           |
        |     ZeroMQ pub/sub  |
        |         |           |
        v         v           v
   Franka Panda  USB Camera  USB Speaker
   (7-DOF arm)   + MobileSAM  + Piper TTS
```

### Component Summary

| Component | Purpose | Key Technology |
|-----------|---------|----------------|
| **franka-mcp** | Robot arm control | panda-py, libfranka FCI |
| **camera-mcp** | Visual perception | OpenCV, MobileSAM segmentation |
| **camera-daemon** | Frame sharing | ZeroMQ pub/sub |
| **voice-mcp** | Speech output | Piper TTS (local, fast) |
| **common/** | Shared utilities | Scene interpretation, calibration |

### Hardware Stack
- **Compute:** Raspberry Pi 5 (16GB) with RT kernel
- **Robot:** Franka Emika Panda (7-DOF research arm)
- **Vision:** USB camera (future: PhotoNeo depth camera)
- **Audio:** USB speaker for voice feedback
- **Network:** Direct Ethernet, Pi as DHCP server

---

## 3. Key Features and Capabilities

### Robot Control (franka-mcp)
- **Cartesian movements:** Move end-effector to (x, y, z) with orientation
- **Joint control:** Direct joint angle control for complex poses
- **Smooth trajectories:** Sequence multiple waypoints as continuous motion
- **Gripper control:** Position control and force-limited grasping
- **Safety built-in:** Workspace limits, velocity limits, large-move confirmation
- **Error recovery:** Automatic recovery from collision/limit errors

### Visual Perception (camera-mcp)
- **Multi-method object detection:**
  - Color-based detection (colored blocks, toys)
  - Contour/edge detection (general shapes)
  - MobileSAM instance segmentation (accurate object boundaries)
- **Scene interpretation:**
  - Natural language scene descriptions
  - Spatial relationships ("block A is left of block B")
  - Object attributes (color, size, position, distance estimate)
- **Frame sharing:** ZeroMQ daemon allows multiple clients (MCP, viewer, data collection)

### Voice Output (voice-mcp)
- **Local TTS:** Piper running entirely on-device (no cloud)
- **Natural speech:** Sentence splitting for proper pacing
- **Audio primer:** Eliminates USB speaker first-syllable clipping
- **Non-blocking mode:** Claude can speak while continuing to work

### Training Infrastructure
- **Data collection scripts:** Random, grid, or manual sampling modes
- **Gripper localizer model:** CNN (MobileNetV3) predicting (x, y, z) from images
- **ONNX export:** Models optimized for Pi inference
- **Multi-backbone training:** Comparison across MobileNetV3, EfficientNet variants

---

## 4. Demo Scenarios (Video Ideas)

### Demo 1: Object Sorting
**Script:** "Sort the blocks by color - put red blocks on the left, blue on the right."

**What it shows:**
- Scene capture and object detection
- Color recognition
- Multi-step planning
- Pick and place execution
- Verification between moves

### Demo 2: Conversational Manipulation
**Script:** Natural back-and-forth conversation:
- Human: "What do you see?"
- Claude: [describes scene, speaks aloud]
- Human: "Can you pick up the elephant?"
- Claude: [locates elephant, grasps, lifts]
- Human: "Now wave hello"
- Claude: [executes waving motion sequence]

**What it shows:**
- Multimodal interaction (see, speak, act)
- Natural language understanding
- Continuous conversation

### Demo 3: Closed-Loop Correction
**Script:** Human moves an object mid-task

**What it shows:**
- Re-perception after unexpected change
- Adaptive planning
- Robustness to perturbation

### Demo 4: Learning to See (Training Data Collection)
**Script:** Time-lapse of robot collecting training data

**What it shows:**
- Autonomous data collection
- Camera + robot state synchronization
- The path toward learned visuomotor policies

---

## 5. Technical Details Section

### 5.1 Real-Time Control
- RT kernel (`linux-image-rpi-v8-rt`) for sub-millisecond latency
- Direct Ethernet link (no switches/routers)
- panda-py wrapper over libfranka FCI

### 5.2 Scene Understanding Pipeline
```
Image -> [Color Detection] -> [Contour Detection] -> [MobileSAM (optional)]
                     \              |                /
                      \             v               /
                       -> Object Fusion & Dedup <-
                                    |
                                    v
                        Spatial Relationship Analysis
                                    |
                                    v
                        Natural Language Summary
```

### 5.3 Gripper Localizer Training
- **Problem:** Predict gripper 3D position from 2D camera image
- **Challenge:** Single viewpoint provides insufficient depth information
- **Dataset v1:** 6,551 images from fixed camera -> all models converge to ~200mm error (predicting mean)
- **Dataset v2:** Multi-viewpoint collection (in progress)
- **Architecture:** MobileNetV3-Small backbone + regression head
- **Insight:** Spatial augmentations (flip, crop) break position labels; only color jitter used

### 5.4 ZeroMQ Camera Architecture
```
USB Camera (/dev/video0)
        |
        v
camera-daemon (systemd user service)
        |
        | ZeroMQ PUB (ipc:///tmp/camera-daemon.sock)
        |
    +---+---+---+
    |   |   |   |
  MCP  Viewer  Data Collection  (any subscriber)
```
- Solves Linux V4L2 single-process limitation
- Daemon runs at boot (systemd + linger)
- Multiple clients subscribe/unsubscribe freely

### 5.5 Safety System
- **Workspace bounds:** Configurable 3D bounding box
- **Velocity limits:** Max translation 0.1 m/s, rotation 0.5 rad/s
- **Large move confirmation:** Moves >20cm require explicit confirmation
- **Dry run mode:** Test motion planning without hardware movement
- **Error recovery tool:** Claude can recover from collision/limit faults

---

## 6. Getting Started / Installation

### Prerequisites
- Raspberry Pi 5 (16GB recommended) with Raspberry Pi OS
- Franka Emika Panda robot arm
- USB camera
- USB speaker (optional, for voice)

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/[user]/panda-mcp.git
cd panda-mcp

# 2. Install RT kernel (recommended)
sudo apt install linux-image-rpi-v8-rt
sudo reboot

# 3. Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Configure network for robot
sudo nmcli con add con-name franka-direct type ethernet ifname eth0 \
    ipv4.addresses 192.168.0.2/24 ipv4.method shared
sudo nmcli con up franka-direct

# 5. Set up camera daemon
mkdir -p ~/.config/systemd/user
cp systemd/camera-daemon.service ~/.config/systemd/user/
systemctl --user enable --now camera-daemon
sudo loginctl enable-linger $USER

# 6. Configure Claude Code MCP servers
# Add to ~/.claude.json (see README for full config)

# 7. Launch Claude Code in this directory
claude
```

### Testing Without Hardware
```bash
# Mock mode for development
FRANKA_MOCK=1 python -m franka_mcp.server
CAMERA_MOCK=1 python -m camera_mcp.server
```

---

## 7. Future Directions

### Near-Term
- [ ] **Depth camera integration:** PhotoNeo or Intel RealSense for accurate Z estimation
- [ ] **Wrist-mounted camera:** Alternative viewpoint, closer to manipulation
- [ ] **Complete multi-viewpoint training:** Test hypothesis that multiple camera positions improve localization

### Medium-Term
- [ ] **Visual servoing policy:** CNN predicts (dx, dy, dz) given image + target pixel
- [ ] **Demonstration learning:** Record human teleoperation, train imitation policy
- [ ] **Force-sensitive grasping:** Use Panda's torque sensors for compliant manipulation

### Long-Term
- [ ] **VLA integration:** Connect vision-language-action models for high-level task understanding
- [ ] **Multi-robot coordination:** Multiple arms working together via MCP
- [ ] **Mobile manipulation:** Integrate with mobile base for room-scale tasks

### Research Questions
1. Can an LLM's reasoning ability compensate for imprecise perception?
2. What's the minimum viable perception for useful manipulation?
3. How should LLMs handle manipulation failures and retries?

---

## 8. Repository Structure

```
panda-mcp/
├── franka_mcp/           # Robot arm MCP server
│   ├── server.py         # MCP tool definitions
│   └── controller.py     # Hardware interface
├── camera_mcp/           # Camera MCP server
│   ├── server.py         # MCP tool definitions
│   └── controller.py     # Camera interface
├── camera_daemon/        # ZeroMQ frame publisher
│   ├── server.py         # Capture and publish
│   └── client.py         # Subscriber helper
├── voice_mcp/            # TTS MCP server
│   └── server.py         # Piper TTS integration
├── common/               # Shared utilities
│   ├── scene_interpreter.py  # Scene understanding
│   ├── segmentation.py       # MobileSAM wrapper
│   ├── vision.py             # Object detection
│   ├── calibration.py        # Camera-robot transforms
│   └── manipulation.py       # High-level helpers
├── models/               # Neural network models
│   └── gripper_localizer.py  # Position regression
├── scripts/              # Utility scripts
│   ├── collect_data.py       # Training data collection
│   ├── train_gripper_localizer.py  # Model training
│   └── visualize_dataset.py  # Data visualization
├── systemd/              # Service definitions
│   └── camera-daemon.service
├── voices/               # Piper TTS voice models
└── docs/                 # Documentation
```

---

## 9. Acknowledgments / Related Work

- **panda-py:** Python wrapper for libfranka
- **libfranka:** Franka Emika's C++ control interface
- **MCP Protocol:** Anthropic's Model Context Protocol
- **Piper TTS:** Fast local text-to-speech
- **MobileSAM:** Efficient segment-anything model
- **LeRobot (HuggingFace):** Inspiration for visuomotor learning approach
- **timm:** PyTorch image models for backbone selection

---

## 10. Call to Action

### For Researchers
- Explore LLM-based robot control without custom training
- Test hypothesis about reasoning vs. perception tradeoffs
- Contribute perception improvements

### For Developers
- Add new MCP servers (depth cameras, additional sensors)
- Improve scene understanding
- Contribute to training pipeline

### For the Curious
- Star the repo
- Watch demo videos
- Try the mock mode to explore the architecture

---

## Appendix: Key Metrics

| Metric | Value |
|--------|-------|
| Lines of Python | ~3,000 |
| MCP Tools Exposed | 23 (12 robot, 6 camera, 5 voice) |
| Supported Backbones | 7 (MobileNetV3, EfficientNet, ConvNeXt) |
| Training Samples Collected | 6,551+ |
| Workspace Volume | ~0.5m x 1.0m x 0.65m |
| Max Cartesian Velocity | 0.1 m/s (safety limited) |

---

*This outline is designed to be expanded into a full project page with screenshots, embedded videos, and interactive diagrams.*
