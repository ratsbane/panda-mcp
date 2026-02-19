# Lab Notebook - panda-mcp

## 2026-02-18 ~afternoon - First Learning Session

### Context
Started a fresh Claude Code session (no --resume) to reset the project direction. The goal: learn to manipulate objects through direct interaction and feedback, rather than using pre-built high-level skills. This is the beginning of a developmental approach to motor learning.

### Workspace Setup
- Dark wood table with white paper in the center
- Objects: green block, natural wood block, dark brown block, 2 red blocks, blue block, tan block, tiger toy
- ArUco calibration markers around the edges
- USB camera mounted above and behind the workspace, looking down at an angle

### Experiment 1: Spatial Mapping
**Goal:** Understand the relationship between robot coordinates and camera image positions.

**Observations:**
- Moved to several positions at z=0.25 and photographed:
  - (0.4, 0, 0.25): gripper appeared right-of-center in image
  - (0.4, 0.15, 0.25): gripper shifted LEFT in image
- At HIGH z (near camera), +Y robot = LEFT in image
- At LOW z (near table), the mapping appears different due to perspective - at (0.48, 0.15, 0.02), the gripper appeared on the RIGHT side of the image
- **Key insight:** The camera is at an angle, not directly overhead. The apparent position of the gripper in the image depends heavily on Z height due to perspective projection. This makes simple pixel-offset-to-robot-offset conversion unreliable.

### Experiment 2: Finding the Red Block
**Goal:** Position gripper over a red block and pick it up.

**Approach 1 - Manual estimation:**
- Guessed block at (0.45, 0.12) based on image position - was wrong
- Guessed (0.52, 0.18) - also wrong
- At (0.48, 0.05, 0.08): gripper was directly over the red block
- **Learned:** The red block on the right side of the white paper is approximately at robot (0.48, 0.05) in XY

**Approach 2 - Visual grounding model:**
- Used `ground_object("the red block on the right edge of the white paper")`
- Returned robot_coords_approx: (0.387, 0.225)
- This is VERY wrong compared to empirical (0.48, 0.05) - off by ~9cm in X and ~17cm in Y
- Moved to grounding model's suggested coordinates: ended up in upper-left corner, nowhere near the block
- **Learned:** The USB camera homography is severely miscalibrated or the grounding model is identifying wrong objects. Cannot trust homography-based robot coordinates.

**Approach 3 - Top-down descent:**
- Positioned at approximately (0.48, 0.05-0.08) above the block
- Lowered incrementally (4cm, 3cm, 2.5cm steps)
- Repeatedly caught a gripper finger on the edge of the block
- Problem: can't judge precise XY alignment from the angled camera, especially as the gripper gets close and occludes the view
- **Learned:** Top-down approach is fragile - small alignment errors cause the gripper to catch on block edges during descent

**Approach 4 - Lateral approach:**
- Moved to (0.48, 0.15, 0.025) - clearly to one side of the block at TABLE height
- Slid in -Y direction toward the block
- At (0.48, 0.11, 0.023): getting very close to the block
- Still in progress but the lateral approach avoids the descent-alignment problem
- **Learned:** Approaching from the side at table height may be more robust than descending from above

### Key Learnings

1. **The homography is not trustworthy.** The pixel-to-robot coordinate transform gives errors of 10+ cm. Direct visual feedback is more reliable than computed transforms.

2. **Perspective makes visual servoing hard from this camera.** The camera is at an oblique angle. The same XY offset on the table produces different pixel offsets depending on the Z height of the object being observed. A top-down camera would make visual servoing much simpler.

3. **Incremental visual feedback is essential.** Every move should be followed by a camera check. "Move blind" commands based on computed coordinates fail.

4. **Descent is the hardest part.** Getting roughly over the block is manageable. Lowering without catching a gripper finger on the block edge requires precise alignment that's hard to achieve from this camera angle.

5. **Lateral approach may be better than top-down.** Moving to the side of the block at table height, then sliding in, avoids the descent alignment problem entirely.

6. **The current workflow is too slow.** Each move-photograph-reason cycle takes several seconds of LLM processing. A reactive system needs to run this loop at hardware speed, not reasoning speed.

### Architecture Ideas for Reactive System

The core problem: the feedback loop (observe -> decide -> act) runs at "Claude reasoning speed" (~5-10 seconds per cycle). For effective motor control, this needs to run at camera framerate (~30Hz).

**Possible approach:**
1. Train a small neural network that takes a camera frame and outputs a correction vector (dx, dy, dz) to move the gripper toward a target
2. Run this in a tight loop on the Pi (or Hailo accelerator)
3. Claude's role shifts from "controlling every movement" to "setting the goal and monitoring progress"

**Training data source:** The very move-photograph sequences I'm generating now. Each (camera_frame, gripper_position, target_position) triple is a training sample. The label is the direction the gripper needs to move.

**Open questions:**
- What's the minimum viable model architecture? Could a simple CNN work?
- How to represent the "target" to the model? (pixel coordinates? natural language?)
- Could the Hailo accelerator run this at framerate?
- How to handle the transition from visual servoing to contact/force-based grasping?

## 2026-02-19 ~afternoon - First Successful Autonomous Pick

### Context
Continuing from yesterday's learning session. Built `learned/visual_servo.py` with color detection + pixel-to-robot coordinate mapping. Integrated as `learned_pick` tool in franka-mcp server.

### Experiment 3: Autonomous Pick Attempts

**Setup:** Two red blocks on the right side of the white paper. Using `learned_pick` tool which: opens gripper → moves arm clear → detects target color → computes robot position → positions above target → descends → grasps → lifts.

**Attempt sequence (all targeting red block):**

| # | y_bias | Robot estimate | Result |
|---|--------|---------------|--------|
| 1 | +0.02 | (0.514, 0.076) | Miss - went further camera-right (wrong direction) |
| 2 | -0.02 | (0.510, 0.046) | Miss - right direction, hit top of block |
| 3 | -0.03 | (0.511, 0.035) | Miss - still slightly off |

**Key problem:** The original calibration point was wrong. Had pixel (801, 493) = robot (0.48, 0.06), but the actual correct position was significantly different.

**Breakthrough - Human demonstration:**
- User took gamepad control and positioned gripper perfectly around the red block
- Tilted gripper toward camera for visual verification
- Recorded ground truth: **robot (0.489, 0.016, 0.028)** for correct grasp position
- This revealed the calibration Y was off by 4.4cm (0.06 vs 0.016)

**Calibration update:**
- Updated calibration point: pixel (795, 534) → robot (0.489, 0.016)
- Set y_bias to 0 (no longer needed with correct calibration)
- Updated grasp_height from 0.020 to 0.025

**Attempt 4 (with corrected calibration):**
- Detected at pixel (814, 535), estimated robot (0.490, 0.008)
- Descended, grasped: `libfranka_success: true`, `final_width: 0.029`
- **SUCCESS** - block held after lifting

### Key Learnings

7. **One good calibration point is worth many bias corrections.** I spent three attempts tuning a y_bias parameter when the underlying calibration point was wrong by 4.4cm. The user's 30-second gamepad demonstration gave me the ground truth that fixed everything.

8. **Camera-right vs robot-right is confusing at table level.** Due to the camera's oblique angle, the mapping between image directions and robot directions flips between high-Z and low-Z. At table level: robot +Y ≈ camera-right (counterintuitive if you think of +Y as left from the camera's perspective at height).

9. **The pick sequence works:** open gripper → move clear → detect → position horizontally at approach height → descend vertically → grasp → lift. Horizontal-first, vertical-second is essential to avoid pushing the block.

10. **Grasp height matters.** 0.020m was slightly too low, 0.025m worked. The block has nonzero height and the gripper needs to be at the right Z to close around the widest part.
