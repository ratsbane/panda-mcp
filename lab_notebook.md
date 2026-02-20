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

## 2026-02-19 ~evening - ArUco Homography: Reliable Picks Across the Workspace

### Context
The single-point calibration from earlier today only worked near the calibration point. Picking the green block (340 pixels from calibration) failed completely, with the Y estimate wildly off. The next step was building a proper pixel-to-robot coordinate mapping that generalizes across the workspace.

### Approach 1: Embedding Database (attempted, partially successful)
Built a DINOv2 embedding server on Spark (NVIDIA DGX, GPU inference) with SQLite-backed spatial memory on the Pi. Ran a 3x3 calibration sweep (9 positions across workspace), embedding each scene image. Verified embeddings encode spatial position: self-query similarity >0.99, neighbors are spatially closest.

However, extracting gripper pixel positions from the sweep images proved unreliable. Tried image differencing (sweep frame minus clean frame) to find the gripper, but the arm body dominated the diff, giving fingertip positions that were off by 50-200 pixels for some positions. Both affine and polynomial fits had 28-93mm mean errors — far too high for picking.

### Approach 2: ArUco Homography (breakthrough)
The table has 6 ArUco markers (DICT_4X4_50, IDs 0-5) with known robot-frame coordinates from a prior calibration. Detected all 6 markers with OpenCV in the current camera frame and recomputed the homography.

**Key finding:** The camera had shifted only ~5 pixels from the original calibration. The marker positions were nearly identical to the stored values, but the fresh homography was more accurate because it used the exact current pixel positions.

**Homography quality:**
- ArUco point residuals: 0.2-1.3mm (sub-millimeter)
- Verified at gamepad point (known ground truth): 8mm error
- 6 calibration points spanning 0.9m x 0.7m of workspace

### Experiment 4: Color Detection + Homography Picking

**Test 1 — Red block:**
- Color detection (HSV): pixel center (729, 377)
- Homography: robot (0.361, -0.003)
- `pick_at(0.361, -0.003)`: **SUCCESS** — gripper closed to 29.5mm, block held

**Test 2 — Green block:**
- Color detection (HSV): pixel center (454, 463)
- Homography: robot (0.402, -0.201)
- `pick_at(0.402, -0.201)`: **SUCCESS** — gripper closed to 28.4mm, block held

Both blocks were at very different positions in the workspace (180mm apart in X, 200mm apart in Y), confirming the homography generalizes well.

### Why the Embedding Approach Failed (for now)

The embedding database encodes scene-level spatial information, but the bottleneck was extracting accurate gripper pixel positions from the sweep images. Image differencing picks up the entire arm body (shoulder, wrist, gripper), not just the fingertips. The systematic error was always in the same direction: detected pixels were 37-205px to the right of the true fingertip position, because the arm extends rightward from the shoulder.

The ArUco markers bypass this problem entirely — they're designed for precise detection and have known fixed positions in the robot frame.

### Pipeline Summary

```
                  ┌─────────────┐
  Camera frame →  │ HSV color   │ → pixel (cx, cy)
                  │ detection   │
                  └─────────────┘
                        │
                        ▼
                  ┌─────────────┐
  pixel (cx,cy) → │ ArUco       │ → robot (x, y)
                  │ homography  │
                  └─────────────┘
                        │
                        ▼
                  ┌─────────────┐
  robot (x,y)  → │ pick_at()   │ → grasp
                  │ IK + descend│
                  └─────────────┘
```

### Embedding Server Thread Issue
The DINOv2 embedding server on Spark spawned ~50 PyTorch OpenMP/MKL threads (one per CPU core on the 20-core Grace CPU), each showing 1.2GB of shared memory in top. Fixed by setting `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` before torch import, plus `torch.set_num_threads(4)`. Also added port-availability check to prevent multiple server instances.

### Key Learnings

11. **ArUco markers make calibration trivial.** Six markers on the table, detected in one frame, give sub-mm homography accuracy. No need for complex gripper detection or image differencing.

12. **The homography was already good.** The original calibration had correct ArUco positions. The camera shifted only ~5 pixels since the original calibration. The earlier "homography is not trustworthy" conclusion (Learning #1) was wrong — the problem was that I was testing the homography with incorrectly-detected gripper pixel positions, not with properly-detected object positions.

13. **Color detection + homography is a robust pick pipeline.** HSV thresholding finds block centers reliably, the homography maps pixels to robot coordinates accurately, and pick_at handles the motion. Two successful picks at different workspace locations prove generalization.

14. **Image differencing is a poor gripper detector.** The arm body dominates the diff, pushing the detected "center" far from the actual fingertip position. For calibration, use fiducial markers (ArUco) or place objects at known positions instead.

## 2026-02-20 - Extended Picking Validation: 6/6 Across Full Workspace

### Context
Continued validation of the ArUco homography + HSV color detection pipeline from the previous session. Goal: test reliability across many picks at diverse workspace positions, including re-picks of moved objects.

### Experiment 5: Full Workspace Pick Series

| # | Color | Pixel center | Robot coords | Gripper width | Result |
|---|-------|-------------|-------------|---------------|--------|
| 1 | Red | (729, 377) | (0.361, -0.003) | 29.5mm | SUCCESS |
| 2 | Green | (454, 463) | (0.402, -0.201) | 28.4mm | SUCCESS |
| 3 | Orange/tan | (554, 531) | (0.462, -0.139) | 28.9mm | SUCCESS |
| 4 | Green (re-pick) | (925, 458) | (0.448, +0.111) | 28.4mm | SUCCESS |
| 5 | Red (re-pick) | (1052, 466) | (0.467, +0.191) | 29.5mm | SUCCESS |
| 6 | Blue | (789, 435) | (0.416, +0.026) | 29.2mm | SUCCESS |

**One IK failure (expected):** Blue block initially at robot (0.270, -0.105) — too close to robot base for straight-down IK solution. After user moved it to (0.416, 0.026), pick succeeded immediately.

### Coverage analysis
Picks spanned nearly the full reachable workspace:
- **X range:** 0.361 to 0.467 (106mm span, workspace is 0.2–0.6)
- **Y range:** -0.201 to +0.191 (392mm span, workspace is -0.2 to +0.2)
- Picks 2 and 5 were at opposite corners of the workspace (y=-0.201 vs y=+0.191)
- Re-picks of the same block at different positions (green: -0.201 then +0.111; red: -0.003 then +0.191) confirmed the pipeline works regardless of where the block is placed

### Observations
- **100% pick success rate** on reachable positions (6/6)
- **Consistent gripper widths** (28.4–29.5mm) suggest blocks are all similar size and grasps are centered
- **No near-misses or edge grasps** — the homography is accurate enough that the gripper centers well on each block
- **cartesian_reflex error** occurred once during place retreat (likely arm configuration near joint limits after a long sequence of picks). Recovered after manual intervention.

### Key Learnings

15. **The pipeline is robust across the full workspace.** 6/6 picks at positions spanning 106mm in X and 392mm in Y, with no calibration adjustments between picks. The ArUco homography generalizes well.

16. **Re-picking moved objects works.** The same block picked from two very different positions confirms the pipeline doesn't depend on the block being near any particular calibration point. The color detection + homography approach is truly position-independent.

17. **IK reachability is the main constraint, not calibration.** The only failure was an IK limitation (arm can't reach close to its base with straight-down orientation), not a calibration or detection error. The useful workspace for picking is roughly x=0.35–0.55, y=-0.20 to +0.20.

## 2026-02-20 ~evening - VLM Zero-Shot Pointing Benchmark

### Context
The long-term goal is to replace Claude-in-the-loop with a small local VLM that outputs parameterized skill calls (e.g. `{"skill": "pick", "x": 0.35, "y": -0.10}`). Before collecting training data, we need to choose which VLM to fine-tune. This experiment tests zero-shot spatial understanding — can these models already locate objects in our camera images?

### Experimental Setup

**Test image:** Single frame from the USB camera showing the workspace with 4 colored blocks on/near the white paper, the tiger toy, ArUco calibration markers, and the robot arm raised to z=0.35m.

**Ground truth:** Block positions from HSV color detection + ArUco homography, verified by 6/6 successful physical picks in the previous session.

| Block | Pixel (x, y) | Robot (x, y) m |
|-------|--------------|----------------|
| Green | (489, 453) | (0.398, -0.178) |
| Red | (608, 432) | (0.394, -0.096) |
| Blue | (783, 515) | (0.473, 0.005) |
| Orange/wood | (1090, 544) | (0.521, 0.185) |

**Models tested:**
1. Moondream 2B (1.9B params, 3.9GB VRAM) — built-in point() and detect() APIs
2. SmolVLM2-500M (500M params, 1.0GB VRAM) — text generation only
3. Florence-2-base (230M params) — could not run (incompatible with transformers 4.57.6)
4. Moondream 0.5B — not publicly available as separate model

All models run on DGX Spark (Grace Blackwell GPU, 128GB unified memory).

### Experiment 6: Moondream 2B — Point API

The `model.point(encoded_image, query)` method returns normalized (x, y) coordinates for the queried object. Image is encoded once, then each query runs against the encoding.

| Block | Query | Predicted pixel | GT pixel | Error |
|-------|-------|----------------|----------|-------|
| Green | "the green block" | (490, 442) | (489, 453) | 11.0 px |
| Red | "the red block on the white paper" | (606, 420) | (608, 432) | 12.2 px |
| Blue | "the blue block" | (782, 505) | (783, 515) | 10.0 px |
| Orange | "the wooden/tan block near the bottom right ArUco marker" | (1086, 537) | (1090, 544) | 8.1 px |

**Summary:** Mean 10.3 px, Median 10.5 px, Max 12.2 px. Latency: ~60-80ms per query (after image encoding).

### Experiment 7: Moondream 2B — Detect API

The `model.detect(encoded_image, query)` method returns bounding boxes. Error measured from bbox center to ground truth pixel center.

| Block | Query | Predicted center | GT center | Error | # Detections |
|-------|-------|-----------------|-----------|-------|-------------|
| Green | "green block" | (492, 452) | (489, 453) | 3.2 px | 1 |
| Red | "red block" | (607, 431) | (608, 432) | 1.4 px | 2 (found both reds!) |
| Blue | "blue block" | (783, 513) | (783, 515) | 2.0 px | 1 |
| Orange | "wooden block" | (1091, 546) | (1090, 544) | 2.2 px | 4 (found all wood blocks) |

**Summary:** Mean 2.2 px, Median 2.1 px, Max 3.2 px. Latency: ~80-250ms per query.

**This is remarkable.** At 2.2px mean error, the detect API is essentially pixel-perfect for our purposes. For reference, our homography has ~0.5mm/pixel sensitivity, so 2.2px ≈ 1.1mm error in robot frame — well within gripper tolerance.

### Experiment 8: Moondream 2B — Query API (free-form text)

When asked "What are the pixel coordinates..." the model returned `(0, 0)` for all blocks. When asked to describe positions, it gave vague answers like "in the middle of the table" for every block. The free-form text API has no spatial precision.

### Experiment 9: SmolVLM2-500M

**Direct coordinate requests:** Returned nonsense values — `(10, 10)` and `(100, 100)` for all blocks.

**Location descriptions:** Incorrect for all blocks — claimed green block was "in the top left corner" when it's actually bottom-left, claimed red block was "in the upper right corner" when it's center.

**Left/right classification:** 2/4 correct (chance level).

**Structured JSON skill output:** Returned identical `{"skill": "pick", "x": -0.3, "y": -0.2}` for ALL four blocks. The model outputs syntactically valid JSON but with no spatial differentiation — it's just pattern-matching the prompt format without understanding the image.

### Experiment 10: Florence-2-base (failed to run)

Florence-2's custom model code (`modeling_florence2.py`) is incompatible with transformers 4.57.6 on two fronts:
1. `_supports_sdpa` attribute missing (worked around with `attn_implementation="eager"`)
2. `prepare_inputs_for_generation` expects old-style `past_key_values` format

Would need to pin transformers to an older version (~4.40) to test Florence-2.

### Results Summary

| Model | API | Mean Error (px) | Parse Rate | Latency | Notes |
|-------|-----|----------------|------------|---------|-------|
| **Moondream 2B** | **detect** | **2.2 px** | **4/4** | **80-250ms** | **Essentially perfect** |
| Moondream 2B | point | 10.3 px | 4/4 | 60-80ms | Very good |
| Moondream 2B | query (text) | N/A | 0/4 | 110-200ms | Cannot output coordinates as text |
| SmolVLM2-500M | text gen | N/A | 0/4 | 180-830ms | No spatial understanding |
| Florence-2-base | — | — | — | — | Failed to load |

### Key Learnings

18. **Moondream 2B is the clear winner for zero-shot object localization.** Its detect API achieves 2.2px mean error — essentially perfect for our pick-and-place task. No fine-tuning needed for the perception part.

19. **Dedicated spatial APIs (point/detect) vastly outperform text generation for coordinate output.** The same Moondream 2B model returns `(0, 0)` when asked for coordinates via text, but pixel-perfect results via its built-in point/detect methods. This suggests that for our fine-tuning task, we should keep the perception (object localization) separate from the action prediction (skill parameters).

20. **SmolVLM2-500M has no spatial understanding at this scale.** It cannot even correctly classify left/right for objects in an image. The 500M parameter count is insufficient for grounded spatial reasoning. It can generate syntactically valid JSON but with meaningless coordinate values.

21. **The VLM doesn't need to learn coordinates from scratch.** Moondream 2B already knows where objects are. The fine-tuning task should be: given object positions (from detect API) + scene context + instruction → output the right skill call. This is a much simpler learning problem than training a model to jointly do perception AND action prediction.

### Revised Architecture

Based on these results, the optimal pipeline is:

```
Camera frame
    │
    ▼
Moondream 2B detect()  ──→  Object positions (pixel + bbox)
    │
    ▼
Homography transform   ──→  Robot-frame positions (x, y)
    │
    ▼
Skill predictor        ──→  {"skill": "pick", "x": ..., "y": ...}
    │
    ▼
execute_plan()         ──→  Robot motion
```

The skill predictor could be:
- **Option A:** Fine-tuned VLM (Moondream 2B with LoRA) that takes image + detected objects + instruction → skill call
- **Option B:** Smaller text-only LLM that takes detected object list + instruction → skill call (no vision needed!)
- **Option C:** Rule-based system for simple tasks (pick X → find X in detections → pick at its position)

Option C is actually viable for data collection — use rule-based picking to collect training data, then train a more capable model later. Option B is interesting because it removes the need for a VLM entirely for the action prediction part.

### Next Steps
1. ~~Integrate Moondream 2B detect() into the picking pipeline~~ — DONE (see Experiment 11)
2. Test whether it can detect blocks the color detector misses (e.g., the natural wood blocks)
3. Design the training data format — likely (image, detected_objects_json, instruction, skill_call)
4. Evaluate fine-tuning Moondream 2B with LoRA for the full image→skill pipeline

## 2026-02-20 ~late evening - Moondream Integration: 3/3 Picks via Natural Language

### Context
Integrated Moondream 2B into the picking pipeline. Built a FastAPI inference server on Spark (port 8091), a client module (`learned/moondream_client.py`), and updated the `learned_pick` MCP tool to accept either `color=` (HSV) or `query=` (Moondream) parameters.

### Architecture

```
Pi (franka-mcp)                         Spark (GPU)
┌─────────────────┐    HTTP/JSON     ┌──────────────────┐
│ learned_pick     │ ──────────────→ │ moondream_server  │
│   query="blue    │   base64 JPEG   │   model.detect()  │
│    block"        │ ←────────────── │   → bboxes        │
│                  │   JSON response  └──────────────────┘
│ homography(H)    │
│ pixel→robot      │
│ pick_at(x,y)     │
└─────────────────┘
```

- Image sent as base64 JPEG (~241KB)
- Round-trip latency: ~2.4s (includes encoding, network, GPU inference, response)
- Server loads model once at startup (19s), then serves requests

### Experiment 11: Moondream-Guided Picks

| # | Query | Detection pixel | Robot coords | Gripper width | # Detections | Result |
|---|-------|----------------|-------------|---------------|-------------|--------|
| 1 | "blue block" | (783, 514) | (0.473, 0.005) | 28.9mm | 1 | SUCCESS |
| 2 | "green block" | (490, 452) | (0.398, -0.177) | 28.4mm | 1 | SUCCESS |
| 3 | "red block on the white paper" | (607, 430) | (0.393, -0.097) | 29.5mm | 2 | SUCCESS |

**Notable:** Query #3 returned 2 red block detections (one on paper, one off). The workspace filter correctly selected the on-paper block. This demonstrates that natural language queries combined with workspace bounds provide robust disambiguation.

### Key Learnings

22. **Natural language picking works end-to-end.** Moondream 2B detect → homography → pick_at gives 3/3 success rate, identical to the color detection pipeline. But it's far more flexible — it can handle "the wooden block", "tiger toy", or any natural language description.

23. **Workspace filtering is essential for multi-instance queries.** "Red block" returns both red blocks in the scene. Filtering by workspace bounds (checking robot coordinates) picks the correct one. For queries like "the block closest to the robot", we'd need the model itself to discriminate — a future fine-tuning target.

24. **2.4s latency per detection is acceptable for scripted tasks** but too slow for a tight control loop. For autonomous data collection, this is fine — each pick-place cycle takes ~15s total anyway. For real-time servoing, would need to run Moondream on-device or use a faster model.
