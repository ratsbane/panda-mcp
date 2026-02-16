#!/usr/bin/env python3
"""
Convert TrajectoryRecorder episodes to LeRobot v3.0 dataset format.

Reads data/recordings/episode_*/episode.json + images/
Outputs LeRobot-compatible dataset with Parquet + images.

Usage:
    python scripts/convert_to_lerobot.py [--input data/recordings] [--output data/lerobot_v1]

Features:
    observation.state: float32 (8,) = [j1, j2, j3, j4, j5, j6, j7, gripper_width]
    action: float32 (8,) = [dj1, dj2, dj3, dj4, dj5, dj6, dj7, dgripper]
    observation.images.rgb: image (H, W, 3)
    task: string (language instruction)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")


def load_raw_episode(episode_dir: Path) -> dict:
    """Load a raw episode from TrajectoryRecorder format."""
    with open(episode_dir / "episode.json") as f:
        return json.load(f)


def compute_actions(frames: list[dict]) -> list[np.ndarray]:
    """Compute per-frame actions as delta to next frame's state.

    action[t] = state[t+1] - state[t] for t < N-1
    action[N-1] = zeros (last frame has no next state)
    """
    actions = []
    for i in range(len(frames)):
        if i < len(frames) - 1:
            curr_joints = np.array(frames[i]["joint_positions"], dtype=np.float32)
            next_joints = np.array(frames[i + 1]["joint_positions"], dtype=np.float32)
            curr_gripper = frames[i]["gripper_width"]
            next_gripper = frames[i + 1]["gripper_width"]

            delta_joints = next_joints - curr_joints
            delta_gripper = next_gripper - curr_gripper
            action = np.concatenate([delta_joints, [delta_gripper]]).astype(np.float32)
        else:
            action = np.zeros(8, dtype=np.float32)

        actions.append(action)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Convert recordings to LeRobot format")
    parser.add_argument("--input", type=str, default="data/recordings",
                        help="Input directory with TrajectoryRecorder episodes")
    parser.add_argument("--output", type=str, default="data/lerobot_v1",
                        help="Output directory for LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="panda-mcp/pick-and-place",
                        help="HuggingFace-style repo ID")
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    parser.add_argument("--successful-only", action="store_true",
                        help="Only include successful episodes")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Find episodes
    episode_dirs = sorted(input_dir.glob("episode_*"))
    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        return 1

    print(f"Found {len(episode_dirs)} episodes in {input_dir}")

    # Load and filter episodes
    episodes = []
    for ep_dir in episode_dirs:
        ep_json = ep_dir / "episode.json"
        if not ep_json.exists():
            print(f"  Skipping {ep_dir.name}: no episode.json")
            continue

        ep_data = load_raw_episode(ep_dir)

        if args.successful_only and not ep_data.get("success", False):
            print(f"  Skipping {ep_dir.name}: not successful")
            continue

        if len(ep_data.get("frames", [])) < 2:
            print(f"  Skipping {ep_dir.name}: too few frames ({len(ep_data.get('frames', []))})")
            continue

        episodes.append((ep_dir, ep_data))
        status = "OK" if ep_data.get("success") else "FAIL"
        print(f"  {ep_dir.name}: {len(ep_data['frames'])} frames, "
              f"'{ep_data.get('language_instruction', '?')}' [{status}]")

    if not episodes:
        print("No valid episodes to convert")
        return 1

    # Determine image dimensions from first episode with images
    img_shape = None
    for ep_dir, ep_data in episodes:
        for frame in ep_data["frames"]:
            if frame.get("image_path"):
                img_path = ep_dir / frame["image_path"]
                if img_path.exists():
                    from PIL import Image
                    img = Image.open(img_path)
                    img_shape = (img.height, img.width, 3)
                    print(f"\nImage dimensions: {img.width}x{img.height}")
                    break
        if img_shape:
            break

    if img_shape is None:
        print("\nWARNING: No images found in any episode. Creating state-only dataset.")

    # Create LeRobot dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["channels"],
        },
    }

    if img_shape is not None:
        features["observation.images.rgb"] = {
            "dtype": "image",
            "shape": img_shape,
            "names": ["height", "width", "channels"],
        }

    # Remove output dir if it exists (fresh conversion)
    if output_dir.exists():
        import shutil
        print(f"\nRemoving existing output at {output_dir}")
        shutil.rmtree(output_dir)

    print(f"\nCreating LeRobot dataset at {output_dir}...")
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=str(output_dir),
        robot_type="franka",
        use_videos=False,
        features=features,
    )

    # Convert each episode
    total_frames = 0
    for ep_idx, (ep_dir, ep_data) in enumerate(episodes):
        frames = ep_data["frames"]
        instruction = ep_data.get("language_instruction", "manipulation task")

        # Compute actions
        actions = compute_actions(frames)

        # Create episode buffer
        ds.create_episode_buffer()

        for i, frame in enumerate(frames):
            # Build observation state: [j1..j7, gripper_width]
            joints = np.array(frame["joint_positions"], dtype=np.float32)
            gripper = np.float32(frame["gripper_width"])
            state = np.concatenate([joints, [gripper]])

            frame_dict = {
                "observation.state": state,
                "action": actions[i],
                "task": instruction,
            }

            # Load image if available
            if img_shape is not None and frame.get("image_path"):
                img_path = ep_dir / frame["image_path"]
                if img_path.exists():
                    from PIL import Image
                    img = Image.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    frame_dict["observation.images.rgb"] = img
                else:
                    # Create black placeholder
                    from PIL import Image
                    frame_dict["observation.images.rgb"] = Image.new(
                        "RGB", (img_shape[1], img_shape[0]))

            ds.add_frame(frame_dict)

        ds.save_episode()
        total_frames += len(frames)
        print(f"  Converted episode {ep_idx} ({ep_dir.name}): "
              f"{len(frames)} frames, task='{instruction}'")

    # Finalize
    print("\nFinalizing dataset...")
    ds.finalize()

    print(f"\n=== Conversion Complete ===")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_dir}")
    print(f"  Repo ID: {args.repo_id}")

    # Print dataset info
    info_path = output_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n  LeRobot info:")
        print(f"    fps: {info.get('fps')}")
        print(f"    robot_type: {info.get('robot_type')}")
        print(f"    total_episodes: {info.get('total_episodes')}")
        print(f"    total_frames: {info.get('total_frames')}")

    print(f"\nTo push to HuggingFace Hub:")
    print(f"  python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
          f"ds = LeRobotDataset('{args.repo_id}', root='{output_dir}'); "
          f"ds.push_to_hub()\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
