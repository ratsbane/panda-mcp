#!/usr/bin/env python3
"""
Convert collected episodes into Moondream fine-tuning dataset.

Reads episode directories from data/moondream_training/ and produces
a JSONL dataset suitable for Moondream fine-tuning.

Each training example is:
  - image: path to camera frame
  - qa: list of (question, answer) pairs

The answer format is a JSON skill call:
  {"skill": "pick", "x": 0.398, "y": -0.178}

Usage:
    python convert_to_moondream_dataset.py [--input data/moondream_training] [--output data/moondream_dataset]
"""

import argparse
import json
import shutil
from pathlib import Path


def convert_episode(episode_dir: Path, output_dir: Path, idx: int) -> list[dict]:
    """Convert a single episode to training examples.

    Returns list of training example dicts.
    """
    examples = []

    # Check if episode was successful
    meta_path = episode_dir / "episode_meta.json"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    if not meta.get("success", False):
        return []

    # Pick example
    pick_frame = episode_dir / "pick.jpg"
    pick_meta_path = episode_dir / "pick.json"
    if pick_frame.exists() and pick_meta_path.exists():
        with open(pick_meta_path) as f:
            pick_meta = json.load(f)

        rx, ry = pick_meta["target_robot"]
        instruction = pick_meta["instruction"]

        # Copy image to output
        img_name = f"img_{idx:05d}_pick.jpg"
        shutil.copy2(pick_frame, output_dir / "images" / img_name)

        examples.append({
            "image": f"images/{img_name}",
            "qa": [{
                "question": instruction,
                "answer": json.dumps({"skill": "pick", "x": round(rx, 3), "y": round(ry, 3)}),
            }],
            "metadata": {
                "episode": meta["episode"],
                "step": "pick",
                "query": pick_meta.get("query", ""),
                "pixel": pick_meta.get("target_pixel", []),
            },
        })

    # Place example
    place_frame = episode_dir / "place.jpg"
    place_meta_path = episode_dir / "place.json"
    if place_frame.exists() and place_meta_path.exists():
        with open(place_meta_path) as f:
            place_meta = json.load(f)

        rx, ry = place_meta["target_robot"]
        instruction = place_meta["instruction"]

        img_name = f"img_{idx:05d}_place.jpg"
        shutil.copy2(place_frame, output_dir / "images" / img_name)

        examples.append({
            "image": f"images/{img_name}",
            "qa": [{
                "question": instruction,
                "answer": json.dumps({"skill": "place", "x": round(rx, 3), "y": round(ry, 3)}),
            }],
            "metadata": {
                "episode": meta["episode"],
                "step": "place",
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert episodes to Moondream dataset")
    parser.add_argument("--input", default="data/moondream_training",
                       help="Input episode directory")
    parser.add_argument("--output", default="data/moondream_dataset",
                       help="Output dataset directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    episodes = sorted(input_dir.glob("episode_*"))
    print(f"Found {len(episodes)} episodes in {input_dir}")

    all_examples = []
    successful = 0
    for i, ep_dir in enumerate(episodes):
        examples = convert_episode(ep_dir, output_dir, i)
        if examples:
            successful += 1
            all_examples.extend(examples)

    # Write JSONL
    dataset_path = output_dir / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Write summary
    summary = {
        "total_episodes": len(episodes),
        "successful_episodes": successful,
        "total_examples": len(all_examples),
        "pick_examples": sum(1 for e in all_examples if "pick" in e["qa"][0]["answer"]),
        "place_examples": sum(1 for e in all_examples if "place" in e["qa"][0]["answer"]),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset created at {output_dir}:")
    print(f"  Episodes: {successful}/{len(episodes)} successful")
    print(f"  Training examples: {len(all_examples)}")
    print(f"    Pick: {summary['pick_examples']}")
    print(f"    Place: {summary['place_examples']}")
    print(f"  Dataset file: {dataset_path}")


if __name__ == "__main__":
    main()
