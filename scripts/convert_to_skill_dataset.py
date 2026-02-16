#!/usr/bin/env python3
"""
Convert skill episodes to VLM fine-tuning dataset (JSONL).

Reads data/skill_episodes/ and produces a JSONL file where each line
is a training example with an image and a conversation:

  User: <image>\nTask: ...\nWhat is the next action?
  Assistant: {"skill": "pick", "x": 0.42, "y": -0.07, "grasp_width": 0.03}

Usage:
    python scripts/convert_to_skill_dataset.py
    python scripts/convert_to_skill_dataset.py --input data/skill_episodes --output data/skill_vlm
    python scripts/convert_to_skill_dataset.py --successful-only
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


# Prompt variations for task augmentation
PROMPT_TEMPLATES = [
    "Task: {task}\nWhat is the next action?",
    "Task: {task}\nWhat should the robot do next?",
    "Task: {task}\nPredict the next skill to execute.",
    "Task: {task}\nWhat action should be performed?",
]


def convert_episode(episode_dir: Path, successful_only: bool = True) -> list[dict]:
    """
    Convert a single episode to training examples.

    Returns list of dicts, each with 'image' and 'conversations' keys.
    """
    episode_json = episode_dir / "episode.json"
    if not episode_json.exists():
        return []

    with open(episode_json) as f:
        episode = json.load(f)

    if successful_only and not episode.get("success", False):
        return []

    task = episode["task"]
    examples = []

    for step in episode.get("steps", []):
        # Check image exists
        image_rel = step.get("image_path", "")
        image_abs = episode_dir / image_rel
        if not image_abs.exists() or not step.get("has_image", True):
            continue

        skill = step["skill"]
        params = step.get("params", {})

        # Build the skill output JSON
        skill_output = {"skill": skill}
        if params:
            skill_output.update(params)

        # Round numeric params for cleaner training data
        for k, v in skill_output.items():
            if isinstance(v, float):
                skill_output[k] = round(v, 4)

        # Use primary prompt template (index 0) — augmentation at training time
        prompt = PROMPT_TEMPLATES[0].format(task=task)

        # Image path relative to output dataset root (will be episode_NNNN/images/step_NNN.jpg)
        image_relative = f"{episode_dir.name}/{image_rel}"

        example = {
            "image": image_relative,
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\n{prompt}",
                },
                {
                    "from": "assistant",
                    "value": json.dumps(skill_output, separators=(",", ":")),
                },
            ],
        }
        examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert skill episodes to VLM training JSONL")
    parser.add_argument("--input", default="./data/skill_episodes",
                        help="Input skill episodes directory")
    parser.add_argument("--output", default="./data/skill_vlm",
                        help="Output directory for JSONL dataset")
    parser.add_argument("--successful-only", action="store_true", default=True,
                        help="Only include successful episodes (default: True)")
    parser.add_argument("--include-failures", action="store_true",
                        help="Include failed episodes too")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    successful_only = not args.include_failures

    if not input_dir.exists():
        print(f"Error: input directory {input_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all episodes
    episode_dirs = sorted(input_dir.glob("episode_*"))
    if not episode_dirs:
        print(f"No episodes found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(episode_dirs)} episodes in {input_dir}")

    # Convert
    all_examples = []
    skill_counts = Counter()
    episodes_used = 0

    for ep_dir in episode_dirs:
        examples = convert_episode(ep_dir, successful_only=successful_only)
        if examples:
            all_examples.extend(examples)
            episodes_used += 1
            for ex in examples:
                # Parse skill from assistant response
                assistant_msg = ex["conversations"][1]["value"]
                try:
                    skill_data = json.loads(assistant_msg)
                    skill_counts[skill_data.get("skill", "unknown")] += 1
                except json.JSONDecodeError:
                    skill_counts["parse_error"] += 1

    if not all_examples:
        print("No training examples generated.")
        if successful_only:
            print("Try --include-failures to include failed episodes.")
        sys.exit(1)

    # Write JSONL
    jsonl_path = output_dir / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    # Write metadata
    meta = {
        "total_examples": len(all_examples),
        "episodes_used": episodes_used,
        "episodes_total": len(episode_dirs),
        "successful_only": successful_only,
        "skill_distribution": dict(skill_counts),
        "prompt_template": PROMPT_TEMPLATES[0],
        "source_dir": str(input_dir),
    }
    meta_path = output_dir / "dataset_info.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print(f"\nDataset written to {output_dir}/")
    print(f"  Episodes used: {episodes_used}/{len(episode_dirs)}")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Skill distribution:")
    for skill, count in sorted(skill_counts.items()):
        print(f"    {skill}: {count}")
    print(f"\n  JSONL: {jsonl_path}")
    print(f"  Metadata: {meta_path}")

    # Validate: check a sample
    print(f"\nSample example:")
    sample = all_examples[0]
    print(f"  Image: {sample['image']}")
    print(f"  User: {sample['conversations'][0]['value'][:80]}...")
    print(f"  Assistant: {sample['conversations'][1]['value']}")


if __name__ == "__main__":
    main()
