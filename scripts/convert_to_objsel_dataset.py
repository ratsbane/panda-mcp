#!/usr/bin/env python3
"""
Convert collected episodes into object-selection Moondream dataset.

Instead of predicting robot coordinates, the model predicts WHICH object
to act on. The detection pipeline handles WHERE.

Pick answer format:  {"skill": "pick", "object": "red block"}
Place answer format: {"skill": "place"}
Done answer format:  {"skill": "done"}

Usage:
    python convert_to_objsel_dataset.py [--input data/moondream_training] [--output data/moondream_objsel]
"""

import argparse
import json
import random
import shutil
from pathlib import Path


# Varied instruction templates for data augmentation
PICK_TEMPLATES = [
    "Pick up the {obj}.",
    "Pick the {obj} up.",
    "Grab the {obj}.",
    "Get the {obj}.",
    "Pick up the {obj} from the table.",
]

PLACE_TEMPLATES = [
    "Put the block down.",
    "Place the block down.",
    "Set the block down.",
    "Release the block.",
    "Put it down.",
]

# Scene description template (gives model context about all visible objects)
SCENE_PROMPT_TEMPLATES = [
    "You see: {objects}. {instruction}",
    "Objects on the table: {objects}. {instruction}",
    "Visible objects: {objects}. {instruction}",
]


def convert_episode(episode_dir: Path, output_dir: Path, idx: int,
                   use_scene_context: bool = True) -> list[dict]:
    """Convert a single episode to object-selection training examples."""
    examples = []

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

        query = pick_meta.get("query", "block")
        instruction = pick_meta["instruction"]

        # Build scene context from detected objects
        ws_objects = pick_meta.get("ws_objects", [])
        if use_scene_context and ws_objects:
            obj_names = sorted(set(o["query"] for o in ws_objects))
            objects_str = ", ".join(obj_names)
            scene_template = random.choice(SCENE_PROMPT_TEMPLATES)
            question = scene_template.format(objects=objects_str, instruction=instruction)
        else:
            question = instruction

        img_name = f"img_{idx:05d}_pick.jpg"
        shutil.copy2(pick_frame, output_dir / "images" / img_name)

        examples.append({
            "image": f"images/{img_name}",
            "qa": [{
                "question": question,
                "answer": json.dumps({"skill": "pick", "object": query}),
            }],
            "metadata": {
                "episode": meta["episode"],
                "step": "pick",
                "original_instruction": instruction,
                "target_query": query,
            },
        })

    # Place example
    place_frame = episode_dir / "place.jpg"
    place_meta_path = episode_dir / "place.json"
    if place_frame.exists() and place_meta_path.exists():
        with open(place_meta_path) as f:
            place_meta = json.load(f)

        instruction = place_meta["instruction"]

        img_name = f"img_{idx:05d}_place.jpg"
        shutil.copy2(place_frame, output_dir / "images" / img_name)

        examples.append({
            "image": f"images/{img_name}",
            "qa": [{
                "question": instruction,
                "answer": json.dumps({"skill": "place"}),
            }],
            "metadata": {
                "episode": meta["episode"],
                "step": "place",
            },
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert episodes to object-selection dataset")
    parser.add_argument("--input", default="data/moondream_training",
                       help="Input episode directory")
    parser.add_argument("--output", default="data/moondream_objsel",
                       help="Output dataset directory")
    parser.add_argument("--no-scene-context", action="store_true",
                       help="Don't include scene object list in prompts")
    args = parser.parse_args()

    random.seed(42)  # Reproducible template selection

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Clean output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    episodes = sorted(input_dir.glob("episode_*"))
    print(f"Found {len(episodes)} episodes in {input_dir}")

    all_examples = []
    successful = 0
    for i, ep_dir in enumerate(episodes):
        examples = convert_episode(ep_dir, output_dir, i,
                                  use_scene_context=not args.no_scene_context)
        if examples:
            successful += 1
            all_examples.extend(examples)

    # Write JSONL
    dataset_path = output_dir / "dataset.jsonl"
    with open(dataset_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Analyze object distribution
    objects_seen = {}
    for ex in all_examples:
        answer = json.loads(ex["qa"][0]["answer"])
        if "object" in answer:
            obj = answer["object"]
            objects_seen[obj] = objects_seen.get(obj, 0) + 1

    summary = {
        "total_episodes": len(episodes),
        "successful_episodes": successful,
        "total_examples": len(all_examples),
        "pick_examples": sum(1 for e in all_examples if '"pick"' in e["qa"][0]["answer"]),
        "place_examples": sum(1 for e in all_examples if '"place"' in e["qa"][0]["answer"]),
        "object_distribution": objects_seen,
        "scene_context": not args.no_scene_context,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset created at {output_dir}:")
    print(f"  Episodes: {successful}/{len(episodes)} successful")
    print(f"  Training examples: {len(all_examples)}")
    print(f"    Pick: {summary['pick_examples']}")
    print(f"    Place: {summary['place_examples']}")
    print(f"  Object distribution: {objects_seen}")
    print(f"  Scene context: {not args.no_scene_context}")
    print(f"  Dataset file: {dataset_path}")

    # Show a few examples
    print("\n=== Sample examples ===")
    for ex in all_examples[:3]:
        print(f"  Q: {ex['qa'][0]['question']}")
        print(f"  A: {ex['qa'][0]['answer']}")
        print()


if __name__ == "__main__":
    main()
