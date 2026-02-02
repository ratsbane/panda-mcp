#!/usr/bin/env python3
"""
Train multiple backbone architectures and compare results.

This script trains all specified backbones sequentially and generates
a comparison report at the end.

Usage:
    python scripts/train_all_backbones.py --data ~/datasets/gripper_v1 --epochs 100
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BACKBONES = [
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "efficientnet_b2",
    # "efficientnet_b4",  # Uncomment for larger model
    # "convnext_tiny",    # Uncomment for best accuracy
]


def train_backbone(backbone: str, data_dir: str, epochs: int, output_dir: str, batch_size: int):
    """Train a single backbone."""
    cmd = [
        sys.executable,
        "scripts/train_gripper_localizer.py",
        "--data", data_dir,
        "--backbone", backbone,
        "--epochs", str(epochs),
        "--output", output_dir,
        "--batch-size", str(batch_size),
        "--mixed-precision",
    ]

    print(f"\n{'='*60}")
    print(f"Training {backbone}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def find_best_checkpoint(output_dir: str, backbone: str) -> dict:
    """Find the best checkpoint for a backbone and extract metrics."""
    output_path = Path(output_dir)

    # Find the run directory for this backbone
    run_dirs = list(output_path.glob(f"{backbone}_*"))
    if not run_dirs:
        return None

    # Get the most recent run
    run_dir = sorted(run_dirs)[-1]
    best_checkpoint = run_dir / "checkpoint_best.pt"

    if not best_checkpoint.exists():
        return None

    # Load checkpoint to get metrics
    import torch
    checkpoint = torch.load(best_checkpoint, map_location="cpu")

    return {
        "backbone": backbone,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "config": checkpoint.get("config", {}),
    }


def generate_report(results: list, output_dir: str):
    """Generate a comparison report."""
    report_path = Path(output_dir) / f"comparison_report_{datetime.now():%Y%m%d_%H%M%S}.md"

    with open(report_path, "w") as f:
        f.write("# Gripper Localizer Backbone Comparison\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        f.write("## Results\n\n")
        f.write("| Backbone | Mean Error (mm) | Median Error (mm) | P90 Error (mm) | Val Loss |\n")
        f.write("|----------|-----------------|-------------------|----------------|----------|\n")

        # Sort by mean error
        results = sorted(results, key=lambda x: x["val_metrics"].get("mean_error_mm", float("inf")))

        for r in results:
            metrics = r["val_metrics"]
            f.write(f"| {r['backbone']} | "
                    f"{metrics.get('mean_error_mm', 'N/A'):.2f} | "
                    f"{metrics.get('median_error_mm', 'N/A'):.2f} | "
                    f"{metrics.get('p90_error_mm', 'N/A'):.2f} | "
                    f"{metrics.get('loss', 'N/A'):.6f} |\n")

        f.write("\n## Best Model\n\n")
        best = results[0]
        f.write(f"**{best['backbone']}** with mean error of {best['val_metrics']['mean_error_mm']:.2f}mm\n\n")
        f.write(f"Checkpoint: `{best['best_checkpoint']}`\n")

        f.write("\n## Run Directories\n\n")
        for r in results:
            f.write(f"- {r['backbone']}: `{r['run_dir']}`\n")

    print(f"\nComparison report saved to {report_path}")

    # Also save as JSON
    json_path = report_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Train and compare multiple backbones")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per backbone")
    parser.add_argument("--output", type=str, default="./runs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbones", type=str, nargs="+", default=BACKBONES,
                        help="Backbones to train")
    args = parser.parse_args()

    print(f"Training {len(args.backbones)} backbones: {args.backbones}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")

    # Train each backbone
    successful = []
    failed = []

    for backbone in args.backbones:
        success = train_backbone(
            backbone, args.data, args.epochs, args.output, args.batch_size
        )
        if success:
            successful.append(backbone)
        else:
            failed.append(backbone)

    # Collect results
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    results = []
    for backbone in successful:
        result = find_best_checkpoint(args.output, backbone)
        if result:
            results.append(result)
            print(f"{backbone}: {result['val_metrics'].get('mean_error_mm', 'N/A'):.2f}mm")

    if failed:
        print(f"\nFailed: {failed}")

    # Generate comparison report
    if results:
        generate_report(results, args.output)


if __name__ == "__main__":
    main()
