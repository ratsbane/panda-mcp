#!/usr/bin/env python3
"""
Visualize collected training data.

Shows images with overlaid robot state information.
Useful for verifying data quality and understanding the dataset.
"""

import sys
import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_session(session_dir: Path) -> tuple[dict, list[dict]]:
    """Load session metadata and samples."""
    with open(session_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(session_dir / "samples.json") as f:
        samples = json.load(f)

    return metadata, samples


def draw_sample_info(image: np.ndarray, sample: dict) -> np.ndarray:
    """Draw robot state info on image."""
    result = image.copy()

    # Extract data
    rs = sample["robot_state"]
    ee_pos = rs["end_effector_position"]
    gripper = rs["gripper_width"]

    # Semi-transparent overlay
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (400, 120), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

    # Position
    pos_text = f"Pos: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
    cv2.putText(result, pos_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Gripper
    grip_text = f"Gripper: {gripper*1000:.1f}mm"
    cv2.putText(result, grip_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Camera pose if available
    if sample.get("camera_pose"):
        cp = sample["camera_pose"]
        conf = cp.get("confidence", 0)
        cam_text = f"Camera conf: {conf:.2f}"
        cv2.putText(result, cam_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Sample info
    sample_text = f"Sample {sample['sample_id']} / Session: {sample['session_id'][:8]}"
    cv2.putText(result, sample_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return result


def interactive_viewer(session_dir: Path):
    """Interactive viewer for browsing samples."""
    metadata, samples = load_session(session_dir)

    print(f"Session: {metadata['session_id']}")
    print(f"Samples: {len(samples)}")
    print("\nControls:")
    print("  Left/Right: Navigate samples")
    print("  Space: Play/pause slideshow")
    print("  q: Quit")

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)

    idx = 0
    playing = False

    while True:
        sample = samples[idx]
        image_path = session_dir / sample["image_path"]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Could not load {image_path}")
            idx = (idx + 1) % len(samples)
            continue

        display = draw_sample_info(image, sample)

        # Navigation bar at bottom
        bar_y = display.shape[0] - 20
        progress = idx / max(1, len(samples) - 1)
        bar_width = int(display.shape[1] * progress)
        cv2.rectangle(display, (0, bar_y), (bar_width, display.shape[0]), (0, 255, 0), -1)

        cv2.imshow("Dataset Viewer", display)

        delay = 100 if playing else 0
        key = cv2.waitKey(delay if delay > 0 else 1) & 0xFF

        if key == ord('q'):
            break
        elif key == 81 or key == ord('a'):  # Left arrow
            idx = (idx - 1) % len(samples)
            playing = False
        elif key == 83 or key == ord('d'):  # Right arrow
            idx = (idx + 1) % len(samples)
            playing = False
        elif key == ord(' '):
            playing = not playing
        elif playing:
            idx = (idx + 1) % len(samples)

    cv2.destroyAllWindows()


def generate_statistics(session_dir: Path):
    """Generate statistics about the dataset."""
    metadata, samples = load_session(session_dir)

    print(f"\n=== Dataset Statistics ===")
    print(f"Session: {metadata['session_id']}")
    print(f"Total samples: {len(samples)}")
    print(f"Start time: {metadata.get('start_time', 'N/A')}")
    print(f"End time: {metadata.get('end_time', 'N/A')}")

    if not samples:
        return

    # Position statistics
    positions = [s["robot_state"]["end_effector_position"] for s in samples]
    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]

    print(f"\nEnd-effector position ranges:")
    print(f"  X: [{min(x_vals):.3f}, {max(x_vals):.3f}]")
    print(f"  Y: [{min(y_vals):.3f}, {max(y_vals):.3f}]")
    print(f"  Z: [{min(z_vals):.3f}, {max(z_vals):.3f}]")

    # Gripper statistics
    grippers = [s["robot_state"]["gripper_width"] for s in samples]
    print(f"\nGripper width range: [{min(grippers)*1000:.1f}, {max(grippers)*1000:.1f}]mm")

    # Camera pose statistics
    cam_samples = [s for s in samples if s.get("camera_pose") and s["camera_pose"].get("confidence", 0) > 0]
    print(f"\nSamples with camera pose: {len(cam_samples)} ({100*len(cam_samples)/len(samples):.1f}%)")

    if cam_samples:
        confidences = [s["camera_pose"]["confidence"] for s in cam_samples]
        print(f"Camera pose confidence: [{min(confidences):.2f}, {max(confidences):.2f}]")


def plot_workspace_coverage(session_dir: Path, output_path: str = None):
    """Plot 3D scatter of end-effector positions."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available for plotting")
        return

    metadata, samples = load_session(session_dir)

    positions = [s["robot_state"]["end_effector_position"] for s in samples]
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    z = [p[2] for p in positions]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=range(len(x)), cmap='viridis', alpha=0.6)

    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left/right)')
    ax.set_zlabel('Z (height)')
    ax.set_title(f'Workspace Coverage - {len(samples)} samples')

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize training dataset")

    parser.add_argument("session_dir", type=str, help="Path to session directory")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--plot", action="store_true", help="Plot workspace coverage")
    parser.add_argument("--output", type=str, help="Output path for plot")

    args = parser.parse_args()

    session_dir = Path(args.session_dir)

    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}")
        sys.exit(1)

    if args.stats:
        generate_statistics(session_dir)
    elif args.plot:
        plot_workspace_coverage(session_dir, args.output)
    else:
        interactive_viewer(session_dir)


if __name__ == "__main__":
    main()
