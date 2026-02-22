#!/usr/bin/env python3
"""
Autonomous NUDGE data collection loop.

Detects blocks by color, picks each one, places at random position, repeats.
NUDGE collector observes passively during each pick_at() approach.

Usage:
    # Enable NUDGE collection first via MCP: nudge_collect_enable
    # Then run this script:
    python scripts/collect_nudge_data.py --episodes 50

    # Or with specific colors:
    python scripts/collect_nudge_data.py --episodes 100 --colors red green
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Workspace bounds for random placement (safe zone)
PLACE_X_MIN = 0.35
PLACE_X_MAX = 0.55
PLACE_Y_MIN = -0.15
PLACE_Y_MAX = 0.15


def run_collection(args):
    from franka_mcp.controller import get_controller
    from camera_daemon.client import get_camera_client
    from learned.block_detector import detect_blocks, load_homography

    controller = get_controller()

    # Connect if needed
    if not controller._connected:
        result = controller.connect()
        if not result.get("success"):
            logger.error(f"Failed to connect: {result}")
            return

    # Enable NUDGE collection
    result = controller.nudge_collect_enable()
    logger.info(f"NUDGE collection: {result}")

    camera = get_camera_client()
    H, workspace = load_homography()

    successful = 0
    failed = 0

    for episode in range(1, args.episodes + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}/{args.episodes} (success={successful}, fail={failed})")
        logger.info(f"{'='*60}")

        # Detect blocks
        frame = camera.get_frame()
        if frame is None:
            logger.warning("No camera frame, waiting...")
            time.sleep(2)
            continue

        colors = args.colors if args.colors else None
        blocks = detect_blocks(frame, H, workspace, colors=colors)

        if not blocks:
            logger.info("No blocks detected, waiting...")
            time.sleep(3)
            continue

        # Pick a random block
        block = random.choice(blocks)
        logger.info(
            f"Picking {block.color} block at ({block.robot_x:.3f}, {block.robot_y:.3f}), "
            f"yaw={block.yaw:.2f}rad"
        )

        pick_result = controller.pick_at(
            x=block.robot_x,
            y=block.robot_y,
            z=0.013,
            grasp_width=0.03,
            yaw=block.yaw,
        )

        if not pick_result.get("success"):
            logger.warning(f"Pick failed: {pick_result.get('abort_reason', 'unknown')}")
            failed += 1
            # Home to reset
            controller.move_cartesian_ik(0.4, 0.0, 0.3, roll=3.14159, pitch=0.0, yaw=0.0, confirmed=True)
            time.sleep(1)
            continue

        successful += 1

        # Place at random position
        place_x = random.uniform(PLACE_X_MIN, PLACE_X_MAX)
        place_y = random.uniform(PLACE_Y_MIN, PLACE_Y_MAX)
        logger.info(f"Placing at ({place_x:.3f}, {place_y:.3f})")

        controller.place_at(x=place_x, y=place_y, z=0.015)

        # Brief pause
        time.sleep(0.5)

        # Log stats periodically
        if episode % 10 == 0:
            stats = controller.nudge_collect_stats()
            logger.info(f"NUDGE stats: {json.dumps(stats, indent=2)}")

    # Final stats
    stats = controller.nudge_collect_stats()
    logger.info(f"\nFinal NUDGE stats: {json.dumps(stats, indent=2)}")
    logger.info(f"Episodes: {successful} successful, {failed} failed out of {args.episodes}")

    # Disable collection
    controller.nudge_collect_disable()


def main():
    parser = argparse.ArgumentParser(description="Collect NUDGE training data")
    parser.add_argument("--episodes", type=int, default=50, help="Number of pick attempts")
    parser.add_argument("--colors", nargs="+", default=None,
                        help="Block colors to pick (default: all)")
    args = parser.parse_args()

    run_collection(args)


if __name__ == "__main__":
    main()
