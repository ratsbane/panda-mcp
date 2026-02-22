#!/usr/bin/env python3
"""
Test the VLM→detect→pick pipeline with the live robot.

Usage:
    ./venv/bin/python scripts/test_vlm_task.py "Pick up the red block."
    ./venv/bin/python scripts/test_vlm_task.py --dry-run "Pick up the red block."
"""

import argparse
import importlib.util
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", nargs="?", default="Pick up the red block.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline but don't execute robot commands")
    parser.add_argument("--url", default="http://spark:8091",
                        help="Moondream server URL")
    args = parser.parse_args()

    # Import controller
    from franka_mcp.controller import FrankaController

    # Import orchestrator
    from learned.objsel_orchestrator import run_task, OrchestratorConfig

    config = OrchestratorConfig(moondream_url=args.url)

    if args.dry_run:
        logger.info("=== DRY RUN — no robot motion ===")
        # Just test the VLM + detection pipeline
        from learned.moondream_client import detect, query
        from learned.block_detector import load_homography, pixel_to_robot
        from camera_daemon.client import CameraClient
        import random

        H, _ = load_homography()
        cam = CameraClient()
        cam.connect()
        frame = cam.get_frame()

        # Scene detection
        objects = []
        seen = set()
        for q in config.scan_queries:
            dets = detect(frame, q, server_url=config.moondream_url)
            for d in dets:
                key = (d.center_x // 30, d.center_y // 30)
                if key in seen:
                    continue
                seen.add(key)
                rx, ry = pixel_to_robot(d.center_x, d.center_y, H)
                logger.info(f"Detected {q}: pixel({d.center_x},{d.center_y}) -> robot({float(rx):.3f},{float(ry):.3f})")
                objects.append(q)

        # VLM query
        obj_str = ", ".join(sorted(set(objects)))
        question = f"You see: {obj_str}. {args.instruction}"
        logger.info(f"VLM query: {question}")
        result = query(frame, question, server_url=config.moondream_url)
        logger.info(f"VLM answer: {result.answer} ({result.latency_ms:.0f}ms)")

        from learned.objsel_orchestrator import _parse_vlm_response
        parsed = _parse_vlm_response(result.answer)
        if "object" in parsed:
            target = parsed["object"]
            logger.info(f"Would pick: {target}")
            dets = detect(frame, target, server_url=config.moondream_url)
            if dets:
                rx, ry = pixel_to_robot(dets[0].center_x, dets[0].center_y, H)
                logger.info(f"Target location: robot({float(rx):.3f},{float(ry):.3f})")

        cam.disconnect()
        return

    # Real execution
    logger.info(f"Connecting to robot...")
    controller = FrankaController()
    controller.connect()
    state = controller.get_state()
    x, y, z = state.ee_position
    logger.info(f"Robot at x={x:.3f} y={y:.3f} z={z:.3f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Running VLM task: {args.instruction}")
    logger.info(f"{'='*60}")

    result = run_task(controller, args.instruction, config)
    logger.info(f"\nResult: {json.dumps(result, indent=2)}")

    # Go home
    try:
        controller.move_cartesian_ik(x=0.4, y=0, z=0.3)
    except Exception:
        pass


if __name__ == "__main__":
    main()
