#!/usr/bin/env python3
"""
Calibration sweep: move gripper across a grid, photograph, embed, store.

Builds the spatial memory database that maps visual appearance to robot
coordinates. Each grid position stores:
  - DINOv2 embedding of a crop around the gripper's pixel location
  - DINOv2 embedding of the full scene
  - Robot (x, y, z) coordinates
  - Gripper pixel (px, py) coordinates

Usage from MCP server:
    from learned.calibration_sweep import run_sweep
    results = run_sweep(controller, camera_client)

Usage standalone:
    python -m learned.calibration_sweep --no-robot  # dry run, show grid
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Default grid spanning the workspace
DEFAULT_GRID = {
    "x_values": [0.35, 0.40, 0.45, 0.50, 0.55],
    "y_values": [-0.10, -0.03, 0.04, 0.11, 0.18],
    "z_sweep": 0.025,   # table height for calibration
    "z_travel": 0.15,   # safe travel height between positions
}


def run_sweep(controller, camera_client, db=None, grid=None,
              detect_gripper_fn=None, embed=True):
    """
    Execute calibration sweep across the workspace.

    Moves gripper to a grid of positions at table height, photographs each,
    and records (robot_position, frame) pairs. Optionally embeds via DINOv2.

    All data is saved to data/sweep_frames/ and data/sweep_points.json
    regardless of embedding, so it can be embedded later.

    Args:
        controller: FrankaController instance (connected)
        camera_client: CameraClient for capturing frames
        db: SpatialDB instance (connected). Only needed if embed=True.
        grid: Dict with x_values, y_values, z_sweep, z_travel
        detect_gripper_fn: Optional callable(frame) -> (px, py) to detect
            gripper pixel position. If None, stores None.
        embed: Whether to compute DINOv2 embeddings (requires Spark)

    Returns:
        dict with sweep results
    """
    if grid is None:
        grid = DEFAULT_GRID

    if embed and db is None:
        from learned.embedding_db import SpatialDB
        db = SpatialDB()
        db.connect()

    x_vals = grid["x_values"]
    y_vals = grid["y_values"]
    z_sweep = grid["z_sweep"]
    z_travel = grid["z_travel"]
    total = len(x_vals) * len(y_vals)

    logger.info(f"Starting calibration sweep: {len(x_vals)}x{len(y_vals)} = "
                f"{total} positions at z={z_sweep} (embed={embed})")

    # First, move to travel height
    controller.move_cartesian_ik(x=x_vals[0], y=y_vals[0], z=z_travel,
                                 confirmed=True)
    time.sleep(0.3)

    # Persistent storage for points (survives crashes, can embed later)
    data_dir = Path(__file__).parent.parent / "data"
    frame_dir = data_dir / "sweep_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    points_file = data_dir / "sweep_points.json"

    # Load existing points if any
    existing_points = []
    if points_file.exists():
        with open(points_file) as f:
            existing_points = json.load(f)

    results = {
        "grid": grid,
        "points": [],
        "errors": [],
        "start_time": time.time(),
    }

    point_idx = 0
    for xi, x in enumerate(x_vals):
        for yi, y in enumerate(y_vals):
            point_idx += 1
            logger.info(f"[{point_idx}/{total}] Moving to ({x:.3f}, {y:.3f}, {z_sweep})")

            # Move to travel height above target, then lower
            move_result = controller.move_cartesian_ik(
                x=x, y=y, z=z_travel, confirmed=True)
            if not move_result.get("success"):
                logger.warning(f"Failed to move to ({x}, {y}, {z_travel}): {move_result}")
                results["errors"].append({
                    "x": x, "y": y, "phase": "travel",
                    "error": str(move_result)})
                continue

            time.sleep(0.1)

            # Lower to sweep height
            move_result = controller.move_cartesian_ik(
                x=x, y=y, z=z_sweep, confirmed=True)
            if not move_result.get("success"):
                logger.warning(f"Failed to lower to z={z_sweep}: {move_result}")
                results["errors"].append({
                    "x": x, "y": y, "phase": "lower",
                    "error": str(move_result)})
                controller.move_cartesian_ik(x=x, y=y, z=z_travel, confirmed=True)
                continue

            time.sleep(0.3)  # settle

            # Get actual robot position
            state = controller.get_state()
            actual_x = state.ee_position[0]
            actual_y = state.ee_position[1]
            actual_z = state.ee_position[2]

            # Capture frame
            frame = camera_client.get_frame()
            if frame is None:
                logger.warning(f"Failed to capture frame at ({x}, {y})")
                results["errors"].append({
                    "x": x, "y": y, "phase": "capture",
                    "error": "no frame"})
                controller.move_cartesian_ik(x=x, y=y, z=z_travel, confirmed=True)
                continue

            # Detect gripper pixel position
            if detect_gripper_fn:
                gripper_px = detect_gripper_fn(frame)
            else:
                gripper_px = None

            # Save frame
            frame_path = frame_dir / f"sweep_{point_idx:03d}_x{x:.2f}_y{y:.2f}.jpg"
            cv2.imwrite(str(frame_path), frame)

            # Build point record
            point_record = {
                "commanded": {"x": x, "y": y, "z": z_sweep},
                "actual": {"x": round(actual_x, 4),
                           "y": round(actual_y, 4),
                           "z": round(actual_z, 4)},
                "gripper_px": gripper_px,
                "grid_ix": xi,
                "grid_iy": yi,
                "frame_path": str(frame_path),
                "timestamp": time.time(),
                "embedded": False,
            }

            # Optionally embed
            embed_time = None
            if embed and db is not None:
                try:
                    t0 = time.time()
                    full_embedding = db.embed_frame(frame)
                    embed_time = time.time() - t0
                    db.add_point(
                        full_embedding,
                        robot_xyz=(actual_x, actual_y, actual_z),
                        pixel_xy=gripper_px,
                        metadata=point_record,
                    )
                    point_record["embedded"] = True
                    point_record["embed_time_ms"] = round(embed_time * 1000)
                except Exception as e:
                    logger.warning(f"Embedding failed at ({x}, {y}): {e}")
                    point_record["embed_error"] = str(e)

            # Save to persistent JSON (append + write atomically)
            existing_points.append(point_record)
            tmp_path = str(points_file) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(existing_points, f, indent=2)
            Path(tmp_path).rename(points_file)

            results["points"].append({
                "commanded": (x, y, z_sweep),
                "actual": (round(actual_x, 4), round(actual_y, 4),
                           round(actual_z, 4)),
                "gripper_px": gripper_px,
                "embedded": point_record["embedded"],
            })

            logger.info(f"  [{point_idx}/{total}] actual=({actual_x:.3f}, "
                        f"{actual_y:.3f}, {actual_z:.3f}), "
                        f"frame={frame_path.name}"
                        + (f", embed={embed_time*1000:.0f}ms" if embed_time else ""))

            # Raise back to travel height
            controller.move_cartesian_ik(x=x, y=y, z=z_travel, confirmed=True)
            time.sleep(0.1)

    results["end_time"] = time.time()
    results["duration_s"] = round(results["end_time"] - results["start_time"], 1)
    results["total_points"] = len(results["points"])
    results["total_errors"] = len(results["errors"])
    results["points_file"] = str(points_file)

    logger.info(f"Sweep complete: {results['total_points']} points stored in "
                f"{points_file}, {results['total_errors']} errors, "
                f"{results['duration_s']}s total")

    return results


def query_location(db, frame, target_pixel=None):
    """
    Given a camera frame (and optional detected target pixel position),
    find the most likely robot coordinates using the spatial database.

    Uses full-scene embedding similarity + optional pixel-distance weighting.
    """
    # Embed the query scene
    query_emb = db.embed_frame(frame)

    # Get all stored points
    ids, all_embs, points = db.get_all_embeddings()
    if len(points) == 0:
        return None

    # Compute embedding similarities
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    emb_norms = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
    similarities = emb_norms @ query_norm

    # If we have a target pixel position and stored pixel positions,
    # combine embedding similarity with pixel proximity
    if target_pixel is not None:
        tx, ty = target_pixel
        pixel_distances = []
        for p in points:
            if p.pixel_x is not None and p.pixel_y is not None:
                dist = np.sqrt((p.pixel_x - tx)**2 + (p.pixel_y - ty)**2)
            else:
                dist = 1000  # large default
            pixel_distances.append(dist)

        pixel_distances = np.array(pixel_distances)
        # Convert distances to weights (closer = higher weight)
        pixel_weights = np.exp(-pixel_distances / 200)  # 200px = scale factor
        # Combine: 50% embedding similarity + 50% pixel proximity
        combined = 0.5 * similarities + 0.5 * pixel_weights
    else:
        combined = similarities

    # Weighted average of top-k
    k = min(5, len(points))
    top_k = np.argsort(combined)[::-1][:k]

    weights = []
    coords = []
    for i in top_k:
        w = max(combined[i], 0.0) ** 2
        weights.append(w)
        coords.append((points[i].robot_x, points[i].robot_y, points[i].robot_z))

    total_w = sum(weights)
    if total_w < 1e-8:
        p = points[top_k[0]]
        return (p.robot_x, p.robot_y, p.robot_z, 0.0)

    rx = sum(w * c[0] for w, c in zip(weights, coords)) / total_w
    ry = sum(w * c[1] for w, c in zip(weights, coords)) / total_w
    rz = sum(w * c[2] for w, c in zip(weights, coords)) / total_w
    confidence = float(combined[top_k[0]])

    return (rx, ry, rz, confidence)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    parser = argparse.ArgumentParser(description="Calibration sweep")
    parser.add_argument("--no-robot", action="store_true",
                        help="Dry run: just show the grid")
    parser.add_argument("--nx", type=int, default=5, help="Grid X points")
    parser.add_argument("--ny", type=int, default=5, help="Grid Y points")
    args = parser.parse_args()

    grid = DEFAULT_GRID.copy()
    if args.nx != 5 or args.ny != 5:
        grid["x_values"] = np.linspace(0.35, 0.55, args.nx).tolist()
        grid["y_values"] = np.linspace(-0.10, 0.18, args.ny).tolist()

    if args.no_robot:
        print("Grid positions:")
        for x in grid["x_values"]:
            for y in grid["y_values"]:
                print(f"  ({x:.3f}, {y:.3f})")
        print(f"Total: {len(grid['x_values']) * len(grid['y_values'])} positions")
    else:
        print("Use run_sweep(controller, camera_client) from MCP server")
