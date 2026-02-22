#!/usr/bin/env python3
"""
Re-label existing NUDGE approach data with per-axis exponential thresholds.

Reads each approach's labels.json, recomputes dx/dy/dz_class from the stored
continuous offsets (dx_m, dy_m, dz_m) using the new per-axis DiscretizeConfig,
and overwrites labels.json in place.

Usage:
    python scripts/relabel_nudge_data.py
    python scripts/relabel_nudge_data.py --data-dir data/nudge_approaches --dry-run
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.nudge.discretize import continuous_to_class, class_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def relabel(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    approaches = sorted(data_dir.glob("approach_*"))
    logger.info(f"Found {len(approaches)} approaches in {data_dir}")

    total_frames = 0
    changed_frames = 0
    cx_old, cy_old, cz_old = Counter(), Counter(), Counter()
    cx_new, cy_new, cz_new = Counter(), Counter(), Counter()

    for approach_dir in approaches:
        labels_path = approach_dir / "labels.json"
        if not labels_path.exists():
            continue

        with open(labels_path) as f:
            labels = json.load(f)

        modified = False
        for lbl in labels:
            total_frames += 1

            old_dx = lbl["dx_class"]
            old_dy = lbl["dy_class"]
            old_dz = lbl["dz_class"]

            new_dx = continuous_to_class(lbl["dx_m"], axis="x")
            new_dy = continuous_to_class(lbl["dy_m"], axis="y")
            new_dz = continuous_to_class(lbl["dz_m"], axis="z")

            cx_old[old_dx] += 1
            cy_old[old_dy] += 1
            cz_old[old_dz] += 1
            cx_new[new_dx] += 1
            cy_new[new_dy] += 1
            cz_new[new_dz] += 1

            if new_dx != old_dx or new_dy != old_dy or new_dz != old_dz:
                changed_frames += 1
                modified = True
                lbl["dx_class"] = new_dx
                lbl["dy_class"] = new_dy
                lbl["dz_class"] = new_dz

        if modified and not args.dry_run:
            with open(labels_path, "w") as f:
                json.dump(labels, f, indent=2)

    logger.info(f"Total frames: {total_frames}, changed: {changed_frames}")

    def show_dist(name, old, new):
        print(f"\n  {name}:")
        print(f"    {'Class':>6s}  {'Label':>8s}  {'Old':>5s}  {'New':>5s}  {'Delta':>6s}")
        for i in range(7):
            o = old.get(i, 0)
            n = new.get(i, 0)
            d = n - o
            label = class_label(i)
            delta_str = f"{d:+d}" if d != 0 else ""
            print(f"    {i:>6d}  {label:>8s}  {o:>5d}  {n:>5d}  {delta_str:>6s}")

    show_dist("dx", cx_old, cx_new)
    show_dist("dy", cy_old, cy_new)
    show_dist("dz", cz_old, cz_new)

    if args.dry_run:
        logger.info("DRY RUN — no files modified")
    else:
        logger.info("Labels updated in place")


def main():
    parser = argparse.ArgumentParser(description="Re-label NUDGE data with per-axis thresholds")
    parser.add_argument("--data-dir", default="data/nudge_approaches")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()
    relabel(args)


if __name__ == "__main__":
    main()
