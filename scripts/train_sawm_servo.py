#!/usr/bin/env python3
"""
Train SAWM model on servo-mode data.

Same architecture as train_sawm.py but reads from data/sawm_servo_approaches/
where offsets range from 0–10cm (vs 0–5mm in descent data).

The servo data comes from tilted-gripper visual servo approaches where
the gripper is visible to the camera. This makes the model actually useful
for visual servoing.

Usage:
    # On Spark, after rsyncing data:
    python scripts/train_sawm_servo.py --data-dir data/sawm_servo_approaches --epochs 50

    # With ONNX export:
    python scripts/train_sawm_servo.py --data-dir data/sawm_servo_approaches --epochs 50 --export sawm_servo.onnx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.sawm.model import SAWMNet, export_onnx
from common.sawm.dataset import SAWMDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default data dir for servo approaches
SERVO_DATA_DIR = "data/sawm_servo_approaches"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load dataset (same format as descent data, different directory)
    data_dir = Path(args.data_dir)
    dataset = SAWMDataset(data_dir=data_dir, success_only=True)

    if len(dataset) == 0:
        logger.error(
            "No training samples found. Collect data first:\n"
            "  1. Enable servo data collection via MCP\n"
            "  2. Run servo_pick_at with blocks at various positions\n"
            "  3. rsync data/sawm_servo_approaches/ to this machine"
        )
        return

    logger.info(f"Dataset: {len(dataset)} samples")

    # Print offset range to verify servo data has large offsets
    offsets = [s[2:4] for s in dataset.samples]  # (dx, dy) tuples
    magnitudes = [np.sqrt(dx**2 + dy**2) * 1000 for dx, dy in offsets]
    logger.info(
        f"Offset magnitudes: min={min(magnitudes):.1f}mm "
        f"max={max(magnitudes):.1f}mm mean={np.mean(magnitudes):.1f}mm"
    )

    # Split train/val by approach
    train_ds, val_ds = dataset.split(val_fraction=args.val_fraction)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    # Create model
    model = SAWMNet(pretrained=True, dropout=args.dropout).to(device)

    # Compute and set normalization
    offset_mean, offset_std = dataset.compute_statistics()
    model.set_normalization(offset_mean.to(device), offset_std.to(device))
    logger.info(f"Offset stats: mean={offset_mean.tolist()}, std={offset_std.tolist()}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    offset_mean_dev = offset_mean.to(device)
    offset_std_dev = offset_std.to(device)

    scale_weight = args.scale_weight
    logger.info(f"Scale loss weight: {scale_weight}")

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_offset_loss = 0.0
        train_scale_loss = 0.0
        train_n = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            scales = batch["crop_scale"].to(device)
            offsets = batch["offset"].to(device)

            offset_targets = (offsets - offset_mean_dev) / offset_std_dev

            pred_offsets, pred_scales = model(images, scales)

            loss_offset = mse(pred_offsets, offset_targets)
            loss_scale = mse(pred_scales, scales)
            loss = loss_offset + scale_weight * loss_scale

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_offset_loss += loss_offset.item() * len(images)
            train_scale_loss += loss_scale.item() * len(images)
            train_n += len(images)

        train_offset_loss /= train_n
        train_scale_loss /= train_n

        # Validate
        model.eval()
        val_offset_loss = 0.0
        val_scale_loss = 0.0
        val_errors_mm = []
        val_scale_errors = []
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                scales = batch["crop_scale"].to(device)
                offsets = batch["offset"].to(device)

                preds_m, pred_scales = model.predict(images, scales)

                offset_targets = (offsets - offset_mean_dev) / offset_std_dev
                pred_offsets_norm, pred_scales_raw = model(images, scales)
                loss_offset = mse(pred_offsets_norm, offset_targets)
                loss_scale = mse(pred_scales_raw, scales)

                val_offset_loss += loss_offset.item() * len(images)
                val_scale_loss += loss_scale.item() * len(images)
                val_n += len(images)

                errors = (preds_m - offsets).abs() * 1000
                euclidean = torch.sqrt((errors ** 2).sum(dim=1))
                val_errors_mm.extend(euclidean.cpu().tolist())

                scale_err = (pred_scales - scales).abs()
                val_scale_errors.extend(scale_err.squeeze(-1).cpu().tolist())

        val_offset_loss /= max(val_n, 1)
        val_scale_loss /= max(val_n, 1)
        val_loss = val_offset_loss + scale_weight * val_scale_loss
        mean_error_mm = np.mean(val_errors_mm) if val_errors_mm else 0
        max_error_mm = np.max(val_errors_mm) if val_errors_mm else 0
        mean_scale_err = np.mean(val_scale_errors) if val_scale_errors else 0

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_offset_loss": round(train_offset_loss, 6),
            "train_scale_loss": round(train_scale_loss, 6),
            "val_offset_loss": round(val_offset_loss, 6),
            "val_scale_loss": round(val_scale_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mean_error_mm": round(mean_error_mm, 2),
            "val_max_error_mm": round(max_error_mm, 2),
            "val_mean_scale_error": round(mean_scale_err, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "offset_mean": offset_mean,
                "offset_std": offset_std,
                "val_loss": val_loss,
                "val_mean_error_mm": mean_error_mm,
                "val_mean_scale_error": mean_scale_err,
                "scale_weight": scale_weight,
                "data_type": "servo",
            }, args.output)
            marker = " *best*"
        else:
            marker = ""

        if epoch % args.log_every == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"offset={train_offset_loss:.5f}/{val_offset_loss:.5f} "
                f"scale={train_scale_loss:.5f}/{val_scale_loss:.5f} | "
                f"err={mean_error_mm:.1f}mm scale_err={mean_scale_err:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f}{marker}"
            )

    logger.info(f"Best model at epoch {best_epoch}, val_loss={best_val_loss:.6f}")

    # Save training history
    history_path = Path(args.output).with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # Export ONNX if requested
    if args.export:
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        model_cpu = SAWMNet(pretrained=False)
        model_cpu.load_state_dict(checkpoint["model_state_dict"])
        model_cpu.set_normalization(checkpoint["offset_mean"], checkpoint["offset_std"])
        model_cpu.eval()
        export_onnx(model_cpu, args.export)
        logger.info(f"ONNX exported to {args.export}")


def main():
    parser = argparse.ArgumentParser(description="Train SAWM servo model")
    parser.add_argument("--data-dir", default=SERVO_DATA_DIR,
                        help="Path to servo approach data")
    parser.add_argument("--output", default="sawm_servo_best.pt",
                        help="Output checkpoint path")
    parser.add_argument("--export", default=None,
                        help="Export ONNX to this path after training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scale-weight", type=float, default=0.3,
                        help="Weight for auxiliary scale loss (default: 0.3)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
