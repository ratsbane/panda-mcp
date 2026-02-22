#!/usr/bin/env python3
"""
Train NUDGE model on collected approach data.

CrossEntropyLoss per head (weighted by class frequency), AdamW, CosineAnnealingLR.
Metrics: per-axis accuracy, within-1 accuracy, mean class distance.

Usage:
    # On Spark or Pi, after rsyncing data:
    python scripts/train_nudge.py --epochs 100

    # With ONNX export:
    python scripts/train_nudge.py --epochs 100 --export nudge.onnx
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.nudge.model import NUDGENet, export_onnx
from common.nudge.dataset import NUDGEDataset
from common.nudge.discretize import class_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute accuracy, within-1 accuracy, and mean class distance."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    within_1 = ((preds - labels).abs() <= 1).float()
    distance = (preds - labels).abs().float()
    return {
        "accuracy": correct.mean().item(),
        "within_1": within_1.mean().item(),
        "mean_dist": distance.mean().item(),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    dataset = NUDGEDataset(data_dir=data_dir, success_only=True)

    if len(dataset) == 0:
        logger.error(
            "No training samples found. Collect data first:\n"
            "  1. Enable NUDGE collection via MCP: nudge_collect_enable\n"
            "  2. Run pick_at() calls with blocks at various positions\n"
            "  3. rsync data/nudge_approaches/ to training machine"
        )
        return

    logger.info(f"Dataset: {len(dataset)} samples")

    # Split train/val by approach
    train_ds, val_ds = dataset.split(val_fraction=args.val_fraction)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Compute class weights from training set
    wx, wy, wz = train_ds.compute_class_weights()
    logger.info(f"Class weights x: {wx.tolist()}")
    logger.info(f"Class weights y: {wy.tolist()}")
    logger.info(f"Class weights z: {wz.tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    model = NUDGENet(dropout=args.dropout).to(device)
    logger.info(f"Model params: {NUDGENet.param_count()}")

    criterion_x = nn.CrossEntropyLoss(weight=wx.to(device))
    criterion_y = nn.CrossEntropyLoss(weight=wy.to(device))
    criterion_z = nn.CrossEntropyLoss(weight=wz.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_n = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            lx = batch["label_x"].to(device)
            ly = batch["label_y"].to(device)
            lz = batch["label_z"].to(device)

            pred_x, pred_y, pred_z = model(images)
            loss = criterion_x(pred_x, lx) + criterion_y(pred_y, ly) + criterion_z(pred_z, lz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)
            train_n += len(images)

        train_loss /= train_n

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_metrics = {"x": [], "y": [], "z": []}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                lx = batch["label_x"].to(device)
                ly = batch["label_y"].to(device)
                lz = batch["label_z"].to(device)

                pred_x, pred_y, pred_z = model(images)
                loss = criterion_x(pred_x, lx) + criterion_y(pred_y, ly) + criterion_z(pred_z, lz)
                val_loss += loss.item() * len(images)
                val_n += len(images)

                for axis, preds, labels in [("x", pred_x, lx), ("y", pred_y, ly), ("z", pred_z, lz)]:
                    all_metrics[axis].append(compute_metrics(preds, labels))

        val_loss /= max(val_n, 1)

        # Aggregate metrics
        def avg_metric(axis, key):
            vals = [m[key] for m in all_metrics[axis]]
            return np.mean(vals) if vals else 0.0

        acc_x = avg_metric("x", "accuracy")
        acc_y = avg_metric("y", "accuracy")
        acc_z = avg_metric("z", "accuracy")
        mean_acc = (acc_x + acc_y + acc_z) / 3

        w1_x = avg_metric("x", "within_1")
        w1_y = avg_metric("y", "within_1")
        w1_z = avg_metric("z", "within_1")
        mean_w1 = (w1_x + w1_y + w1_z) / 3

        dist_x = avg_metric("x", "mean_dist")
        dist_y = avg_metric("y", "mean_dist")
        dist_z = avg_metric("z", "mean_dist")

        scheduler.step()

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "acc_x": round(acc_x, 4), "acc_y": round(acc_y, 4), "acc_z": round(acc_z, 4),
            "mean_acc": round(mean_acc, 4),
            "w1_x": round(w1_x, 4), "w1_y": round(w1_y, 4), "w1_z": round(w1_z, 4),
            "mean_w1": round(mean_w1, 4),
            "dist_x": round(dist_x, 3), "dist_y": round(dist_y, 3), "dist_z": round(dist_z, 3),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(rec)

        # Save best by mean accuracy
        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "mean_acc": mean_acc,
                "mean_w1": mean_w1,
                "class_weights": {"x": wx.tolist(), "y": wy.tolist(), "z": wz.tolist()},
            }, args.output)
            marker = " *best*"
        else:
            marker = ""

        if epoch % args.log_every == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={train_loss:.4f}/{val_loss:.4f} | "
                f"acc={acc_x:.3f}/{acc_y:.3f}/{acc_z:.3f} (mean={mean_acc:.3f}) | "
                f"w1={mean_w1:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f}{marker}"
            )

    logger.info(f"Best model at epoch {best_epoch}, mean_acc={best_val_acc:.4f}")

    # Save history
    history_path = Path(args.output).with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # Export ONNX
    if args.export:
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        model_cpu = NUDGENet()
        model_cpu.load_state_dict(checkpoint["model_state_dict"])
        model_cpu.eval()
        export_onnx(model_cpu, args.export)
        logger.info(f"ONNX exported to {args.export}")


def main():
    parser = argparse.ArgumentParser(description="Train NUDGE model")
    parser.add_argument("--data-dir", default="data/nudge_approaches")
    parser.add_argument("--output", default="nudge_best.pt")
    parser.add_argument("--export", default=None, help="Export ONNX path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
