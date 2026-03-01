#!/usr/bin/env python3
"""
Train NUDGE v2 regression model on collected approach data.

Predicts continuous (dx, dy, dz) offsets in mm from image + gripper position.
Uses Smooth L1 loss, AdamW with CosineAnnealingLR.

Usage:
    python scripts/train_nudge.py --epochs 100
    python scripts/train_nudge.py --epochs 100 --test-fraction 0.15 --export nudge.onnx
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model, loader, criterion, device):
    """Run evaluation. Returns loss and per-axis metrics."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_targets = []
    all_labels = {"x": [], "y": [], "z": []}
    all_pred_classes = {"x": [], "y": [], "z": []}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gripper = batch["gripper_xyz"].to(device)
            targets = batch["offset_mm"].to(device)

            px, py, pz = model(images, gripper)
            preds = torch.cat([px, py, pz], dim=1)  # (B, 3)

            loss = criterion(preds, targets)
            total_loss += loss.item() * len(images)
            total_n += len(images)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            # Discrete class labels for accuracy metrics
            for i, axis in enumerate(["x", "y", "z"]):
                all_labels[axis].append(batch[f"label_{axis}"])
                # Convert predicted mm to class
                threshold = 15.0 if axis == "z" else 4.0
                pred_mm = preds[:, i].cpu()
                pred_cls = torch.ones_like(pred_mm, dtype=torch.long)
                pred_cls[pred_mm < -threshold] = 0
                pred_cls[pred_mm > threshold] = 2
                all_pred_classes[axis].append(pred_cls)

    all_preds = torch.cat(all_preds)    # (N, 3)
    all_targets = torch.cat(all_targets)  # (N, 3)

    errors = all_preds - all_targets      # (N, 3)
    abs_errors = errors.abs()

    results = {
        "loss": total_loss / max(total_n, 1),
        "n_samples": total_n,
    }

    axis_names = ["x", "y", "z"]
    for i, axis in enumerate(axis_names):
        results[f"mae_{axis}"] = abs_errors[:, i].mean().item()
        results[f"rmse_{axis}"] = errors[:, i].pow(2).mean().sqrt().item()
        results[f"median_ae_{axis}"] = abs_errors[:, i].median().item()

        # Classification accuracy from regression predictions
        pred_cls = torch.cat(all_pred_classes[axis])
        true_cls = torch.cat(all_labels[axis])
        results[f"cls_acc_{axis}"] = (pred_cls == true_cls).float().mean().item()

        # Per-class accuracy
        per_class = {}
        labels = ["negative", "aligned", "positive"]
        for c in range(3):
            mask = true_cls == c
            if mask.sum() > 0:
                per_class[labels[c]] = {
                    "count": int(mask.sum()),
                    "accuracy": float((pred_cls[mask] == c).float().mean()),
                }
        results[f"per_class_{axis}"] = per_class

    results["mean_mae"] = np.mean([results[f"mae_{a}"] for a in axis_names])
    results["mean_cls_acc"] = np.mean([results[f"cls_acc_{a}"] for a in axis_names])
    return results


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    dataset = NUDGEDataset(data_dir=data_dir, success_only=True)

    if len(dataset) == 0:
        logger.error("No training samples found.")
        return

    logger.info(f"Dataset: {len(dataset)} samples")

    # Split train/val/test
    if args.test_fraction > 0:
        train_ds, val_ds, test_ds = dataset.split(
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} (held out)")
    else:
        train_ds, val_ds = dataset.split(
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        test_ds = None
        logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Offset statistics
    stats = train_ds.compute_class_weights()

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

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mae = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_n = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            gripper = batch["gripper_xyz"].to(device)
            targets = batch["offset_mm"].to(device)

            px, py, pz = model(images, gripper)
            preds = torch.cat([px, py, pz], dim=1)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)
            train_n += len(images)

        train_loss /= train_n

        # --- Validate ---
        val_results = evaluate(model, val_loader, criterion, device)
        val_loss = val_results["loss"]
        mean_mae = val_results["mean_mae"]
        mean_cls_acc = val_results["mean_cls_acc"]

        scheduler.step()

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "mae_x": round(val_results["mae_x"], 2),
            "mae_y": round(val_results["mae_y"], 2),
            "mae_z": round(val_results["mae_z"], 2),
            "mean_mae": round(mean_mae, 2),
            "cls_acc_x": round(val_results["cls_acc_x"], 4),
            "cls_acc_y": round(val_results["cls_acc_y"], 4),
            "cls_acc_z": round(val_results["cls_acc_z"], 4),
            "mean_cls_acc": round(mean_cls_acc, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(rec)

        # Save best by mean MAE
        if mean_mae < best_val_mae:
            best_val_mae = mean_mae
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "mean_mae": mean_mae,
                "mean_cls_acc": mean_cls_acc,
                "offset_stats": stats,
            }, args.output)
            marker = " *best*"
        else:
            marker = ""

        if epoch % args.log_every == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"loss={train_loss:.4f}/{val_loss:.4f} | "
                f"MAE={val_results['mae_x']:.1f}/{val_results['mae_y']:.1f}/{val_results['mae_z']:.1f}mm "
                f"(mean={mean_mae:.1f}) | "
                f"cls_acc={mean_cls_acc:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f}{marker}"
            )

    logger.info(f"\nBest model at epoch {best_epoch}, mean_mae={best_val_mae:.2f}mm")

    # Save history
    history_path = Path(args.output).with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # --- Final test evaluation ---
    if test_ds is not None and len(test_ds) > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SET EVALUATION ({len(test_ds)} samples)")
        logger.info(f"{'='*60}")

        checkpoint = torch.load(args.output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
        )
        test_results = evaluate(model, test_loader, criterion, device)

        logger.info(f"Test loss: {test_results['loss']:.4f}")
        logger.info(
            f"Test MAE: X={test_results['mae_x']:.1f}mm "
            f"Y={test_results['mae_y']:.1f}mm Z={test_results['mae_z']:.1f}mm "
            f"(mean={test_results['mean_mae']:.1f}mm)"
        )
        logger.info(
            f"Test RMSE: X={test_results['rmse_x']:.1f}mm "
            f"Y={test_results['rmse_y']:.1f}mm Z={test_results['rmse_z']:.1f}mm"
        )
        logger.info(
            f"Test classification accuracy: "
            f"X={test_results['cls_acc_x']:.3f} "
            f"Y={test_results['cls_acc_y']:.3f} "
            f"Z={test_results['cls_acc_z']:.3f} "
            f"(mean={test_results['mean_cls_acc']:.3f})"
        )

        for axis in ["x", "y", "z"]:
            logger.info(f"\nPer-class accuracy ({axis.upper()}):")
            for label, info in test_results[f"per_class_{axis}"].items():
                logger.info(f"  {label:>8s}: {info['accuracy']:.3f} ({info['count']} samples)")

        test_path = Path(args.output).with_suffix(".test_results.json")
        with open(test_path, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"\nTest results saved to {test_path}")

    # Export ONNX
    if args.export:
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
        model_cpu = NUDGENet()
        model_cpu.load_state_dict(checkpoint["model_state_dict"])
        model_cpu.eval()
        export_onnx(model_cpu, args.export)
        logger.info(f"ONNX exported to {args.export}")


def main():
    parser = argparse.ArgumentParser(description="Train NUDGE v2 regression model")
    parser.add_argument("--data-dir", default="data/nudge_approaches")
    parser.add_argument("--output", default="nudge_best.pt")
    parser.add_argument("--export", default=None, help="Export ONNX path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
