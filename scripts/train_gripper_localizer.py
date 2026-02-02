#!/usr/bin/env python3
"""
Train a gripper position localizer model.

Usage:
    python scripts/train_gripper_localizer.py --data ~/datasets/gripper_v1 --backbone mobilenetv3_large
    python scripts/train_gripper_localizer.py --data ~/datasets/gripper_v1 --backbone efficientnet_b2 --epochs 200
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from tqdm import tqdm


class GripperDataset(Dataset):
    """Dataset of (image, gripper_position) pairs."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Check for samples.json (batch format from data collection)
        samples_json = self.data_dir / "samples.json"
        if samples_json.exists():
            with open(samples_json) as f:
                data = json.load(f)
            for sample in data:
                img_path = self.data_dir / sample["image_path"]
                if img_path.exists():
                    # Extract position from robot_state
                    pos = sample["robot_state"]["end_effector_position"]
                    self.samples.append((img_path, pos))
        else:
            # Fallback: individual JSON files per image
            for json_path in sorted(self.data_dir.glob("*.json")):
                if json_path.name in ("metadata.json", "samples.json"):
                    continue
                img_path = json_path.with_suffix(".jpg")
                if img_path.exists():
                    with open(json_path) as f:
                        meta = json.load(f)
                    ee = meta["end_effector"]
                    pos = [ee["position_m"]["x"], ee["position_m"]["y"], ee["position_m"]["z"]]
                    self.samples.append((img_path, pos))

        print(f"Found {len(self.samples)} samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, position = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert position to tensor
        position = torch.tensor(position, dtype=torch.float32)

        return image, position


class GripperLocalizer(nn.Module):
    """CNN model that predicts gripper (x, y, z) from an image."""

    def __init__(self, backbone: str = "mobilenetv3_large_100", pretrained: bool = True):
        super().__init__()

        # Load pretrained backbone from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            feat_dim = features.shape[1]

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # x, y, z
        )

        print(f"Created {backbone} with {feat_dim}D features")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def get_transforms(train: bool = True, img_size: int = 224):
    """Get image transforms for training or validation.

    Note: For position regression, we avoid spatial augmentations (flip, rotation, crop)
    because they would break the image-to-position correspondence.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # Only color augmentation - spatial augmentations break position labels
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0

    for images, positions in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        positions = positions.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss = criterion(predictions, positions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(images)
            loss = criterion(predictions, positions)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_errors = []

    with torch.no_grad():
        for images, positions in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            positions = positions.to(device)

            predictions = model(images)
            loss = criterion(predictions, positions)

            # Calculate per-sample Euclidean error (in mm)
            errors = torch.norm(predictions - positions, dim=1) * 1000  # Convert to mm
            all_errors.extend(errors.cpu().numpy())

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    all_errors = np.array(all_errors)

    return {
        "loss": total_loss / total_samples,
        "mean_error_mm": np.mean(all_errors),
        "median_error_mm": np.median(all_errors),
        "p90_error_mm": np.percentile(all_errors, 90),
        "p99_error_mm": np.percentile(all_errors, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="Train gripper position localizer")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--backbone", type=str, default="mobilenetv3_large_100",
                        choices=["mobilenetv3_small_100", "mobilenetv3_large_100",
                                 "efficientnet_b0", "efficientnet_b2", "efficientnet_b4",
                                 "convnext_tiny", "convnext_small"],
                        help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output", type=str, default="./runs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision training")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.backbone}_{timestamp}"
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Setup TensorBoard
    writer = SummaryWriter(run_dir / "tensorboard")

    # Load dataset
    full_dataset = GripperDataset(args.data, transform=None)  # Transform applied later

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_indices, val_indices = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create datasets with appropriate transforms
    train_dataset = GripperDataset(args.data, transform=get_transforms(train=True, img_size=args.img_size))
    val_dataset = GripperDataset(args.data, transform=get_transforms(train=False, img_size=args.img_size))

    # Subset to train/val splits
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices.indices]
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices.indices]

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    model = GripperLocalizer(backbone=args.backbone, pretrained=True)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Loss function - Smooth L1 is more robust to outliers than MSE
    criterion = nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler - cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_error = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_error = checkpoint.get("best_val_error", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Logging to {run_dir / 'tensorboard'}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Error/mean_mm", val_metrics["mean_error_mm"], epoch)
        writer.add_scalar("Error/median_mm", val_metrics["median_error_mm"], epoch)
        writer.add_scalar("Error/p90_mm", val_metrics["p90_error_mm"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Mean Error: {val_metrics['mean_error_mm']:.2f}mm")
        print(f"  Median Error: {val_metrics['median_error_mm']:.2f}mm")
        print(f"  P90 Error: {val_metrics['p90_error_mm']:.2f}mm")
        print(f"  LR: {current_lr:.2e}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_val_error": best_val_error,
            "config": config,
        }
        torch.save(checkpoint, run_dir / "checkpoint_latest.pt")

        # Save best model
        if val_metrics["mean_error_mm"] < best_val_error:
            best_val_error = val_metrics["mean_error_mm"]
            torch.save(checkpoint, run_dir / "checkpoint_best.pt")
            print(f"  New best model! Error: {best_val_error:.2f}mm")

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, run_dir / f"checkpoint_epoch{epoch + 1}.pt")

    # Final summary
    print(f"\nTraining complete!")
    print(f"Best validation error: {best_val_error:.2f}mm")
    print(f"Checkpoints saved to {run_dir}")

    # Export best model for inference
    print("\nExporting best model for inference...")
    best_checkpoint = torch.load(run_dir / "checkpoint_best.pt")
    model.load_state_dict(best_checkpoint["model"])
    model.eval()

    # Save just the model weights
    torch.save(model.state_dict(), run_dir / "model_best.pt")

    # Export to ONNX for AI Hat deployment
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    onnx_path = run_dir / "model_best.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["image"],
        output_names=["position"],
        dynamic_axes={"image": {0: "batch"}, "position": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {onnx_path}")

    writer.close()


if __name__ == "__main__":
    main()
