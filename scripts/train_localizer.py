#!/usr/bin/env python3
"""
Train the gripper localizer model on collected data.

Usage:
    python scripts/train_localizer.py --data-dir ./training_data --epochs 50

Can run on a machine with GPU, then export to ONNX for Pi deployment.
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gripper_localizer import create_model, export_to_onnx, PositionLoss


class RobotDataset(Dataset):
    """Dataset of (image, position) pairs for training."""

    def __init__(
        self,
        data_dir: str,
        sessions: list[str] = None,
        transform=None,
        include_gripper: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.include_gripper = include_gripper

        # Load samples from all sessions
        self.samples = []

        if sessions is None:
            sessions = [d.name for d in self.data_dir.iterdir() if d.is_dir()]

        for session in sessions:
            session_dir = self.data_dir / session
            samples_file = session_dir / "samples.json"

            if not samples_file.exists():
                continue

            with open(samples_file) as f:
                session_samples = json.load(f)

            for sample in session_samples:
                sample['_session_dir'] = str(session_dir)
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {len(sessions)} sessions")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        session_dir = Path(sample['_session_dir'])
        image_path = session_dir / sample['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract position
        rs = sample['robot_state']
        position = torch.tensor(rs['end_effector_position'], dtype=torch.float32)

        if self.include_gripper:
            gripper = torch.tensor([rs['gripper_width']], dtype=torch.float32)
            position = torch.cat([position, gripper])

        return {
            'image': image,
            'position': position,
        }


def get_transforms(train: bool = True, size: int = 224):
    """Get image transforms for training/validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def compute_statistics(dataset):
    """Compute position statistics for normalization."""
    positions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        positions.append(sample['position'])

    positions = torch.stack(positions)
    mean = positions.mean(dim=0)
    std = positions.std(dim=0).clamp(min=1e-6)

    return mean, std


def train_epoch(model, dataloader, criterion, optimizer, device, pos_mean, pos_std):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        positions = batch['position'].to(device)

        # Normalize targets
        positions_norm = (positions - pos_mean) / pos_std

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, positions_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, pos_mean, pos_std):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_error = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            positions = batch['position'].to(device)

            positions_norm = (positions - pos_mean) / pos_std

            outputs = model(images)
            loss = criterion(outputs, positions_norm)
            total_loss += loss.item()

            # Compute actual position error (in meters)
            pred_positions = outputs * pos_std + pos_mean
            error = (pred_positions - positions).abs().mean()
            total_error += error.item()

    n = len(dataloader)
    return total_loss / n, total_error / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--export-onnx", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    full_dataset = RobotDataset(args.data_dir, transform=train_transform)

    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Compute statistics on training data
    pos_mean, pos_std = compute_statistics(full_dataset)
    pos_mean = pos_mean.to(device)
    pos_std = pos_std.to(device)
    print(f"Position mean: {pos_mean}")
    print(f"Position std: {pos_std}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = create_model(backbone=args.backbone, pretrained=True)
    model.set_normalization(pos_mean.cpu(), pos_std.cpu())
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = PositionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, pos_mean, pos_std)
        val_loss, val_error = validate(model, val_loader, criterion, device, pos_mean, pos_std)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Error: {val_error*1000:.2f}mm")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'pos_mean': pos_mean.cpu(),
                'pos_std': pos_std.cpu(),
            }, output_dir / "best_model.pt")
            print("  Saved best model")

    # Final save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'pos_mean': pos_mean.cpu(),
        'pos_std': pos_std.cpu(),
    }, output_dir / "final_model.pt")

    # Export to ONNX
    if args.export_onnx:
        model.eval()
        export_to_onnx(model, str(output_dir / "gripper_localizer.onnx"))

    print(f"\nTraining complete. Models saved to {output_dir}")
    print(f"Best validation error: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
