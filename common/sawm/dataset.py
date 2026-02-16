"""
PyTorch Dataset for SAWM training.

Loads progressive crops and (dx, dy) labels from successful approaches.
Filters out failed approaches automatically.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "sawm_approaches"


class SAWMDataset(Dataset):
    """
    Dataset of (crop, scale, dx, dy) samples from successful pick approaches.

    Each sample is one frame from one approach trajectory.
    Only includes frames from successful grasps (self-supervised labels).
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        transform=None,
        success_only: bool = True,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # (crop_path, scale, dx, dy)

        self._load_approaches(success_only)
        logger.info(
            f"SAWMDataset: {len(self.samples)} samples from "
            f"{len(set(s[0].parent.parent for s in self.samples))} approaches"
        )

    def _load_approaches(self, success_only: bool):
        """Scan approach directories and build sample list."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        for approach_dir in sorted(self.data_dir.glob("approach_*")):
            meta_path = approach_dir / "metadata.json"
            labels_path = approach_dir / "labels.json"

            if not meta_path.exists() or not labels_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            if success_only and not meta.get("success", False):
                continue

            with open(labels_path) as f:
                labels = json.load(f)

            for label in labels:
                crop_path = approach_dir / "frames" / f"{label['idx']:03d}.jpg"
                if crop_path.exists():
                    self.samples.append((
                        crop_path,
                        float(label["crop_scale"]),
                        float(label["dx"]),
                        float(label["dy"]),
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        crop_path, scale, dx, dy = self.samples[idx]

        # Load image
        img = cv2.imread(str(crop_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            # Default: normalize to ImageNet stats
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, 224, 224)

        return {
            "image": img,
            "crop_scale": torch.tensor([scale], dtype=torch.float32),
            "offset": torch.tensor([dx, dy], dtype=torch.float32),
        }

    def compute_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of (dx, dy) offsets for normalization."""
        offsets = torch.tensor(
            [[s[2], s[3]] for s in self.samples],
            dtype=torch.float32,
        )
        mean = offsets.mean(dim=0)
        std = offsets.std(dim=0).clamp(min=1e-6)
        return mean, std

    def split(
        self, val_fraction: float = 0.15
    ) -> Tuple["SAWMDataset", "SAWMDataset"]:
        """
        Split into train/val by approach (not by frame).

        This ensures no data leakage — all frames from the same approach
        stay in the same split.
        """
        # Group samples by approach directory
        approach_groups = {}
        for i, (path, scale, dx, dy) in enumerate(self.samples):
            approach_key = str(path.parent.parent)
            if approach_key not in approach_groups:
                approach_groups[approach_key] = []
            approach_groups[approach_key].append(i)

        # Split approaches
        approaches = list(approach_groups.keys())
        np.random.shuffle(approaches)
        n_val = max(1, int(len(approaches) * val_fraction))
        val_approaches = set(approaches[:n_val])

        train_ds = SAWMDataset.__new__(SAWMDataset)
        train_ds.data_dir = self.data_dir
        train_ds.transform = self.transform
        train_ds.samples = []

        val_ds = SAWMDataset.__new__(SAWMDataset)
        val_ds.data_dir = self.data_dir
        val_ds.transform = self.transform
        val_ds.samples = []

        for approach_key, indices in approach_groups.items():
            target = val_ds if approach_key in val_approaches else train_ds
            for i in indices:
                target.samples.append(self.samples[i])

        return train_ds, val_ds


if __name__ == "__main__":
    ds = SAWMDataset()
    print(f"Dataset: {len(ds)} samples")

    if len(ds) > 0:
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Crop scale: {sample['crop_scale']}")
        print(f"Offset: {sample['offset']}")

        mean, std = ds.compute_statistics()
        print(f"Offset mean: {mean}")
        print(f"Offset std: {std}")
