"""
PyTorch Dataset for NUDGE training.

Loads frames and discrete labels from successful approaches.
Constructs 4-channel input (RGB + binary target mask) from stored bbox.
Split by approach (not frame) to prevent data leakage.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "nudge_approaches"


class NUDGEDataset(Dataset):
    """
    Dataset of (4ch_image, label_x, label_y, label_z) samples.

    Each sample is one frame from a successful approach trajectory.
    The 4th channel is a binary mask indicating the target bbox region.
    """

    NUM_CLASSES = 7

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        train: bool = True,
        success_only: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.train = train
        self.samples = []  # (frame_path, bbox, label_x, label_y, label_z)
        self._load_approaches(success_only)
        logger.info(
            f"NUDGEDataset: {len(self.samples)} samples from "
            f"{len(set(s[0].parent.parent for s in self.samples))} approaches"
        )

    def _load_approaches(self, success_only: bool):
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
                frame_path = approach_dir / "frames" / f"{label['frame_idx']:03d}.jpg"
                if not frame_path.exists():
                    continue

                bbox = label.get("target_bbox_px")
                if bbox is None:
                    continue

                self.samples.append((
                    frame_path,
                    bbox,  # [x1, y1, x2, y2] in 224x224 coords
                    int(label["dx_class"]),
                    int(label["dy_class"]),
                    int(label["dz_class"]),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame_path, bbox, lx, ly, lz = self.samples[idx]

        # Load image as RGB float [0,1]
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Resize to 224x224 if needed
        if h != 224 or w != 224:
            img = cv2.resize(img, (224, 224))
            # Scale bbox accordingly
            sx, sy = 224.0 / w, 224.0 / h
            bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]

        img = img.astype(np.float32) / 255.0  # [0, 1], NOT ImageNet normalized

        # Construct binary mask from bbox
        mask = np.zeros((224, 224), dtype=np.float32)
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(224, int(bbox[2]))
        y2 = min(224, int(bbox[3]))
        mask[y1:y2, x1:x2] = 1.0

        # Augmentation (train only)
        if self.train:
            img, mask, lx, ly, lz = self._augment(img, mask, lx, ly, lz)

        # Stack to 4 channels: (H, W, 3) + (H, W, 1) -> (4, 224, 224)
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, 224, 224)
        mask_t = torch.from_numpy(mask).unsqueeze(0)     # (1, 224, 224)
        image_4ch = torch.cat([img_t, mask_t], dim=0)    # (4, 224, 224)

        return {
            "image": image_4ch,
            "label_x": torch.tensor(lx, dtype=torch.long),
            "label_y": torch.tensor(ly, dtype=torch.long),
            "label_z": torch.tensor(lz, dtype=torch.long),
        }

    def _augment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        lx: int, ly: int, lz: int,
    ) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
        """Apply training augmentations."""
        rng = np.random.default_rng()

        # Horizontal flip (50% chance) -- flips Y axis label sign
        if rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            # Flip Y label: class index i -> 6 - i (mirrors around center class 3)
            ly = 6 - ly

        # Color jitter (brightness, contrast)
        if rng.random() < 0.5:
            brightness = rng.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        if rng.random() < 0.3:
            contrast = rng.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * contrast + mean, 0, 1)

        # Small bbox jitter (translate mask +-5px) -- doesn't change labels
        if rng.random() < 0.3:
            shift_x = rng.integers(-5, 6)
            shift_y = rng.integers(-5, 6)
            mask = np.roll(mask, shift_x, axis=1)
            mask = np.roll(mask, shift_y, axis=0)

        return img, mask, lx, ly, lz

    def compute_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute inverse-frequency weights for balanced CrossEntropyLoss."""
        counts_x = Counter(s[2] for s in self.samples)
        counts_y = Counter(s[3] for s in self.samples)
        counts_z = Counter(s[4] for s in self.samples)

        def _weights(counts: Counter) -> torch.Tensor:
            total = sum(counts.values())
            w = torch.ones(self.NUM_CLASSES)
            for cls, count in counts.items():
                if 0 <= cls < self.NUM_CLASSES and count > 0:
                    w[cls] = total / (self.NUM_CLASSES * count)
            return w

        return _weights(counts_x), _weights(counts_y), _weights(counts_z)

    def split(
        self, val_fraction: float = 0.15
    ) -> Tuple["NUDGEDataset", "NUDGEDataset"]:
        """Split into train/val by approach (not by frame)."""
        approach_groups: dict[str, list[int]] = {}
        for i, (path, *_) in enumerate(self.samples):
            key = str(path.parent.parent)
            if key not in approach_groups:
                approach_groups[key] = []
            approach_groups[key].append(i)

        approaches = list(approach_groups.keys())
        np.random.shuffle(approaches)
        n_val = max(1, int(len(approaches) * val_fraction))
        val_approaches = set(approaches[:n_val])

        train_ds = NUDGEDataset.__new__(NUDGEDataset)
        train_ds.data_dir = self.data_dir
        train_ds.train = True
        train_ds.samples = []

        val_ds = NUDGEDataset.__new__(NUDGEDataset)
        val_ds.data_dir = self.data_dir
        val_ds.train = False
        val_ds.samples = []

        for key, indices in approach_groups.items():
            target = val_ds if key in val_approaches else train_ds
            for i in indices:
                target.samples.append(self.samples[i])

        return train_ds, val_ds


if __name__ == "__main__":
    ds = NUDGEDataset()
    print(f"Dataset: {len(ds)} samples")

    if len(ds) > 0:
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Labels: x={sample['label_x']}, y={sample['label_y']}, z={sample['label_z']}")

        wx, wy, wz = ds.compute_class_weights()
        print(f"Class weights x: {wx}")
        print(f"Class weights y: {wy}")
        print(f"Class weights z: {wz}")
