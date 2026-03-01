"""
PyTorch Dataset for NUDGE training (v2 -- regression).

Loads frames, gripper positions, and continuous offsets from successful approaches.
Constructs 4-channel input (RGB + binary target mask) from stored bbox.
Returns gripper XYZ and offset in mm for regression training.
Split by approach (not frame) to prevent data leakage.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "nudge_approaches"


class NUDGEDataset(Dataset):
    """
    Dataset of (4ch_image, gripper_xyz, offset_mm) samples.

    Each sample is one frame from a successful approach trajectory.
    The 4th channel is a binary mask indicating the target bbox region.
    """

    NUM_CLASSES = 3  # kept for compat

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        train: bool = True,
        success_only: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.train = train
        # (frame_path, bbox, gripper_xyz, offset_mm)
        self.samples = []
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

                # Gripper position in meters
                gx = float(label.get("gripper_x", 0))
                gy = float(label.get("gripper_y", 0))
                gz = float(label.get("gripper_z", 0))

                # Continuous offset in meters → mm
                dx_mm = float(label.get("dx_m", 0)) * 1000.0
                dy_mm = float(label.get("dy_m", 0)) * 1000.0
                dz_mm = float(label.get("dz_m", 0)) * 1000.0

                self.samples.append((
                    frame_path,
                    bbox,  # [x1, y1, x2, y2] in 224x224 coords
                    [gx, gy, gz],
                    [dx_mm, dy_mm, dz_mm],
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        frame_path, bbox, gripper_xyz, offset_mm = self.samples[idx]

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

        img = img.astype(np.float32) / 255.0

        # Construct binary mask from bbox
        mask = np.zeros((224, 224), dtype=np.float32)
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(224, int(bbox[2]))
        y2 = min(224, int(bbox[3]))
        mask[y1:y2, x1:x2] = 1.0

        dx_mm, dy_mm, dz_mm = offset_mm
        gxyz = list(gripper_xyz)

        # Augmentation (train only)
        if self.train:
            img, mask, dx_mm, dy_mm, dz_mm, gxyz = self._augment(
                img, mask, dx_mm, dy_mm, dz_mm, gxyz
            )

        # Stack to 4 channels
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, 224, 224)
        mask_t = torch.from_numpy(mask).unsqueeze(0)     # (1, 224, 224)
        image_4ch = torch.cat([img_t, mask_t], dim=0)    # (4, 224, 224)

        return {
            "image": image_4ch,
            "gripper_xyz": torch.tensor(gxyz, dtype=torch.float32),
            "offset_mm": torch.tensor([dx_mm, dy_mm, dz_mm], dtype=torch.float32),
            # Keep discrete labels for backwards compat / metrics
            "label_x": torch.tensor(self._to_class(dx_mm, "x"), dtype=torch.long),
            "label_y": torch.tensor(self._to_class(dy_mm, "y"), dtype=torch.long),
            "label_z": torch.tensor(self._to_class(dz_mm, "z"), dtype=torch.long),
        }

    @staticmethod
    def _to_class(offset_mm: float, axis: str) -> int:
        """Convert mm offset to 3-class label for metrics."""
        threshold = 15.0 if axis == "z" else 4.0
        if offset_mm < -threshold:
            return 0
        elif offset_mm > threshold:
            return 2
        return 1

    def _augment(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        dx_mm: float, dy_mm: float, dz_mm: float,
        gxyz: list,
    ) -> tuple:
        """Apply training augmentations."""
        rng = np.random.default_rng()

        # Horizontal flip (50% chance) -- flips Y axis
        if rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            dy_mm = -dy_mm
            gxyz[1] = -gxyz[1]  # flip gripper Y too

        # Color jitter (brightness, contrast)
        if rng.random() < 0.5:
            brightness = rng.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
        if rng.random() < 0.3:
            contrast = rng.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * contrast + mean, 0, 1)

        # Small bbox jitter (translate mask +-5px)
        if rng.random() < 0.3:
            shift_x = rng.integers(-5, 6)
            shift_y = rng.integers(-5, 6)
            mask = np.roll(mask, shift_x, axis=1)
            mask = np.roll(mask, shift_y, axis=0)

        return img, mask, dx_mm, dy_mm, dz_mm, gxyz

    def compute_class_weights(self):
        """Compute offset statistics (replaces class weights for regression)."""
        offsets = np.array([s[3] for s in self.samples])  # (N, 3) in mm
        stats = {
            "mean": offsets.mean(axis=0).tolist(),
            "std": offsets.std(axis=0).tolist(),
            "min": offsets.min(axis=0).tolist(),
            "max": offsets.max(axis=0).tolist(),
        }
        logger.info(f"Offset stats (mm): mean={[f'{v:.1f}' for v in stats['mean']]}, "
                     f"std={[f'{v:.1f}' for v in stats['std']]}")
        return stats

    def split(
        self, val_fraction: float = 0.15, test_fraction: float = 0.0,
        seed: int = 42,
    ) -> Tuple["NUDGEDataset", ...]:
        """Split into train/val (and optionally test) by approach."""
        approach_groups: dict[str, list[int]] = {}
        for i, (path, *_) in enumerate(self.samples):
            key = str(path.parent.parent)
            if key not in approach_groups:
                approach_groups[key] = []
            approach_groups[key].append(i)

        approaches = list(approach_groups.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(approaches)

        n_test = max(1, int(len(approaches) * test_fraction)) if test_fraction > 0 else 0
        n_val = max(1, int(len(approaches) * val_fraction))

        test_approaches = set(approaches[:n_test])
        val_approaches = set(approaches[n_test:n_test + n_val])

        def _make_ds(train_flag):
            ds = NUDGEDataset.__new__(NUDGEDataset)
            ds.data_dir = self.data_dir
            ds.train = train_flag
            ds.samples = []
            return ds

        train_ds = _make_ds(True)
        val_ds = _make_ds(False)
        test_ds = _make_ds(False) if test_fraction > 0 else None

        for key, indices in approach_groups.items():
            if key in test_approaches:
                target = test_ds
            elif key in val_approaches:
                target = val_ds
            else:
                target = train_ds
            for i in indices:
                target.samples.append(self.samples[i])

        if test_ds is not None:
            return train_ds, val_ds, test_ds
        return train_ds, val_ds


if __name__ == "__main__":
    ds = NUDGEDataset()
    print(f"Dataset: {len(ds)} samples")

    if len(ds) > 0:
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Gripper XYZ: {sample['gripper_xyz']}")
        print(f"Offset mm: {sample['offset_mm']}")
        print(f"Labels: x={sample['label_x']}, y={sample['label_y']}, z={sample['label_z']}")

        stats = ds.compute_class_weights()
        print(f"Offset stats: {stats}")
