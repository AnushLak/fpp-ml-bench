"""
Dataset loader for Fringe Projection Profilometry
Handles different normalization types: raw, global_normalized, individual_normalized
Input: PNG fringe images
Output: MAT depth files
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import scipy.io as sio


class FringeFPPDataset(Dataset):
    """
    Dataset for FPP training
    Input: PNG fringe images (grayscale)
    Output: MAT depth files

    Args:
        fringe_dir: Directory containing fringe images (.png)
        depth_dir: Directory containing depth maps (.mat)
        dataset_type: One of ['_raw', '_global_normalized', '_individual_normalized']
                     '_raw': Use raw depth values from .mat
                     '_global_normalized': Normalize by a global constant (65535)
                     '_individual_normalized': Normalize each depth map individually
    """

    def __init__(self, fringe_dir, depth_dir, dataset_type='_individual_normalized'):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.dataset_type = dataset_type

        # Validate dataset type
        valid_types = ['_raw', '_global_normalized', '_individual_normalized']
        if dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, got {dataset_type}")

        # Get all fringe images
        self.fringe_files = sorted(list(self.fringe_dir.glob("*.png")))

        if not self.fringe_files:
            raise ValueError(f"No PNG files found in {fringe_dir}")

        # Find matching depth MAT files (same base name)
        self.pairs = []
        missing = []

        for fringe_path in self.fringe_files:
            stem = fringe_path.stem
            depth_path = self.depth_dir / f"{stem}.mat"

            if depth_path.exists():
                self.pairs.append((fringe_path, depth_path))
            else:
                missing.append(stem)

        if not self.pairs:
            raise ValueError(f"No matching depth MAT files found in {depth_dir}")

        if missing:
            print(f"⚠ Warning: {len(missing)}/{len(self.fringe_files)} fringe images have no depth match")

        print(f"✓ Loaded {len(self.pairs)} sample pairs from {fringe_dir}")
        print(f"  Dataset type: {dataset_type}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fringe_path, depth_path = self.pairs[idx]

        # Load fringe (grayscale, normalize to [0,1])
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32)
        if fringe.max() > 1.5:
            fringe = fringe / 255.0

        # Load depth from .mat file
        mat_data = sio.loadmat(depth_path)
        # Assuming the depth data is stored in a key like 'depth' or the first variable
        if 'depth' in mat_data:
            depth = mat_data['depth']
        else:
            # Get the first non-metadata key
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(keys) == 0:
                raise ValueError(f"No data found in {depth_path}")
            depth = mat_data[keys[0]]

        depth = depth.astype(np.float32)

        # Apply normalization based on dataset_type
        if self.dataset_type == '_raw':
            # Use raw depth values (no normalization)
            pass
        elif self.dataset_type == '_global_normalized':
            # Normalize by global constant (uint16 max value)
            depth = depth / 65535.0
        elif self.dataset_type == '_individual_normalized':
            # Normalize each depth map individually by its max value
            max_val = depth.max()
            if max_val > 1e-6:  # Avoid division by zero
                depth = depth / max_val

        # Convert to tensors
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # (1, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)    # (1, H, W)

        return fringe, depth, str(fringe_path.name)
