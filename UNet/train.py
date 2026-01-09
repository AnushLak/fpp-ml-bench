# -*- coding: utf-8 -*-
"""
PyTorch train.py for paired fringe->depth mapping with explicit splits.

Input:
  train/fringe/<object>_a{angle}.png
  val/fringe/<object>_a{angle}.png
  test/fringe/<object>_a{angle}.png

Target:
  train/depth/<object>_a{angle}.mat
  val/depth/<object>_a{angle}.mat
  test/depth/<object>_a{angle}.mat

Checkpoints:
  ./checkpoints
"""

import os
import time, csv
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import UNetFPP, masked_rmse  # your PyTorch model + masked loss


# -----------------------------
# Repro / device
# -----------------------------
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Config
# -----------------------------
INPUT_HEIGHT = 960
INPUT_WIDTH  = 960
OUTPUT_HEIGHT = 960
OUTPUT_WIDTH  = 960

BATCH_SIZE = 4
NUM_EPOCH = 700

# MODIFY THIS BASED ON YOUR DATA ORGANIZATION
train_fringe_dir = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/train/fringe")
train_depth_dir  = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/train/depth")

val_fringe_dir   = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/val/fringe")
val_depth_dir    = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/val/depth")

test_fringe_dir  = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/test/fringe")
test_depth_dir   = Path(r"/work/arpawar/anushlak/SPIE-PW/fpp_unet_training_data/test/depth")

out_path = Path(r"results")
checkpoint_dir = Path(r"./checkpoints")


# -----------------------------
# Helpers
# -----------------------------
def center_crop_or_pad(arr: np.ndarray, th: int, tw: int, pad_value: float = 0.0) -> np.ndarray:
    """Ensure arr is (th,tw) by center-crop/pad."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    h, w = arr.shape

    # crop
    if h > th:
        top = (h - th) // 2
        arr = arr[top:top + th, :]
    if w > tw:
        left = (w - tw) // 2
        arr = arr[:, left:left + tw]

    # pad
    h, w = arr.shape
    if h < th or w < tw:
        pad_top = (th - h) // 2 if h < th else 0
        pad_bottom = (th - h) - pad_top if h < th else 0
        pad_left = (tw - w) // 2 if w < tw else 0
        pad_right = (tw - w) - pad_left if w < tw else 0
        arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)),
                     mode="constant", constant_values=pad_value)
    return arr.astype(np.float32)


def _ensure_gray2d(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[..., 0]
    raise ValueError(f"Unexpected image ndim: {img.ndim}, shape={img.shape}")


def load_depth_from_mat(mat_path: Path) -> np.ndarray:
    """
    Loads a depth map from a .mat file.
    Tries scipy.io.loadmat (v7) then h5py (v7.3).
    Chooses the 2D array-like variable with largest size.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Depth .mat not found: {mat_path}")

    # 1) Try scipy (MATLAB v7)
    try:
        import scipy.io
        d = scipy.io.loadmat(str(mat_path))
        candidates = []
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                candidates.append((k, v))
        if not candidates:
            raise ValueError("No array candidates via scipy.")
        k_best, v_best = max(candidates, key=lambda kv: np.prod(kv[1].shape))
        arr = np.squeeze(np.array(v_best))
        while arr.ndim > 2:
            arr = np.squeeze(arr[0])
        if arr.ndim != 2:
            raise ValueError(f"Could not reduce '{k_best}' to 2D, got {arr.shape}")
        return arr.astype(np.float32)
    except Exception:
        pass

    # 2) Try h5py (MATLAB v7.3)
    import h5py
    with h5py.File(str(mat_path), "r") as f:
        candidates = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                candidates.append((name, obj))

        f.visititems(visit)
        if not candidates:
            raise ValueError(f"No dataset candidates found in {mat_path} via h5py.")

        name_best, ds_best = max(candidates, key=lambda nd: np.prod(nd[1].shape))
        arr = np.squeeze(np.array(ds_best))
        while arr.ndim > 2:
            arr = np.squeeze(arr[0])
        if arr.ndim != 2:
            raise ValueError(f"Could not reduce '{name_best}' to 2D, got {arr.shape}")

        return arr.astype(np.float32)


def index_split(fringe_dir: Path, depth_dir: Path) -> List[Dict]:
    """
    Pair <stem>.png in fringe_dir with <stem>.mat in depth_dir
    where stem is like: object_name_a0, object_name_a60, ...

    Returns: list of { 'stem', 'fringe', 'depth' }
    """
    if not fringe_dir.exists():
        raise FileNotFoundError(f"Missing fringe dir: {fringe_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth dir: {depth_dir}")

    fringe_files = sorted([p for p in fringe_dir.glob("*.png")])
    if not fringe_files:
        raise RuntimeError(f"No .png found in {fringe_dir}")

    depth_map = {p.stem: p for p in depth_dir.glob("*.mat")}
    items: List[Dict] = []

    missing = []
    for fp in fringe_files:
        stem = fp.stem
        dp = depth_map.get(stem, None)
        if dp is None:
            missing.append(stem)
            continue
        items.append({"stem": stem, "fringe": fp, "depth": dp})

    if not items:
        raise RuntimeError(f"No paired (png, mat) samples found for {fringe_dir} <-> {depth_dir}")

    if missing:
        print(f"[WARN] {len(missing)} fringe files had no matching .mat in {depth_dir} (showing first 10): {missing[:10]}")

    return items


# -----------------------------
# Torch Dataset
# -----------------------------
class FringeToDepthDataset(Dataset):
    """
    Returns:
      x: (1,H,W)
      y: (2,H,W) where y[0]=depth, y[1]=mask
      meta: dict
    """
    def __init__(self, items: List[Dict], input_hw=(960, 960), output_hw=(960, 960)):
        self.items = items
        self.in_h, self.in_w = input_hw
        self.out_h, self.out_w = output_hw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]

        # --- load fringe png ---
        img = imageio.imread(str(meta["fringe"])).astype(np.float32)
        img = _ensure_gray2d(img)
        if np.nanmax(img) > 1.5:
            img = img / 255.0
        img = center_crop_or_pad(img, self.in_h, self.in_w, pad_value=0.0)
        x = img[None, ...]  # (1,H,W)

        # --- load depth mat ---
        depth = load_depth_from_mat(meta["depth"])
        depth = center_crop_or_pad(depth, self.out_h, self.out_w, pad_value=np.nan)

        # mask: finite and >0 (change if your convention differs)
        mask = (np.isfinite(depth) & (depth > 0)).astype(np.float32)

        # numeric depth for loss; NaNs -> 0 but masked out
        depth_num = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

        y = np.stack([depth_num, mask], axis=0)  # (2,H,W)

        return torch.from_numpy(x), torch.from_numpy(y), meta


# -----------------------------
# Checkpoint helper
# -----------------------------
def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "model": state_dict, "optimizer": optimizer.state_dict()}, str(path))


# -----------------------------
# Main
# -----------------------------
def main():
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Index splits (no random split anymore)
    train_items = index_split(train_fringe_dir, train_depth_dir)
    val_items   = index_split(val_fringe_dir,   val_depth_dir)
    test_items  = index_split(test_fringe_dir,  test_depth_dir)

    print("Train:", len(train_items), "Val:", len(val_items), "Test:", len(test_items))

    train_ds = FringeToDepthDataset(train_items, (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    val_ds   = FringeToDepthDataset(val_items,   (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    test_ds  = FringeToDepthDataset(test_items,  (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=(device.type == "cuda"), drop_last=False)

    # Model
    model = UNetFPP(in_ch=960, out_ch=960, p_drop=0.1).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # CSV log
    csv_path = out_path / "training_loss.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "vali_loss", "time_used", "learning_rate", "iteration"])

    iteration = 0
    # (optional) make sure the folder exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(NUM_EPOCH):
        start = time.time()

        model.train()
        train_losses = []
        for x, y, _meta in train_loader:
            x = x.to(device, non_blocking=True)  # (B,1,H,W)
            y = y.to(device, non_blocking=True)  # (B,2,H,W)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)                      # (B,1,H,W)

            loss_val = masked_rmse(y, pred)
            loss_val.backward()
            optimizer.step()

            train_losses.append(float(loss_val.item()))
            iteration += 1

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, _meta in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                val_losses.append(float(masked_rmse(y, pred).item()))

        end = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}, "
            f"Train Loss: {np.mean(train_losses):.6f}, "
            f"Val Loss: {np.mean(val_losses):.6f}, "
            f"Time: {int(np.round(end-start))}s, "
            f"LR: {lr_now:.6f}, "
            f"Iter: {iteration}")

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch + 1,
                        float(np.mean(train_losses)),
                        float(np.mean(val_losses)),
                        int(np.round(end-start)),
                        float(lr_now),
                        int(iteration)])

        # Save every 50 epochs as: checkpoints/ckpt_epoch00050.pth, etc.
        if (epoch + 1) % 50 == 0:
            ckpt_path = checkpoint_dir / f"ckpt_epoch{epoch+1:05d}.pth"
            save_checkpoint(ckpt_path, model, optimizer, epoch + 1)


    # Test
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, y, _meta in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            test_losses.append(float(masked_rmse(y, pred).item()))
    print("test loss:", float(np.mean(test_losses)))


if __name__ == "__main__":
    main()