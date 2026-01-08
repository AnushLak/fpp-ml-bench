# -*- coding: utf-8 -*-
"""
PyTorch train.py for FPP synthetic dataset structure

Input:
  in_dir = fpp_synthetic_dataset/<object_name>/A{angle}/A_fringestep.../(2.png or *.png)

Target:
  out_dir = data/depth_images/<object-name>-a{angle}.mat

Checkpoints:
  ./checkpoints
"""

import os
import re
import time, csv
import random
import gc
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import imageio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_torch import UNetFPP, masked_rmse  # <- your PyTorch model + loss


# -----------------------------
# Repro / device
# -----------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Config (YOU SET THESE)
# -----------------------------
INPUT_HEIGHT = 960
INPUT_WIDTH  = 960
OUTPUT_HEIGHT = 960
OUTPUT_WIDTH  = 960

BATCH_SIZE = 4
NUM_EPOCH = 700

# Your new structure
in_dir = r"fpp_synthetic_dataset"          # root containing object_name folders
out_dir = r"data\depth_images"            # folder containing .mat depth files
out_path = r"results"                     # where training_loss.csv + optional plots go

checkpoint_dir = r"./checkpoints"         # as requested


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
    Chooses the first 2D array-like variable with largest size.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Depth .mat not found: {mat_path}")

    # 1) Try scipy (MATLAB v7)
    try:
        import scipy.io
        d = scipy.io.loadmat(str(mat_path))
        # ignore meta keys
        candidates = []
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and v.ndim >= 2:
                candidates.append((k, v))
        if not candidates:
            raise ValueError(f"No array candidates found in {mat_path} via scipy.")
        # pick largest 2D-ish
        k_best, v_best = max(candidates, key=lambda kv: np.prod(kv[1].shape))
        arr = np.array(v_best)
        # squeeze to 2D if possible
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            # if still not 2D, try taking first slice
            while arr.ndim > 2:
                arr = arr[0]
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Could not reduce mat var '{k_best}' to 2D. Got shape {arr.shape}.")
        return arr.astype(np.float32)
    except Exception:
        pass

    # 2) Try h5py (MATLAB v7.3)
    try:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            # gather datasets
            candidates = []
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    if len(shape) >= 2:
                        candidates.append((name, obj))
            f.visititems(visit)
            if not candidates:
                raise ValueError(f"No dataset candidates found in {mat_path} via h5py.")
            name_best, ds_best = max(candidates, key=lambda nd: np.prod(nd[1].shape))
            arr = np.array(ds_best)
            arr = np.squeeze(arr)
            # MATLAB stores as column-major; sometimes transpose needed.
            # We'll return as-is; if your maps look rotated, transpose here.
            if arr.ndim != 2:
                while arr.ndim > 2:
                    arr = arr[0]
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Could not reduce mat dataset '{name_best}' to 2D. Got shape {arr.shape}.")
            return arr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed loading .mat depth from {mat_path}. Error: {e}")


def resolve_depth_mat(out_dir: Path, object_folder_name: str, angle: int) -> Path:
    """
    Tries a few filename conventions:
      <object>-a{angle}.mat
      <object>_a{angle}.mat
    where <object> may be folder name or folder name with '_'->'-'
    """
    out_dir = Path(out_dir)

    obj1 = object_folder_name
    obj2 = object_folder_name.replace("_", "-")

    candidates = [
        out_dir / f"{obj2}-a{angle}.mat",
        out_dir / f"{obj1}-a{angle}.mat",
        out_dir / f"{obj2}_a{angle}.mat",
        out_dir / f"{obj1}_a{angle}.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    # If none exist, still return the most likely one (better error message upstream)
    return candidates[0]


FRINGE_RE = re.compile(r"^A_fringe(\d+)\.png$", re.IGNORECASE)

def find_fringe_images(path: Path) -> List[Path]:
    """
    Returns ALL fringe images matching A_fringe#.png under:
      - a directory (direct children)
      - or a single file path

    This supports both layouts:
      A{angle}/A_fringestep.../A_fringe#.png
      A{angle}/A_fringe#.png
    """
    path = Path(path)

    if path.is_file():
        return [path] if FRINGE_RE.match(path.name) else []

    if path.is_dir():
        imgs = [p for p in path.iterdir() if p.is_file() and FRINGE_RE.match(p.name)]
        # sort by the numeric # in A_fringe#.png so ordering is stable
        imgs.sort(key=lambda p: int(FRINGE_RE.match(p.name).group(1)))
        return imgs

    return []


# -----------------------------
# Dataset indexing
# -----------------------------
ANGLE_RE = re.compile(r"^A(\d+)$", re.IGNORECASE)

def index_fpp_dataset(in_dir: Path, out_dir: Path) -> List[Dict]:
    """
    Builds samples:
      fpp_synthetic_dataset/<object>/A{angle}/.../(A_fringe#.png)

    Each A_fringe#.png is treated as ONE sample:
      meta = { object, angle, sample, fringe_img, depth_mat }
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    samples: List[Dict] = []
    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir not found: {out_dir}")

    object_folders = sorted([p for p in in_dir.iterdir() if p.is_dir()])

    for obj_path in object_folders:
        obj_name = obj_path.name

        angle_folders = sorted([p for p in obj_path.iterdir()
                                if p.is_dir() and ANGLE_RE.match(p.name)])
        for ang_path in angle_folders:
            angle = int(ANGLE_RE.match(ang_path.name).group(1))
            depth_mat = resolve_depth_mat(out_dir, obj_name, angle)

            # Look inside A{angle}: could contain step folders, or direct A_fringe#.png files
            entries = sorted([p for p in ang_path.iterdir() if p.is_dir() or p.is_file()])

            for e in entries:
                # Collect fringe images from either a dir or a file
                fringe_imgs = find_fringe_images(e)
                if not fringe_imgs:
                    continue

                for fi in fringe_imgs:
                    # make a unique sample name
                    # e.g., "A_fringestep12__A_fringe3" or "A_fringe3"
                    parent_tag = e.name if e.is_dir() else ""
                    stem_tag = fi.stem
                    sample_name = f"{parent_tag}__{stem_tag}" if parent_tag else stem_tag

                    samples.append({
                        "object": obj_name,
                        "angle": angle,
                        "sample": sample_name,
                        "fringe_img": fi,
                        "depth_mat": depth_mat,
                    })

    if not samples:
        raise RuntimeError(
            f"No samples found. Expected A_fringe#.png under: {in_dir}\\<object>\\A<angle>\\..."
        )
    return samples


# -----------------------------
# Torch Dataset
# -----------------------------
class FPPFringeDepthDataset(Dataset):
    """
    y has 2 channels:
      y[0] = depth
      y[1] = mask  (1=valid, 0=background)
    """
    def __init__(self, items: List[Dict], input_hw=(960, 960), output_hw=(960, 960)):
        self.items = items
        self.in_h, self.in_w = input_hw
        self.out_h, self.out_w = output_hw

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]

        # Load fringe image
        img = imageio.imread(str(meta["fringe_img"])).astype(np.float32)
        img = _ensure_gray2d(img)

        # Normalize if looks like 0..255
        if np.nanmax(img) > 1.5:
            img = img / 255.0

        img = center_crop_or_pad(img, self.in_h, self.in_w, pad_value=0.0)
        x = img[None, ...]  # (1,H,W)

        # Load depth from .mat (shared per object-angle)
        depth = load_depth_from_mat(meta["depth_mat"])
        depth = center_crop_or_pad(depth, self.out_h, self.out_w, pad_value=np.nan)

        # Build mask from depth
        # (common convention: valid pixels are finite and > 0)
        mask = np.isfinite(depth) & (depth > 0)
        mask = mask.astype(np.float32)

        # replace NaNs in depth with 0 so model sees numeric targets; mask controls loss
        depth_num = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

        y = np.stack([depth_num, mask], axis=0)  # (2,H,W)

        return torch.from_numpy(x), torch.from_numpy(y), meta


# -----------------------------
# Train / Eval
# -----------------------------
def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({"epoch": epoch, "model": state_dict, "optimizer": optimizer.state_dict()}, str(path))


def main():
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Index all samples
    all_items = index_fpp_dataset(Path(in_dir), Path(out_dir))

    # Split (70/15/15) like the TF version
    n = len(all_items)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    idxs = list(range(n))
    random.shuffle(idxs)

    train_items = [all_items[i] for i in idxs[:n_train]]
    val_items   = [all_items[i] for i in idxs[n_train:n_train + n_val]]
    test_items  = [all_items[i] for i in idxs[n_train + n_val:]]

    print(" Total number of samples:", n,
          "\n Number of training samples", len(train_items),
          "\n Number of validating samples", len(val_items),
          "\n Number of testing samples", len(test_items))

    train_ds = FPPFringeDepthDataset(train_items, (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    val_ds   = FPPFringeDepthDataset(val_items,   (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))
    test_ds  = FPPFringeDepthDataset(test_items,  (INPUT_HEIGHT, INPUT_WIDTH), (OUTPUT_HEIGHT, OUTPUT_WIDTH))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              pin_memory=(device.type == "cuda"), drop_last=False)

    # Model: your 3-downsample UNet (960->480->240->120)
    model = UNetFPP(in_ch=1, out_ch=1, p_drop=0.1).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # CSV log
    csv_path = Path(out_path) / "training_loss.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "vali_loss", "time_used", "learning_rate", "iteration"])

    iteration = 0
    ckpt_path = Path(checkpoint_dir) / "ckpt.pt"

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(NUM_EPOCH):
        start = time.time()
        model.train()
        train_losses = []

        for x, y, _meta in train_loader:
            x = x.to(device, non_blocking=True)      # (B,1,H,W)
            y = y.to(device, non_blocking=True)      # (B,2,H,W)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)                           # (B,1,H,W)
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
              f"Vali Loss: {np.mean(val_losses):.6f}, "
              f"Time used: {int(np.round(end-start))}s, "
              f"Learning rate: {lr_now:.6f}, iteration: {iteration}")

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch + 1,
                        float(np.mean(train_losses)),
                        float(np.mean(val_losses)),
                        int(np.round(end-start)),
                        float(lr_now),
                        int(iteration)])

        # checkpoint cadence (matches your TF script)
        if epoch % 50 == 0:
            save_checkpoint(ckpt_path, model, optimizer, epoch)

    # -----------------------------
    # Test
    # -----------------------------
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