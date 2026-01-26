"""
Training Script for UNet FPP - Optimized for PNG Depth Maps

Uses your existing _normalized_depth.png files directly!
No need to regenerate data.

Expected structure:
    data/
        train/
            fringe/  object_001_a000.png, object_002_a000.png, ...
            depth/   object_001_a000_normalized_depth.png, ...
        val/
            fringe/
            depth/
        test/
            fringe/
            depth/
"""

import os
import time
import csv
import argparse
from pathlib import Path
import scipy.io as sio

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm

from unet import UNetFPP, RMSELoss, MaskedRMSELoss, HybridRMSELoss, L1Loss, MaskedL1Loss, HybridL1Loss, HybridMaskedRMSEWithMaskedL1


# =============================================================================
# Configuration
# =============================================================================
class Config:
    """Training configuration"""
    # Data paths (MODIFY THESE)
    DATA_ROOT = Path("/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_training_data_depth_raw")
    TRAIN_FRINGE = DATA_ROOT / "train" / "fringe"
    TRAIN_DEPTH = DATA_ROOT / "train" / "depth"
    VAL_FRINGE = DATA_ROOT / "val" / "fringe"
    VAL_DEPTH = DATA_ROOT / "val" / "depth"
    TEST_FRINGE = DATA_ROOT / "test" / "fringe"
    TEST_DEPTH = DATA_ROOT / "test" / "depth"
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    DROPOUT_RATE = 0.0
    
    # Training hyperparameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 1000
    INITIAL_LR = 1e-4
    MIN_LR = 1e-6
    
    # Learning rate scheduler
    LR_PATIENCE = 10  # Reduce LR if no improvement for 10 epochs
    LR_FACTOR = 0.1   # Multiply LR by 0.1 when reducing
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 1
    
    # Checkpointing
    CHECKPOINT_DIR = Path("checkpoints_MaskedRMSELoss")
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    
    # Logging
    LOG_DIR = Path("logs_MaskedRMSELoss")
    CSV_LOG_FILE = LOG_DIR / "training_log.csv"
    
    # Random seed
    SEED = 42


# =============================================================================
# Dataset for PNG Depth Maps
# =============================================================================
class FringeFPPDatasetPNG(Dataset):
    """Dataset using depth maps organized by fpp_dataset_preparer.py"""
    
    def __init__(self, fringe_dir: Path, depth_dir: Path, depth_key="depthMap"):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.depth_key = depth_key
        
        # Get all fringe images (PNG only)
        self.fringe_files = sorted(list(self.fringe_dir.glob("*.png")))
        
        if not self.fringe_files:
            raise ValueError(f"No PNG files found in {fringe_dir}")
        
        # Find matching depth PNGs
        self.pairs = []
        missing = []
        
        for fringe_path in self.fringe_files:
            stem = fringe_path.stem  # e.g., "wooden_board_A60", "vial_A120"

            # fpp_dataset_preparer.py copies files with SAME base name
            depth_path = self.depth_dir / f"{stem}.mat"

            if depth_path.exists():
                self.pairs.append((fringe_path, depth_path))
            else:
                missing.append(stem)

        if not self.pairs:
            for f in self.fringe_files[:5]:
                print(f"  {f.name}")
            print(f"\nFirst few files in depth directory:")
            depth_files = sorted(list(self.depth_dir.glob("*.png")))[:5]
            for f in depth_files:
                print(f"  {f.name}")
            raise ValueError(f"No matching depth files found in {depth_dir}")
        
        if missing:
            print(f"⚠ Warning: {len(missing)}/{len(self.fringe_files)} fringe images have no depth match")
            if len(missing) <= 10:
                print(f"  Missing depth files for: {missing}")
        
        print(f"✓ Loaded {len(self.pairs)} sample pairs from {fringe_dir}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        fringe_path, depth_path = self.pairs[idx]
        
        # Load fringe pattern (grayscale)
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32)
        
        # Normalize to [0, 1]
        fringe = fringe / 255.0
        
        # Load depth from mat
        mat = sio.loadmat(depth_path)

        if self.depth_key not in mat:
            raise KeyError(
                f"Key '{self.depth_key}' not found in {depth_path.name}. "
                f"Available keys: {list(mat.keys())}"
            )

        depth = mat[self.depth_key]

        # Ensure shape (H, W)
        if depth.ndim != 2:
            raise ValueError(
                f"Depth must be 2D, got shape {depth.shape} in {depth_path}"
            )

        # Convert double → float32
        depth = depth.astype(np.float32)
        # Normalize depth mm → meters
        depth = depth / 1000.0

        # Shape check
        if fringe.shape != depth.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"  Fringe: {fringe.shape} ({fringe_path})\n"
                f"  Depth:  {depth.shape} ({depth_path})"
            )

        # Convert to tensors
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # (1, H, W)
        depth  = torch.from_numpy(depth).unsqueeze(0)   # (1, H, W)

        return fringe, depth, str(fringe_path.name)


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, start_iteration=0):
    """Train for one epoch and return loss + final iteration count"""
    model.train()
    running_loss = 0.0
    iteration = start_iteration
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for fringe, depth, _ in pbar:
        fringe = fringe.to(device)
        depth = depth.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(fringe)
        loss = criterion(pred, depth)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        iteration += 1
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return running_loss / len(dataloader), iteration


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for fringe, depth, _ in pbar:
            fringe = fringe.to(device)
            depth = depth.to(device)
            
            pred = model(fringe)
            loss = criterion(pred, depth)
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return running_loss / len(dataloader)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, iteration, path):
    """Save model checkpoint with iteration count"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'iteration': iteration,
    }
    
    torch.save(checkpoint, path)


# =============================================================================
# Main Training Loop
# =============================================================================
def main(args):
    # Set seed for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    print("=" * 70)
    print("UNet Training - Fringe Projection Profilometry")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Initial LR: {Config.INITIAL_LR}")
    print(f"Min LR: {Config.MIN_LR}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Using PNG depth maps (uint16 → float32 / 65535)")
    print("=" * 70)
    print()
    
    # Create datasets
    try:
        train_dataset = FringeFPPDatasetPNG(Config.TRAIN_FRINGE, Config.TRAIN_DEPTH)
        val_dataset = FringeFPPDatasetPNG(Config.VAL_FRINGE, Config.VAL_DEPTH)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nExpected file structure:")
        print("  data/train/fringe/object_001_a000.png")
        print("  data/train/depth/object_001_a000_depth.mat")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda")
    )
    
    # Create model
    model = UNetFPP(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        dropout_rate=Config.DROPOUT_RATE
    ).to(Config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Loss function
    criterion = MaskedRMSELoss()
    
    # Optimizer
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=Config.INITIAL_LR)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=Config.INITIAL_LR,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR,
    )
    
    # Initialize CSV logging
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Config.CSV_LOG_FILE
    
    # Create CSV file with headers if it doesn't exist
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'time_seconds', 'learning_rate', 'iteration'])
        print(f"Created CSV log: {csv_path}\n")
    else:
        print(f"Appending to existing CSV log: {csv_path}\n")
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    iteration = 0  # Track total iterations across all epochs
    
    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        iteration = checkpoint.get('iteration', 0)  # Resume iteration count
        print(f"Resuming from epoch {start_epoch}, iteration {iteration}\n")
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train (now returns iteration count)
        train_loss, iteration = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, iteration
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                float(train_loss),
                float(val_loss),
                int(round(epoch_time)),
                float(current_lr),
                int(iteration)
            ])
        
        # Print epoch summary
        print(f"Epoch [{epoch+1:3d}/{Config.NUM_EPOCHS}] "
              f"({epoch_time:5.1f}s) | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Iter: {iteration}", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, iteration,
                Config.CHECKPOINT_DIR / "best_model.pth"
            )
            print(" ← NEW BEST!")
        else:
            print()
        
        # Save periodic checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, iteration,
                Config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth"
            )
        
        # Early stopping if LR reaches minimum
        if current_lr <= Config.MIN_LR:
            print(f"\n⚠ Learning rate reached minimum ({Config.MIN_LR}). Stopping.")
            break
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Training log saved: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for FPP")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()
    
    main(args)