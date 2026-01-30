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

from unet import UNetFPP, RMSELoss, MaskedRMSELoss, HybridRMSELoss, L1Loss, MaskedL1Loss, HybridL1Loss


# =============================================================================
# Configuration
# =============================================================================
class Config:
    """Training configuration"""
    
    def __init__(self, dataset_type="_raw", loss_type="rmse", alpha=0.9):
        """
        Initialize config with dataset type and loss function
        
        Args:
            dataset_type: One of "_raw", "_global_normalized", "_individual_normalized"
            loss_type: One of "rmse", "masked_rmse", "hybrid_rmse", "l1", "masked_l1", "hybrid_l1"
        
        """
        # Validate dataset type
        valid_types = ["_raw", "_global_normalized", "_individual_normalized"]
        if dataset_type not in valid_types:
            raise ValueError(f"dataset_type must be one of {valid_types}, got {dataset_type}")
        
        # Validate loss type
        valid_losses = ["rmse", "masked_rmse", "hybrid_rmse", "l1", "masked_l1", "hybrid_l1"]
        if loss_type not in valid_losses:
            raise ValueError(f"loss_type must be one of {valid_losses}, got {loss_type}")
        
        self.dataset_type = dataset_type
        self.loss_type = loss_type
        self.alpha = alpha  # for hybrid losses
        
        # Data paths (MODIFY BASE PATH ONLY)
        base_path = Path("/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/training_datasets")
        data_root = base_path / f"training_data_depth{dataset_type}"
        
        self.DATA_ROOT = data_root
        self.TRAIN_FRINGE = data_root / "train" / "fringe"
        self.TRAIN_DEPTH = data_root / "train" / "depth"
        self.VAL_FRINGE = data_root / "val" / "fringe"
        self.VAL_DEPTH = data_root / "val" / "depth"
        self.TEST_FRINGE = data_root / "test" / "fringe"
        self.TEST_DEPTH = data_root / "test" / "depth"
        
        # Model parameters
        self.IN_CHANNELS = 1
        self.OUT_CHANNELS = 1
        self.DROPOUT_RATE = 0.0
        
        # Training hyperparameters
        self.BATCH_SIZE = 2
        self.NUM_EPOCHS = 1000
        self.INITIAL_LR = 1e-4
        self.MIN_LR = 1e-6
        
        # Learning rate scheduler
        self.LR_PATIENCE = 10  # Reduce LR if no improvement for 10 epochs
        self.LR_FACTOR = 0.1   # Multiply LR by 0.1 when reducing
        
        # Device
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.NUM_WORKERS = 1
        
        # Checkpointing (dataset-specific)
        self.CHECKPOINT_DIR = Path(f"checkpoints_{loss_type}_depth{dataset_type}")
        self.SAVE_EVERY = 10  # Save checkpoint every N epochs
        
        # Logging (dataset-specific)
        self.LOG_DIR = Path(f"logs_{loss_type}_depth{dataset_type}")
        self.CSV_LOG_FILE = self.LOG_DIR / "training_log.csv"
        
        # Random seed
        self.SEED = 42
        
        # Depth key for MAT files
        if dataset_type == "_raw":
            self.DEPTH_KEY = "depthMap"
        elif dataset_type == "_global_normalized":
            self.DEPTH_KEY = "depthMapMeters"
        elif dataset_type == "_individual_normalized":
            self.DEPTH_KEY = "depthMapNormalized"


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
        # If needed, normalize raw depth mm → meters
        # depth = depth / 1000.0

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
    # Initialize config with dataset type, loss type, and alpha for hybrid losses
    config = Config(dataset_type=args.dataset_type, loss_type=args.loss, alpha=args.alpha)
    
    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    print("=" * 70)
    print("UNet Training - Fringe Projection Profilometry")
    print("=" * 70)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Loss function: {args.loss}")
    print(f"Alpha (for hybrid losses): {args.alpha}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Checkpoint dir: {config.CHECKPOINT_DIR}")
    print(f"Log dir: {config.LOG_DIR}")
    print(f"Depth MAT key: {config.DEPTH_KEY}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Initial LR: {config.INITIAL_LR}")
    print(f"Min LR: {config.MIN_LR}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("=" * 70)
    print()
    
    # Create datasets
    try:
        train_dataset = FringeFPPDatasetPNG(
            config.TRAIN_FRINGE, 
            config.TRAIN_DEPTH,
            depth_key=config.DEPTH_KEY
        )
        val_dataset = FringeFPPDatasetPNG(
            config.VAL_FRINGE, 
            config.VAL_DEPTH,
            depth_key=config.DEPTH_KEY
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nExpected file structure:")
        print(f"  {config.TRAIN_FRINGE}/object_001_a000.png")
        print(f"  {config.TRAIN_DEPTH}/object_001_a000.mat")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda")
    )
    
    # Create model
    model = UNetFPP(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss function - select based on config
    loss_functions = {
        "rmse": RMSELoss(),
        "masked_rmse": MaskedRMSELoss(),
        "hybrid_rmse": HybridRMSELoss(alpha=config.alpha),
        "l1": L1Loss(),
        "masked_l1": MaskedL1Loss(),
        "hybrid_l1": HybridL1Loss(alpha=config.alpha),
    }
    
    criterion = loss_functions[config.loss_type]
    print(f"Using loss function: {config.loss_type}\n")
    
    # Optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.INITIAL_LR,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
        min_lr=config.MIN_LR,
    )
    
    # Initialize CSV logging
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.CSV_LOG_FILE
    
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
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        iteration = checkpoint.get('iteration', 0)  # Resume iteration count
        print(f"Resuming from epoch {start_epoch}, iteration {iteration}\n")
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train (now returns iteration count)
        train_loss, iteration = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, iteration
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
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
        print(f"Epoch [{epoch+1:3d}/{config.NUM_EPOCHS}] "
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
                config.CHECKPOINT_DIR / "best_model.pth"
            )
            print(" ← NEW BEST!")
        else:
            print()
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, iteration,
                config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth"
            )
        
        # Early stopping if LR reaches minimum
        if current_lr <= config.MIN_LR:
            print(f"\n⚠ Learning rate reached minimum ({config.MIN_LR}). Stopping.")
            break
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Training log saved: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for FPP")
    parser.add_argument("--dataset_type", type=str, default="_global_normalized",
                        choices=["_raw", "_global_normalized", "_individual_normalized"],
                        help="Dataset type to use for training")
    parser.add_argument("--loss", type=str, default="rmse",
                        choices=["rmse", "masked_rmse", "hybrid_rmse", "l1", "masked_l1", "hybrid_l1"],
                        help="Loss function to use for training")
    parser.add_argument("--alpha", type=float, default=0.9,
                    help="Alpha parameter for hybrid loss functions (default: 0.9)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()
    
    main(args)