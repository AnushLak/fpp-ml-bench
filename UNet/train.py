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
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm

from unet import UNetFPP, MaskedRMSELoss


# =============================================================================
# Configuration
# =============================================================================
class Config:
    """Training configuration"""
    # Data paths (MODIFY THESE)
    DATA_ROOT = Path("/home/oadam/workspace/fpp/fpp_synthetic_dataset/fpp_unet_training_data_normalized_depth")
    TRAIN_FRINGE = DATA_ROOT / "train" / "fringe"
    TRAIN_DEPTH = DATA_ROOT / "train" / "depth"
    VAL_FRINGE = DATA_ROOT / "val" / "fringe"
    VAL_DEPTH = DATA_ROOT / "val" / "depth"
    TEST_FRINGE = DATA_ROOT / "test" / "fringe"
    TEST_DEPTH = DATA_ROOT / "test" / "depth"
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    DROPOUT_RATE = 0.5
    
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
    NUM_WORKERS = 4
    
    # Checkpointing
    CHECKPOINT_DIR = Path("checkpoints")
    SAVE_EVERY = 10  # Save checkpoint every N epochs
    
    # Random seed
    SEED = 42


# =============================================================================
# Dataset for PNG Depth Maps
# =============================================================================
class FringeFPPDatasetPNG(Dataset):
    """
    Dataset using captured fringe PNG and PNG depth maps organized by fpp_dataset_preparer.py
    
    File naming convention (same base name for both):
        fringe/wooden_board_A60.png → depth/wooden_board_A60.png
        fringe/vial_A120.png → depth/vial_A120.png
    
    The PNG depth maps should already be normalized to uint16 range [0, 65535]
    where 0 = background and (0, 65535] = normalized depth
    """
    
    def __init__(self, fringe_dir: Path, depth_dir: Path):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        
        # Get all fringe images (PNG only)
        self.fringe_files = sorted(list(self.fringe_dir.glob("*.png")))
        
        if not self.fringe_files:
            raise ValueError(f"No PNG files found in {fringe_dir}")
        
        # Find matching normalized depth PNGs
        self.pairs = []
        missing = []
        
        for fringe_path in self.fringe_files:
            stem = fringe_path.stem  # e.g., "wooden_board_A60", "vial_A120"

            # fpp_dataset_preparer.py copies files with SAME base name
            # fringe/wooden_board_A60.png → depth/wooden_board_A60.png
            depth_path = self.depth_dir / f"{stem}.png"

            if depth_path.exists():
                self.pairs.append((fringe_path, depth_path))
            else:
                missing.append(stem)

        if not self.pairs:
            print(f"❌ No matching depth files found!")
            print(f"\nExpected pattern:")
            print(f"  Fringe: {self.fringe_dir}/wooden_board_A60.png")
            print(f"  Depth:  {self.depth_dir}/wooden_board_A60.png")
            print(f"\nFirst few fringe files found:")
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
                print(f"  Missing: {missing}")
        
        print(f"✓ Loaded {len(self.pairs)} sample pairs from {fringe_dir}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        fringe_path, depth_path = self.pairs[idx]
        
        # Load fringe pattern (grayscale)
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32)
        
        # Normalize to [0, 1] if needed
        if fringe.max() > 1.5:
            fringe = fringe / 255.0
        
        # Load normalized depth from PNG (uint16 → float32)
        depth_uint16 = np.array(Image.open(depth_path))
        
        # Convert uint16 [0, 65535] to float32 [0, 1]
        depth = depth_uint16.astype(np.float32) / 65535.0
        
        # Verify shape consistency
        if fringe.shape != depth.shape:
            raise ValueError(
                f"Shape mismatch: fringe {fringe.shape} vs depth {depth.shape}\n"
                f"  Fringe: {fringe_path}\n"
                f"  Depth:  {depth_path}"
            )
        
        # Convert to tensors (add channel dimension)
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # (1, H, W)
        depth = torch.from_numpy(depth).unsqueeze(0)    # (1, H, W)
        
        return fringe, depth, str(fringe_path.name)


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
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
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return running_loss / len(dataloader)


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


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
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
        print("  data/train/depth/object_001_a000_normalized_depth.png")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.INITIAL_LR)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.LR_FACTOR,
        patience=Config.LR_PATIENCE,
        min_lr=Config.MIN_LR,
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f"Resuming from epoch {start_epoch}\n")
    
    # Training loop
    print("Starting training...\n")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch [{epoch+1:3d}/{Config.NUM_EPOCHS}] "
              f"({epoch_time:5.1f}s) | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}", end="")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                Config.CHECKPOINT_DIR / "best_model.pth"
            )
            print(" ← NEW BEST!")
        else:
            print()
        
        # Save periodic checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
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
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for FPP")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()
    
    main(args)
