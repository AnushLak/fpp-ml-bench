"""
UNet Training Script for Fringe Projection Profilometry
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from dataset import FringeFPPDataset
from losses import RMSELoss, MaskedRMSELoss, HybridRMSELoss, L1Loss, MaskedL1Loss, HybridL1Loss
from unet import UNetFPP


def get_loss_function(loss_name, alpha=0.9):
    """Get loss function by name"""
    loss_map = {
        'rmse': RMSELoss(),
        'masked_rmse': MaskedRMSELoss(),
        'hybrid_rmse': HybridRMSELoss(alpha=alpha),
        'l1': L1Loss(),
        'masked_l1': MaskedL1Loss(),
        'hybrid_l1': HybridL1Loss(alpha=alpha),
    }
    if loss_name not in loss_map:
        raise ValueError(f"Unknown loss: {loss_name}. Choose from {list(loss_map.keys())}")
    return loss_map[loss_name]


def train_epoch(model, dataloader, criterion, optimizer, device, start_iteration=0):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    iteration = start_iteration

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for fringe, depth, _ in pbar:
        fringe = fringe.to(device)
        depth = depth.to(device)

        optimizer.zero_grad()
        pred = model(fringe)
        loss = criterion(pred, depth)

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
    """Save model checkpoint"""
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


def main(args):
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("UNet Training - Fringe Projection Profilometry")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Loss function: {args.loss}")
    if args.loss in ['hybrid_rmse', 'hybrid_l1']:
        print(f"Alpha: {args.alpha}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout rate: {args.dropout}")
    print("=" * 70)
    print()

    # Data paths
    data_root = Path(f"/work/arpawar/anushlak/SPIE-PW/training_data_depth{args.dataset_type}")
    train_fringe = data_root / "train" / "fringe"
    train_depth = data_root / "train" / "depth"
    val_fringe = data_root / "val" / "fringe"
    val_depth = data_root / "val" / "depth"

    # Create datasets
    try:
        train_dataset = FringeFPPDataset(train_fringe, train_depth, args.dataset_type)
        val_dataset = FringeFPPDataset(val_fringe, val_depth, args.dataset_type)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    # Create model
    print("Initializing UNet model...")
    model = UNetFPP(
        in_channels=1,
        out_channels=1,
        dropout_rate=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss function
    criterion = get_loss_function(args.loss, args.alpha)

    # Optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=1e-6,
    )

    # Initialize CSV logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"

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
    iteration = 0

    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        iteration = checkpoint.get('iteration', 0)
        print(f"Resuming from epoch {start_epoch}, iteration {iteration}\n")

    # Training loop
    checkpoint_dir = Path("checkpoints")
    print("Starting training...\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, iteration = train_epoch(
            model, train_loader, criterion, optimizer, device, iteration
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

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
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
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
                checkpoint_dir / "best_model.pth"
            )
            print(" ← NEW BEST!")
        else:
            print()

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, iteration,
                checkpoint_dir / f"checkpoint_epoch_{epoch+1:04d}.pth"
            )

        # Early stopping if LR reaches minimum
        if current_lr <= 1e-6:
            print(f"\n⚠ Learning rate reached minimum. Stopping.")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved: {checkpoint_dir / 'best_model.pth'}")
    print(f"Training log saved: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for FPP")
    parser.add_argument("--dataset_type", type=str, default="_individual_normalized",
                        choices=["_raw", "_global_normalized", "_individual_normalized"],
                        help="Dataset normalization type")
    parser.add_argument("--loss", type=str, default="hybrid_l1",
                        choices=["rmse", "masked_rmse", "hybrid_rmse", "l1", "masked_l1", "hybrid_l1"],
                        help="Loss function to use")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Alpha parameter for hybrid losses (0-1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    main(args)
