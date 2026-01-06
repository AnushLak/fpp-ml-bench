import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from hformer import Hformer
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============================================================================
# DATASET CLASS
# ============================================================================
class FringeDataset(Dataset):
    """
    Dataset for fringe images - one fringe image per object
    """
    def __init__(self, fringe_dir, depth_dir):
        """
        Args:
            fringe_dir: Directory containing fringe images
            depth_dir: Directory containing corresponding depth maps
        """
        self.fringe_dir = fringe_dir
        self.depth_dir = depth_dir

        # Get sorted list of files
        self.fringe_files = sorted([f for f in os.listdir(fringe_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.depth_files = sorted([f for f in os.listdir(depth_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        assert len(self.fringe_files) == len(self.depth_files), \
            f"Mismatch: {len(self.fringe_files)} fringe images vs {len(self.depth_files)} depth maps"

        print(f"Loaded {len(self.fringe_files)} samples from {fringe_dir}")

    def __len__(self):
        return len(self.fringe_files)

    def __getitem__(self, idx):
        # Load fringe image (grayscale)
        fringe_path = os.path.join(self.fringe_dir, self.fringe_files[idx])
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        fringe = torch.from_numpy(fringe).unsqueeze(0)  # Add channel dimension

        # Load depth map (grayscale)
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = Image.open(depth_path).convert('L')
        depth = np.array(depth, dtype=np.float32) / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0)

        return fringe, depth

# ============================================================================
# RMSE LOSS FUNCTION
# ============================================================================
class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
class Config:
    # Data paths
    TRAIN_FRINGE_DIR = "data/train/fringe"
    TRAIN_DEPTH_DIR = "data/train/depth"
    VAL_FRINGE_DIR = "data/val/fringe"
    VAL_DEPTH_DIR = "data/val/depth"

    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    DROPOUT_RATE = 0.5

    # Training hyperparameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 600
    LEARNING_RATE = 1e-4

    # RMSProp parameters
    ALPHA = 0.99
    MOMENTUM = 0.0
    WEIGHT_DECAY = 1e-8

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint and logging
    CHECKPOINT_DIR = "checkpoints"
    LOG_FILE = "training_log.json"
    SAVE_EVERY = 5

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()  # Enable dropout and batch norm training mode
    running_loss = 0.0
    batch_losses = []

    loop = tqdm(dataloader, desc="Training", leave=False)
    for fringe, depth_gt in loop:
        fringe = fringe.to(device)
        depth_gt = depth_gt.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        depth_pred = model(fringe)
        loss = criterion(depth_pred, depth_gt)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        batch_loss = loss.item()
        running_loss += batch_loss
        batch_losses.append(batch_loss)

        loop.set_postfix(loss=batch_loss)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, batch_losses

def validate(model, dataloader, criterion, device):
    """Validation loop"""
    model.eval()  # Disable dropout and set batch norm to eval mode
    running_loss = 0.0
    batch_losses = []

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for fringe, depth_gt in loop:
            fringe = fringe.to(device)
            depth_gt = depth_gt.to(device)

            # Forward pass
            depth_pred = model(fringe)
            loss = criterion(depth_pred, depth_gt)

            # Track loss
            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)

            loop.set_postfix(loss=batch_loss)

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, batch_losses

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def log_to_json(log_file, log_data):
    """Append training log to JSON file"""
    # Load existing log if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    # Append new log entry
    logs.append(log_data)

    # Save to file
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    config = Config()

    # Create datasets
    print("=" * 70)
    print("Hformer Training - Fringe Projection Profilometry")
    print("=" * 70)
    print("\nLoading datasets...")

    train_dataset = FringeDataset(
        config.TRAIN_FRINGE_DIR,
        config.TRAIN_DEPTH_DIR
    )
    val_dataset = FringeDataset(
        config.VAL_FRINGE_DIR,
        config.VAL_DEPTH_DIR
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False
    )

    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing Hformer model...")
    print("=" * 70)

    model = Hformer(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # RMSProp optimizer
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=config.LEARNING_RATE,
        alpha=config.ALPHA,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # RMSE Loss
    criterion = RMSELoss()

    # Training configuration summary
    print(f"\nDevice: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Dropout rate: {config.DROPOUT_RATE}")
    print(f"Optimizer: RMSProp (alpha={config.ALPHA}, momentum={config.MOMENTUM})")
    print(f"Loss function: RMSE")

    # Initialize training log
    training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTraining started at: {training_start_time}")

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 70)

        # Train
        train_loss, train_batch_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        print(f"Train RMSE: {train_loss:.6f}")

        # Validate
        val_loss, val_batch_losses = validate(
            model, val_loader, criterion, config.DEVICE
        )
        print(f"Val RMSE:   {val_loss:.6f}")

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != current_lr:
            print(f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")

        # Log to JSON
        log_entry = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": new_lr,
            "train_batch_losses": train_batch_losses,
            "val_batch_losses": val_batch_losses
        }
        log_to_json(config.LOG_FILE, log_entry)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config.CHECKPOINT_DIR}/best_model.pth"
            )
            print(f"★ New best model! Val RMSE: {best_val_loss:.6f}")

        # Periodic checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{config.CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth"
            )

    # Training complete
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Best validation RMSE: {best_val_loss:.6f}")
    print(f"Training log saved to: {config.LOG_FILE}")
    print(f"Best model saved to: {config.CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    main()
