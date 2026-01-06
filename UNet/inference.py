import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet import UNet
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
    """Dataset for fringe images - one fringe image per object"""
    def __init__(self, fringe_dir, depth_dir):
        self.fringe_dir = fringe_dir
        self.depth_dir = depth_dir

        # Get sorted list of files
        self.fringe_files = sorted([f for f in os.listdir(fringe_dir)
                                    if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.depth_files = sorted([f for f in os.listdir(depth_dir)
                                   if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        assert len(self.fringe_files) == len(self.depth_files), \
            f"Mismatch: {len(self.fringe_files)} fringe images vs {len(self.depth_files)} depth maps"

    def __len__(self):
        return len(self.fringe_files)

    def __getitem__(self, idx):
        # Load fringe image
        fringe_path = os.path.join(self.fringe_dir, self.fringe_files[idx])
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32) / 255.0
        fringe = torch.from_numpy(fringe).unsqueeze(0)

        # Load depth map
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        depth = Image.open(depth_path).convert('L')
        depth = np.array(depth, dtype=np.float32) / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0)

        return fringe, depth, self.fringe_files[idx]

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
# INFERENCE CONFIGURATION
# ============================================================================
class Config:
    # Data paths
    VAL_FRINGE_DIR = "data/val/fringe"
    VAL_DEPTH_DIR = "data/val/depth"
    TEST_FRINGE_DIR = "data/test/fringe"
    TEST_DEPTH_DIR = "data/test/depth"

    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    DROPOUT_RATE = 0.5

    # Inference parameters
    BATCH_SIZE = 2
    CHECKPOINT_PATH = "checkpoints/best_model.pth"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    RESULTS_FILE = "inference_results.json"

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================
def evaluate_dataset(model, dataloader, criterion, device, dataset_name):
    """
    Evaluate model on a dataset

    Args:
        model: UNet model
        dataloader: DataLoader for the dataset
        criterion: Loss function (RMSE)
        device: Device to run inference on
        dataset_name: Name of the dataset (for logging)

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    running_loss = 0.0
    sample_losses = []

    print(f"\nEvaluating on {dataset_name} dataset...")
    print("-" * 70)

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"Inference ({dataset_name})")
        for fringe, depth_gt, filenames in loop:
            fringe = fringe.to(device)
            depth_gt = depth_gt.to(device)

            # Forward pass
            depth_pred = model(fringe)

            # Compute loss
            loss = criterion(depth_pred, depth_gt)

            # Per-sample loss
            for i in range(fringe.size(0)):
                sample_loss = criterion(
                    depth_pred[i:i+1],
                    depth_gt[i:i+1]
                ).item()
                sample_losses.append({
                    "filename": filenames[i],
                    "rmse": sample_loss
                })

            running_loss += loss.item()
            loop.set_postfix(rmse=loss.item())

    # Calculate statistics
    avg_loss = running_loss / len(dataloader)
    rmse_values = [s["rmse"] for s in sample_losses]

    results = {
        "dataset": dataset_name,
        "num_samples": len(sample_losses),
        "avg_rmse": avg_loss,
        "min_rmse": min(rmse_values),
        "max_rmse": max(rmse_values),
        "std_rmse": np.std(rmse_values),
        "sample_losses": sample_losses
    }

    print(f"\nResults for {dataset_name}:")
    print(f"  Number of samples: {results['num_samples']}")
    print(f"  Average RMSE: {results['avg_rmse']:.6f}")
    print(f"  Min RMSE: {results['min_rmse']:.6f}")
    print(f"  Max RMSE: {results['max_rmse']:.6f}")
    print(f"  Std RMSE: {results['std_rmse']:.6f}")

    return results

# ============================================================================
# MAIN
# ============================================================================
def main():
    config = Config()

    print("=" * 70)
    print("UNet Inference - Fringe Projection Profilometry")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {config.CHECKPOINT_PATH}")

    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {config.CHECKPOINT_PATH}")
        return

    model = UNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # Load checkpoint
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded successfully!")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"  Best loss: {checkpoint['loss']:.6f}")
    print(f"Device: {config.DEVICE}")

    # Loss function
    criterion = RMSELoss()

    # Results dictionary
    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": config.CHECKPOINT_PATH,
        "trained_epochs": checkpoint['epoch'] + 1,
        "datasets": {}
    }

    # Evaluate on validation set
    if os.path.exists(config.VAL_FRINGE_DIR) and os.path.exists(config.VAL_DEPTH_DIR):
        val_dataset = FringeDataset(config.VAL_FRINGE_DIR, config.VAL_DEPTH_DIR)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.DEVICE == "cuda" else False
        )

        val_results = evaluate_dataset(
            model, val_loader, criterion, config.DEVICE, "validation"
        )
        all_results["datasets"]["validation"] = val_results
    else:
        print(f"\nWarning: Validation dataset not found")
        print(f"  Fringe dir: {config.VAL_FRINGE_DIR}")
        print(f"  Depth dir: {config.VAL_DEPTH_DIR}")

    # Evaluate on test set
    if os.path.exists(config.TEST_FRINGE_DIR) and os.path.exists(config.TEST_DEPTH_DIR):
        test_dataset = FringeDataset(config.TEST_FRINGE_DIR, config.TEST_DEPTH_DIR)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.DEVICE == "cuda" else False
        )

        test_results = evaluate_dataset(
            model, test_loader, criterion, config.DEVICE, "test"
        )
        all_results["datasets"]["test"] = test_results
    else:
        print(f"\nWarning: Test dataset not found")
        print(f"  Fringe dir: {config.TEST_FRINGE_DIR}")
        print(f"  Depth dir: {config.TEST_DEPTH_DIR}")

    # Save results to JSON
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 70)
    print("Inference completed!")
    print("=" * 70)
    print(f"Results saved to: {config.RESULTS_FILE}")

    # Summary
    print("\nSummary:")
    for dataset_name, results in all_results["datasets"].items():
        print(f"  {dataset_name.capitalize()}: RMSE = {results['avg_rmse']:.6f}")

if __name__ == "__main__":
    main()
