"""
Inference Script for UNet FPP
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from unet import UNetFPP, MaskedRMSELoss
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Data paths (MODIFY THESE)
    DATA_ROOT = Path("/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_unet_training_data_normalized_depth")
    TRAIN_FRINGE = DATA_ROOT / "train" / "fringe"
    TRAIN_DEPTH = DATA_ROOT / "train" / "depth"
    VAL_FRINGE = DATA_ROOT / "val" / "fringe"
    VAL_DEPTH = DATA_ROOT / "val" / "depth"
    TEST_FRINGE = DATA_ROOT / "test" / "fringe"
    TEST_DEPTH = DATA_ROOT / "test" / "depth"

    # Model parameters (must match training)
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    DROPOUT_RATE = 0.5

    # Inference parameters
    BATCH_SIZE = 2
    CHECKPOINT_PATH = "checkpoints/best_model.pth"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Output
    RESULTS_DIR = Path("results")  # FIXED: was "resuls"
    RESULTS_FILE = RESULTS_DIR / "inference_results.json"  # FIXED: added space


# =============================================================================
# Dataset for PNG Depth Maps
# =============================================================================
class FringeFPPDatasetPNG(Dataset):
    """
    Dataset using PNG depth maps organized by fpp_dataset_preparer.py
    
    File naming convention (same base name for both):
        fringe/wooden_board_A60.png → depth/wooden_board_A60.png
        fringe/vial_A120.png → depth/vial_A120.png
    
    The PNG files should already be normalized to uint16 range [0, 65535]
    where 0 = background and (0, 65535] = normalized depth
    """
    
    def __init__(self, fringe_dir: Path, depth_dir: Path):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        
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
                print(f"  Missing depth files for: {missing}")
        
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


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================
def evaluate_dataset(model, dataloader, criterion, device, dataset_name):
    """
    Evaluate model on a dataset

    Args:
        model: UNetFPP model
        dataloader: DataLoader for the dataset
        criterion: Loss function (MaskedRMSELoss)
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
    print(f"  Average RMSE (normalized): {results['avg_rmse']:.6f}")
    print(f"  Min RMSE: {results['min_rmse']:.6f}")
    print(f"  Max RMSE: {results['max_rmse']:.6f}")
    print(f"  Std RMSE: {results['std_rmse']:.6f}")
    
    # Add interpretation
    print(f"\n  Interpretation (assuming 150mm depth range):")
    print(f"    Average error: ~{results['avg_rmse'] * 150:.1f} mm")
    print(f"    Min error: ~{results['min_rmse'] * 150:.1f} mm")
    print(f"    Max error: ~{results['max_rmse'] * 150:.1f} mm")

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
        print(f"❌ Error: Checkpoint not found at {config.CHECKPOINT_PATH}")
        return

    # Create model with same architecture as training
    model = UNetFPP(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # Load checkpoint
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Model loaded successfully!")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"  Best validation loss: {checkpoint['loss']:.6f}")
    print(f"  Total iterations: {checkpoint.get('iteration', 'N/A')}")
    print(f"  Device: {config.DEVICE}")

    # Loss function (use masked RMSE like in training)
    criterion = MaskedRMSELoss()

    # Results dictionary
    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": config.CHECKPOINT_PATH,
        "trained_epochs": checkpoint['epoch'] + 1,
        "checkpoint_loss": checkpoint['loss'],
        "datasets": {}
    }

    # Evaluate on validation set
    if os.path.exists(config.VAL_FRINGE) and os.path.exists(config.VAL_DEPTH):  # FIXED
        try:
            val_dataset = FringeFPPDatasetPNG(config.VAL_FRINGE, config.VAL_DEPTH)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=True if config.DEVICE == "cuda" else False
            )

            val_results = evaluate_dataset(
                model, val_loader, criterion, config.DEVICE, "validation"
            )
            all_results["datasets"]["validation"] = val_results
        except Exception as e:
            print(f"\n❌ Error loading validation set: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Warning: Validation dataset not found")
        print(f"  Fringe dir: {config.VAL_FRINGE}")
        print(f"  Depth dir: {config.VAL_DEPTH}")

    # Evaluate on test set
    if os.path.exists(config.TEST_FRINGE) and os.path.exists(config.TEST_DEPTH):  # FIXED
        try:
            test_dataset = FringeFPPDatasetPNG(config.TEST_FRINGE, config.TEST_DEPTH)
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=True if config.DEVICE == "cuda" else False
            )

            test_results = evaluate_dataset(
                model, test_loader, criterion, config.DEVICE, "test"
            )
            all_results["datasets"]["test"] = test_results
        except Exception as e:
            print(f"\n❌ Error loading test set: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Warning: Test dataset not found")
        print(f"  Fringe dir: {config.TEST_FRINGE}")
        print(f"  Depth dir: {config.TEST_DEPTH}")

    # Create results directory if needed
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        print(f"  {dataset_name.capitalize()}: RMSE = {results['avg_rmse']:.6f} (normalized)")
        print(f"    → ~{results['avg_rmse'] * 150:.1f} mm for 150mm depth range")
    
    # Overfitting check
    if "validation" in all_results["datasets"]:
        val_rmse = all_results["datasets"]["validation"]["avg_rmse"]
        checkpoint_val_rmse = checkpoint['loss']
        
        print(f"\n  Checkpoint validation RMSE: {checkpoint_val_rmse:.6f}")
        print(f"  Current validation RMSE: {val_rmse:.6f}")
        
        if abs(val_rmse - checkpoint_val_rmse) > 0.01:
            print(f"  ⚠ Warning: Results differ from checkpoint (expected ~{checkpoint_val_rmse:.6f})")


if __name__ == "__main__":
    main()
