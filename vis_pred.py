"""
Visualization script for UNet predictions with support for different dataset types
and denormalization strategies.

Supports:
- Single image visualization (--image_idx)
- All images in folder (--process_all)
- Correct denormalization for raw, global_normalized, and individual_normalized datasets
"""

import os
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('UNet')
sys.path.append('ResUNet')
sys.path.append('Hformer')
from UNet.unet import UNetFPP
from ResUNet.resunet import ResUNet
from Hformer.hformer import Hformer


# =============================================================================
# Dataset
# =============================================================================
class FringeFPPDatasetPNG(Dataset):
    """Dataset for loading fringe patterns and depth maps"""
    
    def __init__(self, fringe_dir: Path, depth_dir: Path, depth_key="depthMapMeters"):
        self.fringe_dir = Path(fringe_dir)
        self.depth_dir = Path(depth_dir)
        self.depth_key = depth_key
        
        # Get all fringe images
        self.fringe_files = sorted(list(self.fringe_dir.glob("*.png")))
        
        if not self.fringe_files:
            raise ValueError(f"No PNG files found in {fringe_dir}")
        
        # Find matching depth files
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
            raise ValueError(f"No matching depth files found in {depth_dir}")
        
        if missing:
            print(f"⚠ Warning: {len(missing)}/{len(self.fringe_files)} fringe images have no depth match")
        
        print(f"✓ Loaded {len(self.pairs)} sample pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        fringe_path, depth_path = self.pairs[idx]
        
        # Load fringe pattern
        fringe = Image.open(fringe_path).convert('L')
        fringe = np.array(fringe, dtype=np.float32)
        fringe = fringe / 255.0
        
        # Load depth from mat
        mat = sio.loadmat(depth_path)
        
        if self.depth_key not in mat:
            raise KeyError(
                f"Key '{self.depth_key}' not found in {depth_path.name}. "
                f"Available keys: {list(mat.keys())}"
            )
        
        depth = mat[self.depth_key].astype(np.float32)
        
        if depth.ndim != 2:
            raise ValueError(f"Depth must be 2D, got shape {depth.shape}")
        
        if fringe.shape != depth.shape:
            raise ValueError(
                f"Shape mismatch: Fringe {fringe.shape} vs Depth {depth.shape}"
            )
        
        # Convert to tensors
        fringe = torch.from_numpy(fringe).unsqueeze(0)
        depth = torch.from_numpy(depth).unsqueeze(0)
        
        return fringe, depth, str(fringe_path.stem)


# =============================================================================
# Denormalization Functions
# =============================================================================
def denormalize_depth(pred_depth, gt_depth, dataset_type, sample_name=None, depth_params_dir=None, split=None):
    """
    Denormalize predicted and ground truth depth based on dataset type.
    
    Args:
        pred_depth: Predicted depth array (H, W)
        gt_depth: Ground truth depth array (H, W)
        dataset_type: One of "_raw", "_global_normalized", "_individual_normalized"
        sample_name: Sample name (stem) for individual normalization
        depth_params_dir: Directory containing depth_params files for individual normalization
        split: Data split ('train', 'val', or 'test') for individual normalization
    
    Returns:
        Tuple of (denormalized_pred, denormalized_gt) in millimeters
    """
    if dataset_type == "_raw":
        # Raw depth was not normalized; already in mm
        pred_mm = pred_depth
        gt_mm = gt_depth
        
    elif dataset_type == "_global_normalized":
        # Global normalized: was raw_mm / 1000, so multiply by 1000
        pred_mm = pred_depth * 1000.0
        gt_mm = gt_depth * 1000.0
        
    elif dataset_type == "_individual_normalized":
        # Individual normalized: was (raw - min) / (max - min)
        # Need to reverse: raw = normalized * (max - min) + min
        
        if sample_name is None or depth_params_dir is None or split is None:
            raise ValueError(
                "sample_name, depth_params_dir, and split required for individual_normalized"
            )
        
        # Construct path: info_depth_params/{split}/depth/{sample_name}.mat
        params_file = Path(depth_params_dir) / split / "depth" / f"{sample_name}.mat"
        
        if not params_file.exists():
            raise FileNotFoundError(f"Depth params not found: {params_file}")
        
        params = sio.loadmat(params_file)
        depth_min = float(params['depth_min'])
        depth_max = float(params['depth_max'])
        
        # Reverse normalization
        pred_mm = pred_depth * (depth_max - depth_min) + depth_min
        gt_mm = gt_depth * (depth_max - depth_min) + depth_min
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    return pred_mm, gt_mm


# =============================================================================
# Object vs Background Error Analysis
# =============================================================================
def analyze_object_vs_background_error(pred_depth, gt_depth, dataset_type, sample_name=None, depth_params_dir=None, split=None):
    """
    Separate object vs background error analysis.
    
    Returns detailed metrics distinguishing object pixels (GT > 0) from background (GT = 0).
    
    Args:
        pred_depth: Predicted depth tensor (1, H, W)
        gt_depth: Ground truth depth tensor (1, H, W)
        dataset_type: One of "_raw", "_global_normalized", "_individual_normalized"
        sample_name: Sample name for individual normalization
        depth_params_dir: Path to depth params directory
        split: Data split (train/val/test)
    
    Returns:
        Dictionary with object-only, background-only, and overall metrics
    """
    # Convert tensors to numpy
    gt_np = gt_depth.squeeze().cpu().numpy()
    pred_np = pred_depth.squeeze().cpu().numpy()
    
    # Create masks BEFORE denormalization for individual_normalized
    # (background is 0 in normalized space, but becomes depth_min after denormalization)
    if dataset_type == "_individual_normalized":
        object_mask = gt_np > 0  # Object pixels in [0,1] space
        background_mask = gt_np == 0  # Background pixels in [0,1] space
    
    # Denormalize depths to mm
    pred_mm, gt_mm = denormalize_depth(
        pred_np, gt_np, dataset_type, sample_name, depth_params_dir, split
    )
    
    # Create masks AFTER denormalization for raw and global_normalized
    # (background is actually 0 mm in these cases)
    if dataset_type != "_individual_normalized":
        object_mask = gt_mm > 0  # Object pixels (GT has depth)
        background_mask = gt_mm == 0  # Background pixels (GT is zero)
    
    # Calculate errors
    error_mm = np.abs(pred_mm - gt_mm)
    
    # Object-only metrics
    if object_mask.sum() > 0:
        object_errors = error_mm[object_mask]
        object_mae = float(object_errors.mean())
        object_rmse = float(np.sqrt(np.mean(object_errors ** 2)))
        object_median = float(np.median(object_errors))
        object_95th = float(np.percentile(object_errors, 95))
        object_max = float(object_errors.max())
    else:
        object_mae = object_rmse = object_median = object_95th = object_max = 0.0
    
    # Background-only metrics (error on background pixels)
    if background_mask.sum() > 0:
        bg_errors = np.abs(pred_mm[background_mask] - gt_mm[background_mask])
        bg_mae = float(bg_errors.mean())
        bg_rmse = float(np.sqrt(np.mean(bg_errors ** 2)))
        bg_median = float(np.median(bg_errors))
        bg_95th = float(np.percentile(bg_errors, 95))
        bg_max = float(bg_errors.max())
        bg_pixel_count = int(background_mask.sum())
    else:
        bg_mae = bg_rmse = bg_median = bg_95th = bg_max = 0.0
        bg_pixel_count = 0
    
    # Overall metrics (for comparison)
    overall_mae = float(error_mm.mean())
    overall_rmse = float(np.sqrt(np.mean(error_mm ** 2)))
    
    return {
        'sample_name': sample_name,
        # Object metrics
        'object_mae': object_mae,
        'object_rmse': object_rmse,
        'object_median': object_median,
        'object_95th': object_95th,
        'object_max': object_max,
        'object_pixels': int(object_mask.sum()),
        # Background metrics
        'bg_mae': bg_mae,
        'bg_rmse': bg_rmse,
        'bg_median': bg_median,
        'bg_95th': bg_95th,
        'bg_max': bg_max,
        'bg_pixels': bg_pixel_count,
        # Overall metrics
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'total_pixels': error_mm.size
    }


# =============================================================================
# Visualization
# =============================================================================
def visualize_prediction(fringe, gt_depth, pred_depth, sample_name, save_path,
                        dataset_type="_raw", depth_params_dir=None, split=None):
    """
    Visualize fringe pattern, ground truth, prediction, and error.
    
    All depths are displayed in millimeters after denormalization.
    """
    # Convert tensors to numpy
    fringe_np = fringe.squeeze().cpu().numpy()
    gt_np = gt_depth.squeeze().cpu().numpy()
    pred_np = pred_depth.squeeze().cpu().numpy()
    
    # Denormalize depths to mm
    pred_mm, gt_mm = denormalize_depth(
        pred_np, gt_np, dataset_type, sample_name, depth_params_dir, split
    )
    
    # Calculate error in mm
    error_mm = np.abs(pred_mm - gt_mm)
    
    # Calculate object-only metrics for error map title
    if dataset_type == "_individual_normalized":
        object_mask = gt_np > 0
    else:
        object_mask = gt_mm > 0
    
    if object_mask.sum() > 0:
        object_errors = error_mm[object_mask]
        object_mae = object_errors.mean()
        object_rmse = np.sqrt(np.mean(object_errors ** 2))
    else:
        object_mae = 0.0
        object_rmse = 0.0
    
    # Create visualization - 1 row, 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    TITLE_FONTSIZE = 16
    CBAR_LABEL_FONTSIZE = 16
    CBAR_TICK_FONTSIZE = 14

    # -----------------------------
    # Fringe pattern
    # -----------------------------
    im0 = axes[0].imshow(fringe_np, cmap='gray')
    axes[0].set_title('Fringe Pattern', fontsize=TITLE_FONTSIZE)
    axes[0].axis('off')

    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046)
    cbar0.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    # -----------------------------
    # Ground truth depth (mm)
    # -----------------------------
    im1 = axes[1].imshow(gt_mm, cmap='viridis')
    axes[1].set_title(
        f'Ground Truth Depth (mm)\nRange: [{gt_mm.min():.1f}, {gt_mm.max():.1f}]',
        fontsize=TITLE_FONTSIZE
    )
    axes[1].axis('off')

    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046)
    cbar1.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    # -----------------------------
    # Predicted depth (mm)
    # -----------------------------
    im2 = axes[2].imshow(
        pred_mm,
        cmap='viridis',
        vmin=gt_mm.min(),
        vmax=gt_mm.max()
    )
    axes[2].set_title(
        f'Predicted Depth (mm)\nRange: [{pred_mm.min():.1f}, {pred_mm.max():.1f}]',
        fontsize=TITLE_FONTSIZE
    )
    axes[2].axis('off')

    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046)
    cbar2.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    # -----------------------------
    # Absolute error (mm)
    # -----------------------------
    valid_errors = error_mm[error_mm > 0]
    vmax_95 = np.percentile(valid_errors, 95) if len(valid_errors) > 0 else error_mm.max()

    im3 = axes[3].imshow(error_mm, cmap='hot', vmin=0, vmax=vmax_95)
    axes[3].set_title(
        f'Absolute Error (mm)\n'
        f'Object MAE: {object_mae:.2f}, RMSE: {object_rmse:.2f}',
        fontsize=TITLE_FONTSIZE
    )
    axes[3].axis('off')

    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046)
    cbar3.set_label(
        f'Error (mm, 95th %ile: {vmax_95:.1f})',
        fontsize=CBAR_LABEL_FONTSIZE
    )
    cbar3.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

    # -----------------------------
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return metrics
    return {
        'sample_name': sample_name,
        'mean_error_mm': float(error_mm.mean()),
        'max_error_mm': float(error_mm.max()),
        'rmse_mm': float(np.sqrt(np.mean(error_mm ** 2)))
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Visualize UNet predictions')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory (should contain fringe/ and depth/)')
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['_raw', '_global_normalized', '_individual_normalized'],
                       help='Dataset type for correct denormalization')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    # Processing mode: single image or all images
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_idx', type=int,
                      help='Index of single image to visualize')
    group.add_argument('--process_all', action='store_true',
                      help='Process all images in the dataset')
    
    # Depth params for individual normalization
    parser.add_argument('--depth_params_dir', type=str, default=None,
                       help='Directory containing depth_params files (required for individual_normalized)')
    
    args = parser.parse_args()
    
    # Validate depth_params_dir for individual_normalized
    if args.dataset_type == '_individual_normalized' and args.depth_params_dir is None:
        parser.error("--depth_params_dir is required when using --dataset_type _individual_normalized. "
                    "Path should point to info_depth_params base directory (containing train/val/test subdirs).")
    
    # Set up paths
    data_dir = Path(args.data_dir)
    fringe_dir = data_dir / 'fringe'
    depth_dir = data_dir / 'depth'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract split (train/val/test) from data_dir path
    # Assumes data_dir ends with /train, /val, or /test
    split = data_dir.name  # Gets the last part of the path (e.g., 'test')
    if split not in ['train', 'val', 'test']:
        raise ValueError(
            f"Could not determine data split from path: {data_dir}. "
            f"Expected path to end with 'train', 'val', or 'test', got '{split}'"
        )
    
    # Determine depth key based on dataset type
    depth_key_map = {
        '_raw': 'depthMap',
        '_global_normalized': 'depthMapMeters',
        '_individual_normalized': 'depthMapNormalized'
    }
    depth_key = depth_key_map[args.dataset_type]
    
    print("=" * 70)
    print("UNet Prediction Visualization")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Data split: {split}")
    print(f"Depth key: {depth_key}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    if args.process_all:
        print(f"Mode: Process all images")
    else:
        print(f"Mode: Single image (index {args.image_idx})")
    if args.depth_params_dir:
        print(f"Depth params dir: {args.depth_params_dir}")
        print(f"Depth params path: {Path(args.depth_params_dir) / split / 'depth'}")
    print("=" * 70)
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = FringeFPPDatasetPNG(fringe_dir, depth_dir, depth_key=depth_key)
    print(f"Dataset size: {len(dataset)}\n")
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = UNetFPP(in_channels=1, out_channels=1).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}\n")
    
    # Determine which images to process
    if args.process_all:
        indices = range(len(dataset))
        print(f"Processing all {len(dataset)} images...\n")
    else:
        if args.image_idx >= len(dataset):
            raise ValueError(f"image_idx {args.image_idx} >= dataset size {len(dataset)}")
        indices = [args.image_idx]
        print(f"Processing single image at index {args.image_idx}...\n")
    
    # Process images
    all_metrics = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Visualizing"):
            # Load sample
            fringe, gt_depth, sample_name = dataset[idx]
            fringe = fringe.unsqueeze(0).to(device)
            gt_depth = gt_depth.unsqueeze(0).to(device)
            
            # Predict
            pred_depth = model(fringe)
            
            # Collect detailed metrics (object vs background)
            detailed_metrics = analyze_object_vs_background_error(
                pred_depth[0],
                gt_depth[0],
                args.dataset_type,
                sample_name,
                args.depth_params_dir,
                split
            )
            all_metrics.append(detailed_metrics)
            
            # Visualize and save
            save_path = save_dir / f"{sample_name}_prediction.png"
            visualize_prediction(
                fringe[0],
                gt_depth[0],
                pred_depth[0],
                sample_name,
                save_path,
                dataset_type=args.dataset_type,
                depth_params_dir=args.depth_params_dir,
                split=split
            )
            
            if not args.process_all:
                print(f"\n✓ Saved: {save_path}")
                print(f"  Overall MAE: {detailed_metrics['overall_mae']:.2f} mm")
                print(f"  Object MAE: {detailed_metrics['object_mae']:.2f} mm")
                print(f"  Background MAE: {detailed_metrics['bg_mae']:.2f} mm")
    
    # Summary statistics for batch processing
    if args.process_all:
        # Extract metrics
        overall_maes = [m['overall_mae'] for m in all_metrics]
        overall_rmses = [m['overall_rmse'] for m in all_metrics]
        object_maes = [m['object_mae'] for m in all_metrics]
        object_rmses = [m['object_rmse'] for m in all_metrics]
        bg_maes = [m['bg_mae'] for m in all_metrics]
        bg_rmses = [m['bg_rmse'] for m in all_metrics]
        
        # Calculate percentiles
        overall_mae_percentiles = np.percentile(overall_maes, [25, 50, 75, 90, 95])
        object_mae_percentiles = np.percentile(object_maes, [25, 50, 75, 90, 95])
        bg_mae_percentiles = np.percentile(bg_maes, [25, 50, 75, 90, 95])
        
        print("\n" + "=" * 70)
        print("OVERALL ERROR STATISTICS (all pixels, in mm)")
        print("=" * 70)
        print(f"MAE:   avg={np.mean(overall_maes):.2f}, std={np.std(overall_maes):.2f}, "
              f"median={overall_mae_percentiles[1]:.2f}")
        print(f"       min={np.min(overall_maes):.2f}, max={np.max(overall_maes):.2f}")
        print(f"       25th={overall_mae_percentiles[0]:.2f}, 75th={overall_mae_percentiles[2]:.2f}, "
              f"90th={overall_mae_percentiles[3]:.2f}, 95th={overall_mae_percentiles[4]:.2f}")
        print()
        print(f"RMSE:  avg={np.mean(overall_rmses):.2f}, std={np.std(overall_rmses):.2f}")
        print()
        print("=" * 70)
        print("OBJECT-ONLY STATISTICS (pixels where GT > 0, in mm)")
        print("=" * 70)
        print(f"MAE:   avg={np.mean(object_maes):.2f}, std={np.std(object_maes):.2f}, "
              f"median={object_mae_percentiles[1]:.2f}")
        print(f"       min={np.min(object_maes):.2f}, max={np.max(object_maes):.2f}")
        print(f"       25th={object_mae_percentiles[0]:.2f}, 75th={object_mae_percentiles[2]:.2f}, "
              f"90th={object_mae_percentiles[3]:.2f}, 95th={object_mae_percentiles[4]:.2f}")
        print()
        print(f"RMSE:  avg={np.mean(object_rmses):.2f}, std={np.std(object_rmses):.2f}")
        print()
        print("=" * 70)
        print("BACKGROUND STATISTICS (predicted depth where GT = 0, in mm)")
        print("=" * 70)
        print(f"MAE:   avg={np.mean(bg_maes):.2f}, std={np.std(bg_maes):.2f}, "
              f"median={bg_mae_percentiles[1]:.2f}")
        print(f"       min={np.min(bg_maes):.2f}, max={np.max(bg_maes):.2f}")
        print(f"       25th={bg_mae_percentiles[0]:.2f}, 75th={bg_mae_percentiles[2]:.2f}, "
              f"90th={bg_mae_percentiles[3]:.2f}, 95th={bg_mae_percentiles[4]:.2f}")
        print()
        print(f"RMSE:  avg={np.mean(bg_rmses):.2f}, std={np.std(bg_rmses):.2f}")
        print()
        print("=" * 70)
        print(f"IMPACT: Background contributes {np.mean(overall_maes) - np.mean(object_maes):.2f} mm to overall MAE")
        print("=" * 70)
        
        # Create histograms (MAE and RMSE only)
        print("\nGenerating histograms...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # fig.suptitle(f'Overall Error Distribution Across All Images ({len(all_metrics)} samples) - {args.dataset_type}', 
        #              fontsize=14, fontweight='bold')
        
        # MAE (Mean Absolute Error) histogram
        axes[0].hist(overall_maes, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(overall_maes), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(overall_maes):.2f}')
        axes[0].axvline(overall_mae_percentiles[1], color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {overall_mae_percentiles[1]:.2f}')
        axes[0].set_xlabel('MAE (mm)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Mean Absolute Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE histogram
        axes[1].hist(overall_rmses, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(overall_rmses), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(overall_rmses):.2f}')
        axes[1].axvline(np.percentile(overall_rmses, 50), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.percentile(overall_rmses, 50):.2f}')
        axes[1].set_xlabel('RMSE (mm)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Root Mean Squared Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        histogram_path = save_dir / 'error_histograms.png'
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Histograms saved to: {histogram_path}")
        
        # Save metrics to CSV
        import csv
        csv_path = save_dir / 'metrics.csv'
        fieldnames = [
            'sample_name',
            'overall_mae', 'overall_rmse',
            'object_mae', 'object_rmse', 'object_median', 'object_95th', 'object_max', 'object_pixels',
            'bg_mae', 'bg_rmse', 'bg_median', 'bg_95th', 'bg_max', 'bg_pixels',
            'total_pixels'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"✓ Metrics saved to: {csv_path}")
        
        # Save summary statistics to text file
        summary_path = save_dir / 'summary_statistics.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OVERALL ERROR STATISTICS (all pixels, in mm)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MAE:\n")
            f.write(f"  Average:     {np.mean(overall_maes):.2f}\n")
            f.write(f"  Std Dev:     {np.std(overall_maes):.2f}\n")
            f.write(f"  Median:      {overall_mae_percentiles[1]:.2f}\n")
            f.write(f"  Min:         {np.min(overall_maes):.2f}\n")
            f.write(f"  Max:         {np.max(overall_maes):.2f}\n")
            f.write(f"  25th %ile:   {overall_mae_percentiles[0]:.2f}\n")
            f.write(f"  75th %ile:   {overall_mae_percentiles[2]:.2f}\n")
            f.write(f"  90th %ile:   {overall_mae_percentiles[3]:.2f}\n")
            f.write(f"  95th %ile:   {overall_mae_percentiles[4]:.2f}\n\n")
            
            f.write("RMSE:\n")
            f.write(f"  Average:     {np.mean(overall_rmses):.2f}\n")
            f.write(f"  Std Dev:     {np.std(overall_rmses):.2f}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("OBJECT-ONLY STATISTICS (pixels where GT > 0, in mm)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MAE:\n")
            f.write(f"  Average:     {np.mean(object_maes):.2f}\n")
            f.write(f"  Std Dev:     {np.std(object_maes):.2f}\n")
            f.write(f"  Median:      {object_mae_percentiles[1]:.2f}\n")
            f.write(f"  Min:         {np.min(object_maes):.2f}\n")
            f.write(f"  Max:         {np.max(object_maes):.2f}\n")
            f.write(f"  25th %ile:   {object_mae_percentiles[0]:.2f}\n")
            f.write(f"  75th %ile:   {object_mae_percentiles[2]:.2f}\n")
            f.write(f"  90th %ile:   {object_mae_percentiles[3]:.2f}\n")
            f.write(f"  95th %ile:   {object_mae_percentiles[4]:.2f}\n\n")
            
            f.write("RMSE:\n")
            f.write(f"  Average:     {np.mean(object_rmses):.2f}\n")
            f.write(f"  Std Dev:     {np.std(object_rmses):.2f}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("BACKGROUND STATISTICS (predicted depth where GT = 0, in mm)\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MAE:\n")
            f.write(f"  Average:     {np.mean(bg_maes):.2f}\n")
            f.write(f"  Std Dev:     {np.std(bg_maes):.2f}\n")
            f.write(f"  Median:      {bg_mae_percentiles[1]:.2f}\n")
            f.write(f"  Min:         {np.min(bg_maes):.2f}\n")
            f.write(f"  Max:         {np.max(bg_maes):.2f}\n")
            f.write(f"  25th %ile:   {bg_mae_percentiles[0]:.2f}\n")
            f.write(f"  75th %ile:   {bg_mae_percentiles[2]:.2f}\n")
            f.write(f"  90th %ile:   {bg_mae_percentiles[3]:.2f}\n")
            f.write(f"  95th %ile:   {bg_mae_percentiles[4]:.2f}\n\n")
            
            f.write("RMSE:\n")
            f.write(f"  Average:     {np.mean(bg_rmses):.2f}\n")
            f.write(f"  Std Dev:     {np.std(bg_rmses):.2f}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write(f"IMPACT ANALYSIS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Background contribution to overall MAE: {np.mean(overall_maes) - np.mean(object_maes):.2f} mm\n")
            f.write(f"  ({(np.mean(overall_maes) - np.mean(object_maes)) / np.mean(overall_maes) * 100:.1f}% of total error)\n")
            
        print(f"✓ Summary statistics saved to: {summary_path}")
    
    print(f"\n✓ All visualizations saved to: {save_dir}")


if __name__ == '__main__':
    main()
