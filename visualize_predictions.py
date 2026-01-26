"""
Visualize Depth Predictions from Trained Models

Given a fringe image, this script:
1. Loads a trained model (UNet or Hformer)
2. Predicts the depth map
3. Shows side-by-side comparison: fringe, ground truth, prediction, error

Usage:
    python visualize_predictions.py --model unet --checkpoint UNet/checkpoints/best_model.pth --image_idx 0
    python visualize_predictions.py --model hformer --checkpoint Hformer/checkpoints/best_model.pth --image_idx 0
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# =============================================================================
# Load Models
# =============================================================================

def load_unet_model(checkpoint_path, device='cuda'):
    """Load UNet model from checkpoint"""
    from UNet.unet import UNetFPP  # Your UNet implementation
    
    model = UNetFPP(in_channels=1, out_channels=1).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def load_hformer_model(checkpoint_path, device='cuda'):
    """Load Hformer model from checkpoint"""
    from Hformer.hformer import Hformer  # Your Hformer implementation
    
    model = Hformer(in_channels=1, out_channels=1, dropout_rate=0.5).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


# =============================================================================
# Load Data
# =============================================================================

def load_fringe_depth_pair(fringe_path, depth_path):
    """
    Load fringe and depth images
    
    Returns:
        fringe: torch tensor (1, 1, H, W) normalized [0, 1]
        depth: torch tensor (1, 1, H, W) normalized [0, 1]
        fringe_np: numpy array for visualization
        depth_np: numpy array for visualization
    """
    # Load fringe (grayscale)
    fringe_img = Image.open(fringe_path).convert('L')
    fringe_np = np.array(fringe_img, dtype=np.float32)
    
    # Normalize fringe
    if fringe_np.max() > 1.5:
        fringe_np = fringe_np / 255.0
    
    # Load depth (uint16 → float32)
    depth_img = Image.open(depth_path)
    depth_uint16 = np.array(depth_img)
    depth_np = depth_uint16.astype(np.float32) / 65535.0
    
    # Convert to tensors
    fringe = torch.from_numpy(fringe_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)    # (1, 1, H, W)
    
    return fringe, depth, fringe_np, depth_np


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_depth(model, fringe, device='cuda'):
    """
    Predict depth map from fringe image
    
    Args:
        model: Trained model (UNet or Hformer)
        fringe: Input fringe tensor (1, 1, H, W)
        device: Device to run on
        
    Returns:
        pred_np: Predicted depth as numpy array (H, W)
    """
    fringe = fringe.to(device)
    pred = model(fringe)
    pred_np = pred.cpu().squeeze().numpy()
    
    return pred_np


# =============================================================================
# Visualization
# =============================================================================

def visualize_prediction(fringe_np, depth_gt_np, depth_pred_np, save_path=None):
    """
    Create side-by-side visualization of fringe, GT, prediction, and error
    
    Args:
        fringe_np: Fringe image (H, W)
        depth_gt_np: Ground truth depth (H, W)
        depth_pred_np: Predicted depth (H, W)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input, GT, Prediction
    # Fringe image
    axes[0, 0].imshow(fringe_np, cmap='gray')
    axes[0, 0].set_title('Input Fringe Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Ground truth depth
    im1 = axes[0, 1].imshow(depth_gt_np, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Depth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Predicted depth
    im2 = axes[0, 2].imshow(depth_pred_np, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Predicted Depth', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Error maps and statistics
    # Absolute error
    error = np.abs(depth_gt_np - depth_pred_np)
    im3 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 0].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Masked error (only on object pixels)
    mask = depth_gt_np > 0
    masked_error = error.copy()
    masked_error[~mask] = 0
    im4 = axes[1, 1].imshow(masked_error, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title('Error (Object Only)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Statistics
    axes[1, 2].axis('off')
    
    # Calculate metrics
    mse = np.mean((depth_gt_np - depth_pred_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(depth_gt_np - depth_pred_np))
    
    # Masked metrics (object only)
    if mask.sum() > 0:
        masked_mse = np.mean(((depth_gt_np - depth_pred_np) ** 2)[mask])
        masked_rmse = np.sqrt(masked_mse)
        masked_mae = np.mean(np.abs(depth_gt_np - depth_pred_np)[mask])
    else:
        masked_rmse = 0
        masked_mae = 0
    
    stats_text = f"""
    Metrics (All Pixels):
    ─────────────────────
    RMSE: {rmse:.6f}
    MAE:  {mae:.6f}
    
    Metrics (Object Only):
    ─────────────────────
    RMSE: {masked_rmse:.6f}
    MAE:  {masked_mae:.6f}
    
    Object Coverage:
    ─────────────────────
    {mask.sum()} / {mask.size} pixels
    ({mask.sum() / mask.size * 100:.1f}%)
    
    Depth Range:
    ─────────────────────
    GT:   [{depth_gt_np.min():.3f}, {depth_gt_np.max():.3f}]
    Pred: [{depth_pred_np.min():.3f}, {depth_pred_np.max():.3f}]
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()


def visualize_multiple_predictions(model, data_dir, num_samples=5, device='cuda', save_dir=None):
    """
    Visualize predictions for multiple samples
    
    Args:
        model: Trained model
        data_dir: Directory containing fringe and depth subdirectories
        num_samples: Number of samples to visualize
        device: Device to run on
        save_dir: Optional directory to save visualizations
    """
    data_dir = Path(data_dir)
    fringe_dir = data_dir / 'fringe'
    depth_dir = data_dir / 'depth'
    
    fringe_files = sorted(list(fringe_dir.glob('*.png')))[:num_samples]
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, fringe_path in enumerate(fringe_files):
        depth_path = depth_dir / fringe_path.name
        
        if not depth_path.exists():
            print(f"⚠ Skipping {fringe_path.name} - no matching depth map")
            continue
        
        print(f"\nProcessing {fringe_path.name}...")
        
        # Load data
        fringe, depth_gt, fringe_np, depth_gt_np = load_fringe_depth_pair(
            fringe_path, depth_path
        )
        
        # Predict
        depth_pred_np = predict_depth(model, fringe, device)
        
        # Visualize
        save_path = None
        if save_dir:
            save_path = save_dir / f"prediction_{idx:03d}_{fringe_path.stem}.png"
        
        visualize_prediction(fringe_np, depth_gt_np, depth_pred_np, save_path)


def process_all_images(model, data_dir, device='cuda', save_dir=None, model_name='model'):
    """
    Process ALL images in the data directory and save outputs
    
    Args:
        model: Trained model
        data_dir: Directory containing fringe and depth subdirectories
        device: Device to run on
        save_dir: Directory to save outputs
        model_name: Name of model (unet/hformer) for output filenames
    """
    data_dir = Path(data_dir)
    fringe_dir = data_dir / 'fringe'
    depth_dir = data_dir / 'depth'
    
    fringe_files = sorted(list(fringe_dir.glob('*.png')))
    
    if not fringe_files:
        print(f"❌ No PNG files found in {fringe_dir}")
        return
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing {len(fringe_files)} images")
    print(f"{'='*70}\n")
    
    success_count = 0
    error_count = 0
    
    for idx, fringe_path in enumerate(fringe_files):
        depth_path = depth_dir / fringe_path.name
        
        if not depth_path.exists():
            print(f"⚠ [{idx+1}/{len(fringe_files)}] Skipping {fringe_path.name} - no matching depth map")
            error_count += 1
            continue
        
        try:
            print(f"[{idx+1}/{len(fringe_files)}] Processing {fringe_path.name}...", end='')
            
            # Load data
            fringe, depth_gt, fringe_np, depth_gt_np = load_fringe_depth_pair(
                fringe_path, depth_path
            )
            
            # Predict
            depth_pred_np = predict_depth(model, fringe, device)
            
            # Clip to [0, 1] range
            depth_pred_np_clipped = np.clip(depth_pred_np, 0, 1)
            
            if save_dir:
                # Save as CSV
                # np.savetxt(
                #     save_dir / f"depth_pred_{model_name}_{fringe_path.stem}.csv", 
                #     depth_pred_np_clipped, 
                #     fmt="%.6f"
                # )
                
                # Save as uint16 PNG
                depth_pred_uint16 = (depth_pred_np_clipped * 65535).astype(np.uint16)
                depth_pred_img = Image.fromarray(depth_pred_uint16)
                depth_pred_img.save(
                    save_dir / f"depth_pred_{model_name}_{fringe_path.stem}_normalized_depth.png"
                )
            
            success_count += 1
            print(" ✓")
            
        except Exception as e:
            print(f" ❌ Error: {e}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully processed: {success_count}/{len(fringe_files)}")
    if error_count > 0:
        print(f"❌ Errors: {error_count}")
    print(f"Output directory: {save_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize depth predictions')
    parser.add_argument('--model', type=str, required=True, choices=['unet', 'hformer'],
                        help='Model type (unet or hformer)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_unet_training_data_normalized_depth/test',
                        help='Directory containing fringe and depth subdirectories')
    parser.add_argument('--image_idx', type=int, default=None,
                        help='Specific image index to visualize (default: process all images)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize if using --visualize flag')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations (default: just save outputs)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading {args.model.upper()} model from: {args.checkpoint}")
    
    if args.model == 'unet':
        model = load_unet_model(args.checkpoint, device)
    else:
        model = load_hformer_model(args.checkpoint, device)
    
    print("✓ Model loaded successfully!")
    
    data_dir = Path(args.data_dir)
    
    if args.image_idx is not None:
        # Visualize specific image (original behavior)
        fringe_dir = data_dir / 'fringe'
        depth_dir = data_dir / 'depth'
        
        fringe_files = sorted(list(fringe_dir.glob('*.png')))
        
        if args.image_idx >= len(fringe_files):
            print(f"❌ Error: image_idx {args.image_idx} out of range (0-{len(fringe_files)-1})")
            return
        
        fringe_path = fringe_files[args.image_idx]
        depth_path = depth_dir / fringe_path.name
        
        if not depth_path.exists():
            print(f"❌ Error: No matching depth map for {fringe_path.name}")
            return
        
        print(f"\nVisualizing: {fringe_path.name}")
        
        # Load data
        fringe, depth_gt, fringe_np, depth_gt_np = load_fringe_depth_pair(
            fringe_path, depth_path
        )
        
        # Predict
        depth_pred_np = predict_depth(model, fringe, device)
        depth_pred_np_clipped = np.clip(depth_pred_np, 0, 1)
        
        # Save outputs
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.savetxt(save_dir / f"depth_pred_{args.model}_{fringe_path.stem}.csv", 
                      depth_pred_np_clipped, fmt="%.6f")
            
            depth_pred_uint16 = (depth_pred_np_clipped * 65535).astype(np.uint16)
            depth_pred_img = Image.fromarray(depth_pred_uint16)
            depth_pred_img.save(save_dir / f"depth_pred_{args.model}_{fringe_path.stem}_normalized_depth.png")
            
            print(f"✓ Saved outputs to: {save_dir}")
        
        # Visualize if requested
        if args.visualize:
            visualize_prediction(fringe_np, depth_gt_np, depth_pred_np_clipped, None)
        
    elif args.visualize:
        # Visualize multiple images (original behavior)
        print(f"\nVisualizing {args.num_samples} samples from {data_dir}")
        visualize_multiple_predictions(
            model, data_dir, args.num_samples, device, args.save_dir
        )
    else:
        # NEW: Process all images without visualization
        process_all_images(model, data_dir, device, args.save_dir, args.model)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()