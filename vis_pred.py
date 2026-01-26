"""
python vis_pred.py \
--model unet \
--checkpoint /work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/FPP-ML-Benchmarking/UNet/checkpoints_L1Loss/best_model.pth \
--data_dir /work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_training_data_depth_raw/test \
--image_idx 0 \
--save_dir visualizations \
--device cuda \
--erosion_pixels 3
"""

import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2

# ---------------------------------------------------------
# Load UNet
# ---------------------------------------------------------
def load_unet_model(checkpoint_path, device='cuda'):
    from UNet.unet import UNetFPP
    model = UNetFPP(in_channels=1, out_channels=1).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------
# Load fringe + depth pair
# ---------------------------------------------------------
def load_fringe_depth_pair(fringe_path, depth_path):
    from PIL import Image
    # Fringe (PNG → float32 normalized)
    fringe_np = np.array(Image.open(fringe_path).convert("L"), dtype=np.float32)
    fringe_np /= 255.0

    # Depth (.mat)
    mat = sio.loadmat(depth_path)
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    depth_gt_np = mat[key].astype(np.float32)

    # Convert to tensors
    fringe = torch.from_numpy(fringe_np)[None, None, :, :]  # (1,1,H,W)
    depth_gt = torch.from_numpy(depth_gt_np)[None, None, :, :]

    return fringe, depth_gt, fringe_np, depth_gt_np


# ---------------------------------------------------------
# Predict depth
# ---------------------------------------------------------
@torch.no_grad()
def predict_depth(model, fringe, device='cuda'):
    fringe = fringe.to(device)
    pred = model(fringe)
    return pred.cpu().squeeze().numpy()


# ---------------------------------------------------------
# Create eroded masks
# ---------------------------------------------------------
def create_masks(depth_gt_np, erosion_pixels=3):
    """
    Create full object mask, interior mask (eroded), and boundary mask
    
    Returns:
        full_mask: Binary mask of all object pixels (GT > 0)
        interior_mask: Eroded mask (interior pixels only)
        boundary_mask: Boundary pixels only (full - interior)
    """
    # Full object mask
    full_mask = (depth_gt_np > 0).astype(np.uint8)
    
    if erosion_pixels > 0:
        # Erode to get interior
        kernel = np.ones((erosion_pixels, erosion_pixels), np.uint8)
        interior_mask = cv2.erode(full_mask, kernel, iterations=1)
        
        # Boundary = full - interior
        boundary_mask = full_mask - interior_mask
    else:
        # No erosion
        interior_mask = full_mask
        boundary_mask = np.zeros_like(full_mask)
    
    return full_mask, interior_mask, boundary_mask


# ---------------------------------------------------------
# Compute metrics with masks
# ---------------------------------------------------------
def compute_metrics(depth_gt_np, depth_pred_np, full_mask, interior_mask, boundary_mask):
    """
    Compute RMSE for full object, interior only, and boundary only
    """
    metrics = {}
    
    # Full object metrics
    if full_mask.sum() > 0:
        gt_full = depth_gt_np[full_mask > 0]
        pred_full = depth_pred_np[full_mask > 0]
        metrics['rmse_full'] = np.sqrt(((pred_full - gt_full) ** 2).mean())
        metrics['mae_full'] = np.abs(pred_full - gt_full).mean()
        metrics['n_full'] = full_mask.sum()
    else:
        metrics['rmse_full'] = np.nan
        metrics['mae_full'] = np.nan
        metrics['n_full'] = 0
    
    # Interior metrics
    if interior_mask.sum() > 0:
        gt_interior = depth_gt_np[interior_mask > 0]
        pred_interior = depth_pred_np[interior_mask > 0]
        metrics['rmse_interior'] = np.sqrt(((pred_interior - gt_interior) ** 2).mean())
        metrics['mae_interior'] = np.abs(pred_interior - gt_interior).mean()
        metrics['n_interior'] = interior_mask.sum()
    else:
        metrics['rmse_interior'] = np.nan
        metrics['mae_interior'] = np.nan
        metrics['n_interior'] = 0
    
    # Boundary metrics
    if boundary_mask.sum() > 0:
        gt_boundary = depth_gt_np[boundary_mask > 0]
        pred_boundary = depth_pred_np[boundary_mask > 0]
        metrics['rmse_boundary'] = np.sqrt(((pred_boundary - gt_boundary) ** 2).mean())
        metrics['mae_boundary'] = np.abs(pred_boundary - gt_boundary).mean()
        metrics['n_boundary'] = boundary_mask.sum()
    else:
        metrics['rmse_boundary'] = np.nan
        metrics['mae_boundary'] = np.nan
        metrics['n_boundary'] = 0
    
    return metrics


# ---------------------------------------------------------
# Visualization with erosion analysis
# ---------------------------------------------------------
def visualize_with_erosion(fringe_np, depth_gt_np, depth_pred_np, 
                          full_mask, interior_mask, boundary_mask,
                          metrics, save_path, erosion_pixels):
    """
    Create comprehensive visualization showing:
    - GT, Pred, Full Error
    - Interior Error, Boundary Error, Masks
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Calculate errors
    error_full = np.abs(depth_pred_np - depth_gt_np)
    error_interior = np.where(interior_mask > 0, error_full, np.nan)
    error_boundary = np.where(boundary_mask > 0, error_full, np.nan)
    
    vmin = depth_gt_np.min()
    vmax = depth_gt_np.max()
    
    # Determine error colorbar range (use same for all error maps)
    error_vmax = np.nanpercentile(error_full[full_mask > 0], 99)  # 99th percentile to avoid outliers
    
    # Row 1: GT, Pred, Full Error
    plt.subplot(2, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(depth_gt_np, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Depth (mm)')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Prediction")
    im = plt.imshow(depth_pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Depth (mm)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title(f"Full Error (RMSE={metrics['rmse_full']:.2f}mm)")
    plt.imshow(error_full, cmap='inferno', vmin=0, vmax=error_vmax)
    plt.colorbar(label='Error (mm)')
    plt.axis('off')
    
    # Row 2: Interior Error, Boundary Error, Masks
    plt.subplot(2, 3, 4)
    if not np.isnan(metrics['rmse_interior']):
        plt.title(f"Interior Error (RMSE={metrics['rmse_interior']:.2f}mm, n={metrics['n_interior']})")
        plt.imshow(error_interior, cmap='inferno', vmin=0, vmax=error_vmax)
        plt.colorbar(label='Error (mm)')
    else:
        plt.title("Interior Error (No pixels after erosion)")
        plt.imshow(np.zeros_like(error_full), cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    if not np.isnan(metrics['rmse_boundary']):
        plt.title(f"Boundary Error (RMSE={metrics['rmse_boundary']:.2f}mm, n={metrics['n_boundary']})")
        plt.imshow(error_boundary, cmap='inferno', vmin=0, vmax=error_vmax)
        plt.colorbar(label='Error (mm)')
    else:
        plt.title("Boundary Error (No boundary pixels)")
        plt.imshow(np.zeros_like(error_full), cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title(f"Masks (Erosion={erosion_pixels}px)")
    # Create RGB visualization: R=boundary, G=interior, B=background
    mask_vis = np.zeros((*full_mask.shape, 3))
    mask_vis[boundary_mask > 0] = [1, 0, 0]  # Red = boundary
    mask_vis[interior_mask > 0] = [0, 1, 0]  # Green = interior
    plt.imshow(mask_vis)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='g', lw=4, label='Interior'),
        plt.Line2D([0], [0], color='r', lw=4, label='Boundary')
    ], loc='upper right')
    plt.axis('off')
    
    plt.suptitle(f"Depth Prediction Analysis (Erosion: {erosion_pixels} pixels)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {save_path}")


def save_pred_mat(depth_pred_np, save_path):
    sio.savemat(save_path, {"depth_pred": depth_pred_np.astype(np.float32)})
    print(f"✓ Saved predicted depth MAT to {save_path}")


# ---------------------------------------------------------
# Print metrics summary
# ---------------------------------------------------------
def print_metrics_summary(metrics, erosion_pixels):
    """Print formatted metrics summary"""
    print("\n" + "="*70)
    print(f"METRICS SUMMARY (Erosion: {erosion_pixels} pixels)")
    print("="*70)
    
    print(f"\nFull Object (n={metrics['n_full']} pixels):")
    print(f"  RMSE: {metrics['rmse_full']:.2f} mm")
    print(f"  MAE:  {metrics['mae_full']:.2f} mm")
    
    if not np.isnan(metrics['rmse_interior']):
        print(f"\nInterior Only (n={metrics['n_interior']} pixels, {100*metrics['n_interior']/metrics['n_full']:.1f}% of object):")
        print(f"  RMSE: {metrics['rmse_interior']:.2f} mm")
        print(f"  MAE:  {metrics['mae_interior']:.2f} mm")
        improvement = ((metrics['rmse_full'] - metrics['rmse_interior']) / metrics['rmse_full']) * 100
        print(f"  Improvement: {improvement:.1f}% better than full object")
    
    if not np.isnan(metrics['rmse_boundary']):
        print(f"\nBoundary Only (n={metrics['n_boundary']} pixels, {100*metrics['n_boundary']/metrics['n_full']:.1f}% of object):")
        print(f"  RMSE: {metrics['rmse_boundary']:.2f} mm")
        print(f"  MAE:  {metrics['mae_boundary']:.2f} mm")
        if not np.isnan(metrics['rmse_interior']):
            ratio = metrics['rmse_boundary'] / metrics['rmse_interior']
            print(f"  Boundary/Interior ratio: {ratio:.2f}x worse")
    
    print("="*70 + "\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize depth predictions with boundary analysis")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "hformer"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--erosion_pixels", type=int, default=3, 
                       help="Number of pixels to erode from boundary (default: 3)")
    parser.add_argument("--compare_erosions", action='store_true',
                       help="Compare multiple erosion values [0,2,3,5]")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.model == "unet":
        model = load_unet_model(args.checkpoint, device)
    else:
        raise NotImplementedError("HFormer loader not implemented yet")

    # Paths
    data_dir = Path(args.data_dir)
    fringe_dir = data_dir / "fringe"
    depth_dir = data_dir / "depth"

    fringe_files = sorted(list(fringe_dir.glob("*.png")))
    if args.image_idx >= len(fringe_files):
        print(f"❌ image_idx {args.image_idx} out of range")
        return

    fringe_path = fringe_files[args.image_idx]
    depth_path = depth_dir / (fringe_path.stem + ".mat")

    if not depth_path.exists():
        print(f"❌ No matching depth file for {fringe_path.name}")
        return

    print(f"Visualizing {fringe_path.name}")

    # Load data
    fringe, depth_gt, fringe_np, depth_gt_np = load_fringe_depth_pair(fringe_path, depth_path)

    # Predict
    depth_pred_np = predict_depth(model, fringe, device)
    depth_pred_np = depth_pred_np * 1000.0  # Convert to mm

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save predicted depth .mat
    mat_save_path = save_dir / f"{fringe_path.stem}_pred.mat"
    save_pred_mat(depth_pred_np, mat_save_path)

    # Erosion analysis
    if args.compare_erosions:
        # Compare multiple erosion values
        erosion_values = [0, 2, 3, 5]
        print("\n" + "="*70)
        print("COMPARING DIFFERENT EROSION VALUES")
        print("="*70)
        
        all_metrics = []
        for erosion_px in erosion_values:
            full_mask, interior_mask, boundary_mask = create_masks(depth_gt_np, erosion_px)
            metrics = compute_metrics(depth_gt_np, depth_pred_np, full_mask, interior_mask, boundary_mask)
            all_metrics.append((erosion_px, metrics))
            
            # Save visualization for each erosion value
            save_path = save_dir / f"{fringe_path.stem}_erosion{erosion_px}px_viz.png"
            visualize_with_erosion(fringe_np, depth_gt_np, depth_pred_np,
                                  full_mask, interior_mask, boundary_mask,
                                  metrics, save_path, erosion_px)
            print_metrics_summary(metrics, erosion_px)
        
        # Summary comparison table
        print("\n" + "="*70)
        print("EROSION COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Erosion (px)':<15} {'Full RMSE':<15} {'Interior RMSE':<15} {'Boundary RMSE':<15}")
        print("-"*70)
        for erosion_px, metrics in all_metrics:
            interior_str = f"{metrics['rmse_interior']:.2f}" if not np.isnan(metrics['rmse_interior']) else "N/A"
            boundary_str = f"{metrics['rmse_boundary']:.2f}" if not np.isnan(metrics['rmse_boundary']) else "N/A"
            print(f"{erosion_px:<15} {metrics['rmse_full']:<15.2f} {interior_str:<15} {boundary_str:<15}")
        print("="*70 + "\n")
        
    else:
        # Single erosion value
        full_mask, interior_mask, boundary_mask = create_masks(depth_gt_np, args.erosion_pixels)
        metrics = compute_metrics(depth_gt_np, depth_pred_np, full_mask, interior_mask, boundary_mask)
        
        # Save visualization
        save_path = save_dir / f"{fringe_path.stem}_erosion{args.erosion_pixels}px_viz.png"
        visualize_with_erosion(fringe_np, depth_gt_np, depth_pred_np,
                              full_mask, interior_mask, boundary_mask,
                              metrics, save_path, args.erosion_pixels)
        
        print_metrics_summary(metrics, args.erosion_pixels)
    
    # Original metrics (for comparison with previous output)
    print("\n" + "="*70)
    print("ADDITIONAL METRICS (Original format)")
    print("="*70)
    
    mask = depth_gt_np > 0
    gt_valid = depth_gt_np[mask]
    pred_valid = depth_pred_np[mask]

    # Linear correction
    A = np.vstack([gt_valid.flatten(), np.ones_like(gt_valid.flatten())]).T
    a, b = np.linalg.lstsq(A, pred_valid.flatten(), rcond=None)[0]

    print(f"Linear correction: pred ≈ {a:.4f} * gt + {b:.2f}")
    pred_corr = (pred_valid - b) / a
    rmse_corr = np.sqrt(((pred_corr - gt_valid) ** 2).mean())
    print(f"RMSE after linear correction: {rmse_corr:.2f} mm")

    print(f"\nGT min/max/mean: {depth_gt_np.min():.1f} / {depth_gt_np.max():.1f} / {depth_gt_np[mask].mean():.1f} mm")
    print(f"Pred min/max/mean: {depth_pred_np.min():.1f} / {depth_pred_np.max():.1f} / {depth_pred_np[mask].mean():.1f} mm")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()