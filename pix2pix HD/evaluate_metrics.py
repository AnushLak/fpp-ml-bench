"""
Evaluate pix2pixHD depth prediction results

Calculates RMSE and MAE between generated depth images and ground truth

Usage:
    python evaluate_metrics.py --results_dir ./results/fringe2depth_exp --ground_truth_dir /path/to/ground_truth
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm


def load_depth_image(path):
    """
    Load depth image and normalize to [0, 1]

    Args:
        path: Path to depth image (uint16 PNG or uint8 PNG)

    Returns:
        numpy array of shape (H, W) with values in [0, 1]
    """
    img = Image.open(path)

    # Check if uint16 or uint8
    if img.mode == 'I;16' or img.mode == 'I':
        # uint16 image
        depth = np.array(img, dtype=np.float32) / 65535.0
    else:
        # uint8 image
        depth = np.array(img, dtype=np.float32) / 255.0

    return depth


def calculate_rmse(pred, gt, mask=None):
    """
    Calculate Root Mean Square Error

    Args:
        pred: Predicted depth map (H, W)
        gt: Ground truth depth map (H, W)
        mask: Optional binary mask (H, W) - only compute on masked pixels

    Returns:
        RMSE value
    """
    if mask is not None:
        diff = (pred - gt)[mask > 0]
    else:
        diff = (pred - gt).flatten()

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(pred, gt, mask=None):
    """
    Calculate Mean Absolute Error

    Args:
        pred: Predicted depth map (H, W)
        gt: Ground truth depth map (H, W)
        mask: Optional binary mask (H, W) - only compute on masked pixels

    Returns:
        MAE value
    """
    if mask is not None:
        diff = np.abs(pred - gt)[mask > 0]
    else:
        diff = np.abs(pred - gt).flatten()

    mae = np.mean(diff)
    return mae


def evaluate_results(results_dir, ground_truth_dir, use_mask=True, output_file=None):
    """
    Evaluate all generated images against ground truth

    Args:
        results_dir: Directory containing generated depth images
        ground_truth_dir: Directory containing ground truth depth images
        use_mask: If True, only compute metrics on non-zero pixels (object only)
        output_file: Optional path to save results JSON

    Returns:
        Dictionary with evaluation metrics
    """
    results_dir = Path(results_dir)
    ground_truth_dir = Path(ground_truth_dir)

    # Find all generated depth images
    generated_files = sorted(list(results_dir.glob('*_depth.png')))

    if len(generated_files) == 0:
        print(f"❌ No generated depth images found in {results_dir}")
        return None

    print(f"Found {len(generated_files)} generated images")
    print(f"Ground truth directory: {ground_truth_dir}")

    # Show a sample of what we're looking for
    if len(generated_files) > 0:
        sample_gen = generated_files[0]
        sample_name = sample_gen.stem.replace('_depth', '')
        sample_gt = ground_truth_dir / f"{sample_name}.png"
        print(f"\nExample file matching:")
        print(f"  Generated: {sample_gen.name}")
        print(f"  Looking for GT: {sample_gt.name}")
        print(f"  GT exists: {sample_gt.exists()}")

    results = {
        'all_pixels': {'rmse': [], 'mae': []},
        'masked_pixels': {'rmse': [], 'mae': []},
        'per_image': []
    }

    for gen_path in tqdm(generated_files, desc="Evaluating images"):
        # Extract image name (remove _depth.png suffix)
        img_name = gen_path.stem.replace('_depth', '')

        # Find corresponding ground truth
        gt_path = ground_truth_dir / f"{img_name}.png"

        if not gt_path.exists():
            print(f"⚠ Warning: No ground truth found for {img_name}, skipping")
            continue

        # Load images
        pred = load_depth_image(gen_path)
        gt = load_depth_image(gt_path)

        # Ensure same shape
        if pred.shape != gt.shape:
            print(f"⚠ Warning: Shape mismatch for {img_name}: pred {pred.shape} vs gt {gt.shape}, skipping")
            continue

        # Create mask (non-zero pixels in ground truth)
        mask = gt > 0

        # Calculate metrics for all pixels
        rmse_all = calculate_rmse(pred, gt, mask=None)
        mae_all = calculate_mae(pred, gt, mask=None)

        results['all_pixels']['rmse'].append(rmse_all)
        results['all_pixels']['mae'].append(mae_all)

        # Calculate metrics for masked pixels only
        if use_mask and mask.sum() > 0:
            rmse_masked = calculate_rmse(pred, gt, mask=mask)
            mae_masked = calculate_mae(pred, gt, mask=mask)

            results['masked_pixels']['rmse'].append(rmse_masked)
            results['masked_pixels']['mae'].append(mae_masked)
        else:
            rmse_masked = rmse_all
            mae_masked = mae_all

        # Store per-image results
        results['per_image'].append({
            'image': img_name,
            'rmse_all': float(rmse_all),
            'mae_all': float(mae_all),
            'rmse_masked': float(rmse_masked),
            'mae_masked': float(mae_masked),
            'mask_coverage': float(mask.sum() / mask.size)
        })

    # Check if we found any matching files
    if len(results['per_image']) == 0:
        print("\n" + "=" * 70)
        print("❌ ERROR: No matching ground truth files found!")
        print("=" * 70)
        print("\nPlease check:")
        print("1. Ground truth directory path is correct")
        print("2. Generated file names match ground truth file names")
        print("3. Ground truth files have .png extension")
        print("\nGenerated files found:")
        for gf in generated_files[:5]:
            print(f"  - {gf.name}")
        if len(generated_files) > 5:
            print(f"  ... and {len(generated_files) - 5} more")
        print("\nExpected ground truth files (examples):")
        for gf in generated_files[:5]:
            img_name = gf.stem.replace('_depth', '')
            print(f"  - {img_name}.png")
        print("=" * 70)
        return None, None

    # Calculate summary statistics
    summary = {
        'num_images': len(results['per_image']),
        'all_pixels': {
            'rmse_mean': float(np.mean(results['all_pixels']['rmse'])),
            'rmse_std': float(np.std(results['all_pixels']['rmse'])),
            'rmse_min': float(np.min(results['all_pixels']['rmse'])),
            'rmse_max': float(np.max(results['all_pixels']['rmse'])),
            'mae_mean': float(np.mean(results['all_pixels']['mae'])),
            'mae_std': float(np.std(results['all_pixels']['mae'])),
            'mae_min': float(np.min(results['all_pixels']['mae'])),
            'mae_max': float(np.max(results['all_pixels']['mae'])),
        },
        'masked_pixels': {
            'rmse_mean': float(np.mean(results['masked_pixels']['rmse'])),
            'rmse_std': float(np.std(results['masked_pixels']['rmse'])),
            'rmse_min': float(np.min(results['masked_pixels']['rmse'])),
            'rmse_max': float(np.max(results['masked_pixels']['rmse'])),
            'mae_mean': float(np.mean(results['masked_pixels']['mae'])),
            'mae_std': float(np.std(results['masked_pixels']['mae'])),
            'mae_min': float(np.min(results['masked_pixels']['mae'])),
            'mae_max': float(np.max(results['masked_pixels']['mae'])),
        } if use_mask else None
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of images evaluated: {summary['num_images']}")
    print("\nAll Pixels:")
    print(f"  RMSE: {summary['all_pixels']['rmse_mean']:.6f} ± {summary['all_pixels']['rmse_std']:.6f}")
    print(f"        (min: {summary['all_pixels']['rmse_min']:.6f}, max: {summary['all_pixels']['rmse_max']:.6f})")
    print(f"  MAE:  {summary['all_pixels']['mae_mean']:.6f} ± {summary['all_pixels']['mae_std']:.6f}")
    print(f"        (min: {summary['all_pixels']['mae_min']:.6f}, max: {summary['all_pixels']['mae_max']:.6f})")

    if use_mask and summary['masked_pixels'] is not None:
        print("\nMasked Pixels (Object Only):")
        print(f"  RMSE: {summary['masked_pixels']['rmse_mean']:.6f} ± {summary['masked_pixels']['rmse_std']:.6f}")
        print(f"        (min: {summary['masked_pixels']['rmse_min']:.6f}, max: {summary['masked_pixels']['rmse_max']:.6f})")
        print(f"  MAE:  {summary['masked_pixels']['mae_mean']:.6f} ± {summary['masked_pixels']['mae_std']:.6f}")
        print(f"        (min: {summary['masked_pixels']['mae_min']:.6f}, max: {summary['masked_pixels']['mae_max']:.6f})")

    print("=" * 70)

    # Save results
    if output_file:
        output_data = {
            'summary': summary,
            'per_image_results': results['per_image']
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    return summary, results['per_image']


def main():
    parser = argparse.ArgumentParser(description='Evaluate pix2pixHD depth predictions')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing generated depth images (*_depth.png)')
    parser.add_argument('--ground_truth_dir', type=str, required=True,
                        help='Directory containing ground truth depth images')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation results JSON (default: results_dir/evaluation_metrics.json)')
    parser.add_argument('--no_mask', action='store_true',
                        help='Do not compute masked metrics (object-only)')

    args = parser.parse_args()

    # Set default output file
    if args.output_file is None:
        args.output_file = os.path.join(args.results_dir, 'evaluation_metrics.json')

    # Run evaluation
    summary, per_image = evaluate_results(
        results_dir=args.results_dir,
        ground_truth_dir=args.ground_truth_dir,
        use_mask=not args.no_mask,
        output_file=args.output_file
    )

    if summary is None:
        print("\n❌ Evaluation failed. Please check the error messages above.")
        exit(1)


if __name__ == '__main__':
    main()
