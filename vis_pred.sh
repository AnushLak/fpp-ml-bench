#!/bin/bash
#SBATCH --job-name="unet_vis_rmse_all_datasets"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --partition=nova
#SBATCH --output=unet_vis_rmse_all_%j.out
#SBATCH --error=unet_vis_rmse_all_%j.err

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
source ~/.bashrc
conda activate fpp-ml-bench

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --------------------------------------------------
# Paths
# --------------------------------------------------
REPO_ROOT="/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/FPP-ML-Benchmarking"

# Base paths for datasets
DATA_BASE="/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/training_datasets"

cd "$REPO_ROOT" || exit 1

# --------------------------------------------------
# 1. Raw Depth Dataset
# --------------------------------------------------
echo -e "\n\n=========================================="
echo "PROCESSING: RAW DEPTH DATASET"
echo "=========================================="

python -u vis_pred.py \
    --checkpoint UNet/checkpoints_rmse_depth_raw/best_model.pth \
    --data_dir "$DATA_BASE/training_data_depth_raw/test" \
    --dataset_type _raw \
    --save_dir visualizations_rmse_depth_raw \
    --device cuda \
    --process_all

# --------------------------------------------------
# 2. Global Normalized Depth Dataset
# --------------------------------------------------
echo -e "\n\n=========================================="
echo "PROCESSING: GLOBAL NORMALIZED DEPTH DATASET"
echo "=========================================="

python -u vis_pred.py \
    --checkpoint UNet/checkpoints_rmse_depth_global_normalized/best_model.pth \
    --data_dir "$DATA_BASE/training_data_depth_global_normalized/test" \
    --dataset_type _global_normalized \
    --save_dir visualizations_rmse_depth_global_normalized \
    --device cuda \
    --process_all

# --------------------------------------------------
# 3. Individual Normalized Depth Dataset
# --------------------------------------------------
echo -e "\n\n=========================================="
echo "PROCESSING: INDIVIDUAL NORMALIZED DEPTH DATASET"
echo "=========================================="

python -u vis_pred.py \
    --checkpoint UNet/checkpoints_rmse_depth_individual_normalized/best_model.pth \
    --data_dir "$DATA_BASE/training_data_depth_individual_normalized/test" \
    --dataset_type _individual_normalized \
    --depth_params_dir "$DATA_BASE/info_depth_params" \
    --save_dir visualizations_rmse_depth_individual_normalized \
    --device cuda \
    --process_all

# # --------------------------------------------------
# # 1. Raw Depth Dataset
# # --------------------------------------------------
# echo -e "\n\n=========================================="
# echo "PROCESSING: RAW DEPTH DATASET"
# echo "=========================================="

# python -u vis_pred.py \
#     --checkpoint UNet/checkpoints_rmse_depth_raw/best_model.pth \
#     --data_dir "$DATA_BASE/training_data_bgremoved_depth_raw/test" \
#     --dataset_type _raw \
#     --save_dir visualizations_rmse_bgremoved_depth_raw \
#     --device cuda \
#     --process_all

# # --------------------------------------------------
# # 2. Global Normalized Depth Dataset
# # --------------------------------------------------
# echo -e "\n\n=========================================="
# echo "PROCESSING: GLOBAL NORMALIZED DEPTH DATASET"
# echo "=========================================="

# python -u vis_pred.py \
#     --checkpoint UNet/checkpoints_rmse_depth_global_normalized/best_model.pth \
#     --data_dir "$DATA_BASE/training_data_bgremoved_depth_global_normalized/test" \
#     --dataset_type _global_normalized \
#     --save_dir visualizations_rmse_bgremoved_depth_global_normalized \
#     --device cuda \
#     --process_all

# # --------------------------------------------------
# # 3. Individual Normalized Depth Dataset
# # --------------------------------------------------
# echo -e "\n\n=========================================="
# echo "PROCESSING: INDIVIDUAL NORMALIZED DEPTH DATASET"
# echo "=========================================="

# python -u vis_pred.py \
#     --checkpoint UNet/checkpoints_rmse_depth_individual_normalized/best_model.pth \
#     --data_dir "$DATA_BASE/training_data_bgremoved_depth_individual_normalized/test" \
#     --dataset_type _individual_normalized \
#     --depth_params_dir "$DATA_BASE/info_depth_params" \
#     --save_dir visualizations_rmse_bgremoved_depth_individual_normalized \
#     --device cuda \
#     --process_all

echo -e "\n\n=========================================="
echo "ALL VISUALIZATIONS COMPLETE"
echo "=========================================="
echo "Job finished at: $(date)"