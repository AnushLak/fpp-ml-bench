#!/bin/bash
#SBATCH --job-name="unet_vis_all_losses"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --partition=nova
#SBATCH --output=unet_vis_all_%j.out
#SBATCH --error=unet_vis_all_%j.err

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
DATA_DIR="/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_training_data_depth_raw/test"

cd "$REPO_ROOT" || exit 1

# --------------------------------------------------
# L1 Loss
# --------------------------------------------------
echo -e "\n\n========== L1 LOSS ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_L1Loss/best_model.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_L1Loss \
        --device cuda
done

# --------------------------------------------------
# Masked L1 Loss
# --------------------------------------------------
echo -e "\n\n========== MASKED L1 LOSS ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_MaskedL1Loss/best_model.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_MaskedL1Loss \
        --device cuda
done

# --------------------------------------------------
# Hybrid L1 Loss (alpha = 0.9)
# --------------------------------------------------
echo -e "\n\n========== HYBRID L1 LOSS (alpha=0.9) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_HybridL1Loss_alpha09/best_model.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_HybridL1Loss_alpha09 \
        --device cuda
done

# --------------------------------------------------
# Hybrid L1 Loss (alpha = 0.8)
# --------------------------------------------------
echo -e "\n\n========== HYBRID L1 LOSS (alpha=0.8) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_HybridL1Loss_alpha08/best_model.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_HybridL1Loss_alpha08 \
        --device cuda
done

# --------------------------------------------------
# Masked RMSE Loss (epoch 150)
# --------------------------------------------------
echo -e "\n\n========== MASKED RMSE LOSS (epoch 150) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_MaskedRMSELoss/checkpoint_epoch_0150.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_MaskedRMSELoss_epoch150 \
        --device cuda
done

# --------------------------------------------------
# RMSE Loss (epoch 150)
# --------------------------------------------------
echo -e "\n\n========== RMSE LOSS (epoch 150) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_RMSELoss/checkpoint_epoch_0150.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_RMSELoss_epoch150 \
        --device cuda
done

# --------------------------------------------------
# Hybrid RMSE Loss (alpha = 0.9, epoch 150)
# --------------------------------------------------
echo -e "\n\n========== HYBRID RMSE LOSS (alpha=0.9, epoch 150) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_HybridRMSELoss_alpha09/checkpoint_epoch_0150.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_HybridRMSELoss_alpha09_epoch150 \
        --device cuda
done

# --------------------------------------------------
# Hybrid RMSE Loss (alpha = 0.8, epoch 150)
# --------------------------------------------------
echo -e "\n\n========== HYBRID RMSE LOSS (alpha=0.8, epoch 150) ==========\n"
for IDX in 23 0 7; do
    python -u vis_pred.py \
        --model unet \
        --checkpoint UNet/checkpoints_HybridRMSELoss_alpha08/checkpoint_epoch_0150.pth \
        --data_dir "$DATA_DIR" \
        --image_idx $IDX \
        --save_dir visualizations_HybridRMSELoss_alpha08_epoch150 \
        --device cuda
done

echo "Job finished at: $(date)"