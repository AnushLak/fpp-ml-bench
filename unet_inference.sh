#!/bin/bash
#SBATCH --job-name="unet_infer"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --partition=nova
#SBATCH --output=unet_infer_%j.out
#SBATCH --error=unet_infer_%j.err

# Activate conda environment
source ~/.bashrc
conda activate fpp-ml-bench

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Set threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go to repo
cd "/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/FPP-ML-Benchmarking" || exit 1

# Run inference on all test images
python -u visualize_predictions.py \
    --model unet \
    --checkpoint UNet/checkpoints/best_model.pth \
    --data_dir /work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/fpp_training_data/train \
    --save_dir depth_pred_unet_train \
    --device cuda

echo "Job finished at: $(date)"