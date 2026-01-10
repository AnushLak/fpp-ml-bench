#!/bin/bash
#SBATCH --job-name="unet_train"                 # Name of your job
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # One training process
#SBATCH --cpus-per-task=8                      # CPU cores for dataloaders
#SBATCH --gres=gpu:a100:1                      # Request 1x A100 GPU
#SBATCH --mem=128G                              # Memory allocation
#SBATCH --time=1-00:00:00                        # Time limit (hh:mm:ss)
#SBATCH --partition=nova                       # Partition/queue
#SBATCH --output=unet_train_%j.out              # Output file (%j = job ID)
#SBATCH --error=unet_train_%j.err               # Error file

# -----------------------------
# Activate your conda env
# -----------------------------
source ~/.bashrc
conda activate fpp-ml-bench

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs on node: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Improve dataloader stability
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -----------------------------
# Go to your repo
# -----------------------------
cd "/work/flemingc/aharoon/workspace/fpp/fpp_synthetic_dataset/FPP-ML-Benchmarking/UNet" || exit 1

# -----------------------------
# Run training
# -----------------------------
python -u train.py

echo "Job finished at: $(date)"
