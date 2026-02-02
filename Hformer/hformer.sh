#!/bin/bash
#SBATCH --job-name="hformer_train"          # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # One training process
#SBATCH --cpus-per-task=8                # CPU cores for dataloaders
#SBATCH --gres=gpu:a100:1                # Request 1x A100 GPU
#SBATCH --mem=128G                       # Memory allocation
#SBATCH --time=1-00:00:00                # Time limit (1 day)
#SBATCH --partition=nova                 # Partition/queue
#SBATCH --output=hformer_train_%j.out       # STDOUT (%j = job ID)
#SBATCH --error=hformer_train_%j.err        # STDERR

# -----------------------------
# Activate your conda env
# -----------------------------
module load micromamba
micromamba activate "/work/arpawar/anushlak/temp/micromamba/envs/spiepw"

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
cd "/work/arpawar/anushlak/SPIE-PW/FPP-ML-Benchmarking/Hformer" || exit 1

# -----------------------------
# Run training
# -----------------------------
python -u train.py

echo "Job finished at: $(date)"