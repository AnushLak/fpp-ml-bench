#!/bin/bash
#SBATCH --job-name="unet_train"                 # Name of your job
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # One training process
#SBATCH --cpus-per-task=8                      # CPU cores for dataloaders
#SBATCH --gres=gpu:a100:1                      # Request 1x A100 GPU
#SBATCH --mem=32G                              # Memory allocation
#SBATCH --time=10:00:00                        # Time limit (hh:mm:ss)
#SBATCH --partition=nova                       # Partition/queue
#SBATCH --output=unet_train_%j.out              # Output file (%j = job ID)
#SBATCH --error=unet_train_%j.err               # Error file

module load micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate spiepw

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs on node: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# (Optional) for more stable multi-worker dataloading on HPC
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Go to your repo / training folder
cd "/work/arpawar/anushlak/SPIE-PW/UNet" || exit 1

# Run training
# Make sure train.py has the correct paths set:
#   train/fringe, train/depth, val/fringe, val/depth, test/fringe, test/depth
python -u train.py

echo "Job finished at: $(date)"
