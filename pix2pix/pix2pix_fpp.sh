#!/bin/bash
# Example training script for pix2pixHD with FPP depth estimation
# Usage: bash train_fpp.sh

#!/bin/bash
#SBATCH --job-name="pix2pixHD_depth"           # Name of your job
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # One training process
#SBATCH --cpus-per-task=8                       # CPU cores for dataloaders
#SBATCH --gres=gpu:a100:1                       # Request 1x A100 GPU
#SBATCH --mem=128G                              # Memory allocation
#SBATCH --time=12:00:00                       # Time limit (2 days for 200 epochs)
#SBATCH --partition=nova                        # Partition/queue
#SBATCH --output=pix2pixHD_depth_%j.out         # Output file (%j = job ID)
#SBATCH --error=pix2pixHD_depth_%j.err          # Error file

module load micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/arpawar/anushlak/temp/micromamba/envs/spiepw

# Navigate to pix2pixHD directory
cd "/work/arpawar/anushlak/SPIE-PW/FPP-ML-Benchmarking/pix2pixHD" || exit 1

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

python train.py \
    --name fpp_pix2pixHD \
    --dataset_type _individual_normalized \
    --loss hybrid_l1 \
    --alpha 0.7 \
    --loadSize 960 \
    --fineSize 960 \
    --input_nc 1 \
    --output_nc 1 \
    --ngf 64 \
    --ndf 64 \
    --n_layers_D 3 \
    --num_D 2 \
    --batchSize 1 \
    --niter 100 \
    --niter_decay 100 \
    --no_instance \
    --label_nc 0 \
    --no_flip \
    --display_freq 100 \
    --print_freq 50 \
    --save_epoch_freq 10 \
    --save_latest_freq 1000 \
    --gpu_ids 0 \
    --lambda_feat 10 \
    --lambda_L1 10 \
    --lambda_grad 5 \
    --lambda_si 2 \
    --no_vgg_loss \
    --nThreads 8

echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="

# Optional: Print final checkpoint location
if [ -d "checkpoints/fringe2depth_exp" ]; then
    echo "Checkpoints saved in: checkpoints/fringe2depth_exp"
    echo "Latest checkpoint files:"
    ls -lh checkpoints/fringe2depth_exp/latest_*.pth 2>/dev/null || echo "No checkpoint files found yet"
fi

# Optional: Show last few lines of loss log
if [ -f "checkpoints/fringe2depth_exp/loss_log.txt" ]; then
    echo "Last 10 lines of loss log:"
    tail -10 checkpoints/fringe2depth_exp/loss_log.txt
fi
