#!/bin/bash
#SBATCH --job-name=eegformer_nocopy
#SBATCH --partition=Global
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --output=train_%j.log

set -e

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Change to your project directory
cd /Brain/private/a23bono/recherche/BIOT
source env_BIOT/bin/activate

# Set HOME to /tmp to avoid Kerberos issues with expired credentials
# This redirects all cache directories (.cache, .config, etc.) to /tmp
export OLD_HOME="$HOME"
export HOME="/tmp/home_$USER"
mkdir -p "$HOME"

# Set additional cache directories
export MPLCONFIGDIR="$HOME/.config/matplotlib"
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

echo "Training directly from network storage (no copy)..."
echo "Optimized for network I/O with reduced workers and batch size"
echo ""

# Train with optimized settings for network storage
python experiments/train_chbmit.py \
    --model EEGformer \
    --batch-size 16 \
    --lr 0.000005 \
    --num-workers 2 \
    --skip-resample \
    --no-validation \
    --epochs 50

echo "========================================="
echo "Training complete: $(date)"
echo "========================================="
