#!/bin/bash
#SBATCH --job-name=eegformer_test
#SBATCH --partition=Global
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=train_%j.log

set -e

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Use /users/local for fast local SSD storage
LOCAL_DATA="/users/local/chbmit_data_$$"  # $$ = job PID for uniqueness
REMOTE_DATA="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments"

echo "Local storage: $LOCAL_DATA"

# Clean up any old leftover data from previous canceled jobs
echo "Cleaning up old data (if any)..."
rm -rf /users/local/chbmit_data_* 2>/dev/null || true

echo "Available space:"
df -h /users/local | tail -1

# Count files to copy (only patients 1-8)
echo ""
echo "Counting files to copy (only patients 1-8)..."
FILE_COUNT=$(find $REMOTE_DATA -name "chb0[1-8]*.pkl" | wc -l)
echo "Total files to copy: $FILE_COUNT (patients 1-8 only)"

echo "========================================="
echo "Copying data to local fast storage..."
echo "Source: $REMOTE_DATA"
echo "Destination: $LOCAL_DATA"
echo "========================================="

mkdir -p $LOCAL_DATA

# Copy using tar with progress monitoring (only patients 1-8)
# This streams the data and avoids per-file network overhead
echo "Using tar for faster copy (better for network storage)..."
echo "Copying only patients 1-8 (train/val/test splits)..."
echo "This will take 10-30 minutes depending on network speed..."
echo ""

# Start monitoring progress in background
(
  while [ ! -f /tmp/copy_done_$$ ]; do
    sleep 30
    if [ -d "$LOCAL_DATA" ]; then
      COPIED=$(find $LOCAL_DATA -name "chb0[1-8]*.pkl" 2>/dev/null | wc -l)
      SIZE=$(du -sh $LOCAL_DATA 2>/dev/null | cut -f1)
      echo "[$(date +%H:%M:%S)] Progress: $COPIED files, $SIZE copied..."
    fi
  done
) &
MONITOR_PID=$!

# Do the actual copy with tar (only patients 1-8)
time (cd $REMOTE_DATA && find . -name "chb0[1-8]*.pkl" -print0 | tar cf - --null -T -) | (cd $LOCAL_DATA && tar xf -)

# Stop monitoring
touch /tmp/copy_done_$$
wait $MONITOR_PID 2>/dev/null || true
rm -f /tmp/copy_done_$$

echo ""
echo "Data copy complete!"
echo "Local data size:"
du -sh $LOCAL_DATA
FINAL_COUNT=$(find $LOCAL_DATA -name "chb0[1-8]*.pkl" | wc -l)
echo "Files copied: $FINAL_COUNT / $FILE_COUNT (patients 1-8 only)"

echo "========================================="
echo "Starting training with local data..."
echo "========================================="

# Change to your project directory
cd /Brain/private/a23bono/recherche/BIOT

# Activate environment
source env_BIOT/bin/activate

# Set HOME to /tmp to avoid Kerberos issues with expired credentials
export OLD_HOME="$HOME"
export HOME="/tmp/home_$USER"
mkdir -p "$HOME"

# Set additional cache directories
export MPLCONFIGDIR="$HOME/.config/matplotlib"
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

# Train with local data
python experiments/train_chbmit.py \
    --data-path $LOCAL_DATA \
    --model EEGformer \
    --batch-size 16 \
    --lr 0.000005 \
    --num-workers 8 \
    --skip-resample \
    --no-validation \
    --epochs 50

echo "========================================="
echo "Training complete: $(date)"
echo "========================================="

# Cleanup local data
echo "Cleaning up local data..."
rm -rf $LOCAL_DATA
echo "Done!"
