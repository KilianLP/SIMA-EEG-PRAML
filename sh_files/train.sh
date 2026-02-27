#!/bin/bash
#SBATCH --job-name=EEGFormer
#SBATCH --output=logs/eeg_former.out
#SBATCH --error=logs/eeg_former.err
#SBATCH --gres=gpu:rtx3090:1               
#SBATCH --partition=Brain       
#SBATCH --cpus-per-gpu=4            
#SBATCH --qos=highbrain

export HOME="/Brain/private/k23preus"               
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"

cd /Brain/private/k23preus/SIMA-EEG-PRAML
source venv/bin/activate
python train.py --data-root /Brain/private/Clean_CHB_MIT_4c_8p/clean_segments
