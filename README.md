# SIMA-EEG-PRAML

Lightweight PyTorch Lightning pipeline for seizure detection on CHB-MIT 10‑second EEG segments using either a CNN+Transformer hybrid or a 1D EEGformer.

## Structure
- `main.py` — CLI entry to train/test; builds configs, sets seeds, launches dataloaders and Lightning trainer.
- `trainer.py` — Config dataclasses, Lightning module (stable BCE + class weights, AUROC/PR metrics), checkpointing/early stopping setup.
- `data/` — Dataset loader, patient filtering, resampling/normalization, and cached pickle validation helpers.
- `model/` — Model definitions: `eegformer.py` (1D conv + Transformer) and `cnn_transformer.py` (STFT + CNN + Transformer).
- `visualize_training_losses.py` — Reads TensorBoard logs and produces loss/metric plots.
- `sh_files/` — SLURM/helper scripts for copying CHB-MIT data and running training on cluster storage.

## Setup
1) Python: 3.9–3.11 recommended. Create a virtualenv:
```
python -m venv .venv && source .venv/bin/activate
```
2) Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```
> Note: You may install a GPU-specific `torch` build first (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121`) if needed, then install the rest.

## Data expectations
- CHB-MIT processed pickles under `data/clean_segments` by default, split into `train/`, `val/`, `test/`.
- Filenames encode patients (`chb01`, …). Loader can resample 256 Hz → 200 Hz unless `--skip-resample` is set.
- Validation caches (`.validation_cache_*.pkl`) are auto-written next to the data to skip re-checking files.

## Quick start
Training EEGformer (defaults: mixed precision if GPU/MPS, class-weighting auto-computed):
```
python main.py --data-path data/clean_segments --model EEGformer
```
Quick smoke test on 1% of data:
```
python main.py --data-path data/clean_segments --model EEGformer --data-fraction 0.01 --epochs 2
```
Evaluate a checkpoint:
```
python main.py --mode test --model EEGformer --checkpoint checkpoints/<exp>/best.ckpt --data-path data/clean_segments
```

## Useful flags (common)
- `--model {EEGformer,CNNTransformer}` — choose architecture.
- `--skip-resample` — keep native 256 Hz (faster IO, slightly longer sequences).
- `--no-class-weights` or `--pos-weight <float>` — control imbalance handling.
- `--early-stopping --patience <n>` — enable early stopping.
- `--data-fraction <0-1>` — subsample for quick runs.

## Logging & plots
- TensorBoard logs: `logs/<experiment>/`.
- Checkpoints: `checkpoints/<experiment>/`.
- Plot losses/metrics from logs:
```
python visualize_training_losses.py --auto --save_dir ./plots
```

## Cluster scripts
- `sh_files/train_local.sh` — SLURM example copying CHB-MIT to local SSD, then running training.
- `sh_files/check_data_size.sh` — estimate dataset size and file counts before copying.
