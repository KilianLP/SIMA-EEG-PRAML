"""
Simple PyTorch training script for 4‑channel CHB-MIT data.

Uses the 4ch dataloaders (data/fourch_dataset.py) and trains EEGformer
without PyTorch Lightning.
"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from data import prepare_fourch_dataloaders
from model.eegformer import EEGformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEGformer on 4-channel CHB-MIT data")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing train/val/test splits of 4ch pickles")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=10e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simple_eegformer.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for X, y in loader:
        X = X.to(device)  # (batch, C, T)
        y = y.float().to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs
        total += bs

    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X = X.to(device)
        y = y.float().to(device)
        logits = model(X)
        loss = criterion(logits, y)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == y.long()).sum().item()
        bs = X.size(0)
        running_loss += loss.item() * bs
        total += bs

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    args = parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = prepare_fourch_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Build model from a sample batch to set shapes
    model = EEGformer(
        in_channels=4,
        n_classes=1,
        seq_length=2048,
        embed_dim=32,
        num_heads=8,
        num_layers=1,
        dim_feedforward=128,
        kernel_size=5,
        dropout=0.0,
    )
    model.to(device)
    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                args.checkpoint,
            )
            print(f"  ✓ Saved checkpoint to {args.checkpoint}")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.3f}")


if __name__ == "__main__":
    main()
