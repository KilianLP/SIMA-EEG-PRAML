import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import CNNTransformer
from utils import CHBMITLoader, BCE
from pyhealth.metrics import binary_metrics_fn


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.validation_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.validation_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if sum(gt) * (len(gt) - sum(gt)) != 0:
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)
        print(f"Test Results: {result}")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return [optimizer]


def filter_files_by_patient(files, max_patient=8):
    """Filter files to only include first N patients (chb01 to chbN)"""
    filtered_files = []
    for f in files:
        # Extract patient number from filename (e.g., chb01, chb02, etc.)
        if f.startswith('chb'):
            try:
                patient_num = int(f[3:5])
                if patient_num <= max_patient:
                    filtered_files.append(f)
            except ValueError:
                continue
    return filtered_files


def prepare_CHB_MIT_8patients_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = "/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments"

    # Load all files and filter for first 8 patients
    train_files = filter_files_by_patient(
        os.listdir(os.path.join(root, "train")), max_patient=8
    )
    val_files = filter_files_by_patient(
        os.listdir(os.path.join(root, "val")), max_patient=8
    )
    test_files = filter_files_by_patient(
        os.listdir(os.path.join(root, "test")), max_patient=8
    )

    print(f"Dataset size (first 8 patients): Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"), train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(f"Dataloader batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    return train_loader, test_loader, val_loader


def train_seizure_detection(args):
    # get data loaders for first 8 patients
    train_loader, test_loader, val_loader = prepare_CHB_MIT_8patients_dataloader(args)
    
    # Debug: Check first batch
    print("\nInspecting first batch of data:")
    for X_batch, y_batch in train_loader:
        print(f"  Input batch shape: {X_batch.shape}")
        print(f"  Label batch shape: {y_batch.shape}")
        print(f"  Input dtype: {X_batch.dtype}")
        print(f"  Input device: {X_batch.device}")
        break

    # define CNNTransformer model
    print(f"\nModel configuration:")
    print(f"  in_channels: {args.in_channels}")
    print(f"  n_classes: {args.n_classes}")
    print(f"  fft (token_size): {args.token_size}")
    print(f"  steps: {args.hop_length // 5}")
    
    model = CNNTransformer(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        fft=args.token_size,
        steps=args.hop_length // 5,
        dropout=0.2,
        nhead=4,
        emb_size=256,
    )

    lightning_model = LitModel_finetune(args, model)

    # logger and callbacks
    version = f"CHB_MIT_8patients-CNNTransformer-lr{args.lr}-bs{args.batch_size}-sr{args.sampling_rate}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log_seizure_detection",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=10, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    # train the model
    print("Starting training for seizure detection on CHB-MIT (first 8 patients)...")
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    print("Testing the model...")
    test_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(f"Final Test Results: {test_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNNTransformer for seizure detection on CHB-MIT (first 8 patients)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--in_channels", type=int, default=16, help="number of input channels (CHB-MIT has 16)")
    parser.add_argument("--sample_length", type=float, default=10, help="length (s) of sample")
    parser.add_argument("--n_classes", type=int, default=1, help="number of output classes (binary)")
    parser.add_argument("--sampling_rate", type=int, default=200, help="sampling rate (Hz)")
    parser.add_argument("--token_size", type=int, default=200, help="token size (FFT window)")
    parser.add_argument("--hop_length", type=int, default=100, help="token hop length (STFT hop)")
    
    args = parser.parse_args()
    print("="*80)
    print("Seizure Detection Training - CHB-MIT Dataset (First 8 Patients)")
    print("Model: CNNTransformer")
    print("="*80)
    print(args)
    print("="*80)

    train_seizure_detection(args)
