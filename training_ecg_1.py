# ================================ #
#          Imports & Setup         #
# ================================ #
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.signal import medfilt  # <-- added for median filter

# ================================ #
#          Configuration           #
# ================================ #
NUM_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42
TARGET_LENGTH = 500  # new compressed length
MEDIAN_WINDOW = 11   # must be odd

# ================================ #
#         Reproducibility          #
# ================================ #
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ================================ #
#      ECG Compression Utility     #
# ================================ #
def compress_ecg(ecg, target_length=TARGET_LENGTH, window=MEDIAN_WINDOW):
    """
    Compress 3-channel ECG from original_length -> target_length using
    median filter + stride downsampling.
    """
    c, L = ecg.shape
    stride = L // target_length
    compressed_ecg = torch.zeros((c, target_length), dtype=ecg.dtype)
    
    for ch in range(c):
        signal = ecg[ch].numpy()
        filtered = medfilt(signal, kernel_size=window)
        compressed = filtered[::stride]
        # Handle rounding mismatches
        if len(compressed) > target_length:
            compressed = compressed[:target_length]
        elif len(compressed) < target_length:
            compressed = np.pad(compressed, (0, target_length - len(compressed)), mode='edge')
        compressed_ecg[ch] = torch.tensor(compressed, dtype=ecg.dtype)
    
    return compressed_ecg

# ================================ #
#           Dataset Class          #
# ================================ #
class FoldwiseLazyDataset(Dataset):
    """Lazy-loading dataset from pre-saved .pt folds, with automatic compression."""
    def __init__(self, fold_paths):
        self.data = []
        for fpath in fold_paths:
            fold_data = torch.load(fpath, map_location='cpu')
            self.data.extend(fold_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if isinstance(sample, dict):
            x = sample.get('input', None)
            y = sample.get('output', None)
        else:
            x, y = sample, None

        # Ensure x shape is (channels, length)
        if x.shape[0] in [5000, 500] and x.shape[1] == 3:
            x = x.T
        x = x.float()

        # Apply median-filter + compression
        if x.shape[1] > TARGET_LENGTH:
            x = compress_ecg(x, target_length=TARGET_LENGTH, window=MEDIAN_WINDOW)

        # Make target a 1D tensor of shape (1,)
        if y is None:
            y = torch.tensor([0.0], dtype=torch.float32)
        elif torch.is_tensor(y):
            y = y.float().reshape(-1)  # shape (1,)
        else:
            y = torch.tensor([y], dtype=torch.float32)

        return x, y

# ================================ #
#             Model                #
# ================================ #
class WeightedAverage1D(nn.Module):
    """
    Learnable weighted average along sequence dimension,
    with independent weights for each input channel.
    Each channel's weights are normalized (sum to 1).
    """
    def __init__(self, channels=3, length=500):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.randn(channels, length))

    def forward(self, x):
        # x: (batch, channels, length)
        w = self.raw_weights
        # Normalize weights per channel so that they sum to 1
        # Weighted sum across sequence per channel
        x_weighted = (x * w[None, :, :]).sum(dim=2)  # (batch, channels)
        return x_weighted


class NimeshaNet(nn.Module):
    """1D CNN for 3-channel input → scalar output"""
    def __init__(self, input_length=TARGET_LENGTH):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.AvgPool1d(2),

        )

        # Calculate length after 4 pooling layers
        pooled_length = input_length
        for _ in range(5):
            pooled_length = pooled_length // 2

        self.weighted_pool = WeightedAverage1D(channels=128, length=pooled_length)

        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.weighted_pool(x)  # learnable weighted average
        return self.fc(x) * 20.0

# ================================ #
#          Loss Function           #
# ================================ #
class StandardizedMSELoss(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_std = (pred - self.mean) / self.std
        target_std = (target - self.mean) / self.std
        return self.mse(pred_std, target_std)

# ================================ #
#          Data Loaders            #
# ================================ #
def get_loaders(dataset_type, test_fold):
    base_path = Path("/home/bsmse2/CNN_project/data/ecg_data") / dataset_type
    folds = [f"fold_{i}" for i in range(1,6)]
    train_folds = [f for f in folds if f != test_fold]

    train_files = [base_path / f"{fold}.pt" for fold in train_folds]
    test_files = [base_path / f"{test_fold}.pt"]

    train_dataset = FoldwiseLazyDataset(train_files)
    test_dataset = FoldwiseLazyDataset(test_files)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )

    return train_loader, test_loader, train_dataset

# ================================ #
#         Training Loop            #
# ================================ #
def train_and_validate(dataset_type, test_fold):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training {dataset_type} | Test fold: {test_fold} | Device: {device}")

    train_loader, test_loader, train_dataset = get_loaders(dataset_type, test_fold)

    # Compute mean/std of targets
    sample_targets = torch.cat([train_dataset[i][1] for i in range(len(train_dataset))])
    target_mean = sample_targets.mean().to(device)
    target_std = sample_targets.std().to(device)
    if target_std < 1e-6:
        target_std = 1.0

    model = NimeshaNet().to(device)
    criterion = StandardizedMSELoss(target_mean, target_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_dir = Path("/home/bsmse2/CNN_project/logs/ecg_logs/model_1")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset_type}_{test_fold}_log.csv"

    history = {"epoch": [], "train_r2": [], "test_r2": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_preds, train_targets = [], []

        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_preds.append(pred.detach().cpu())
            train_targets.append(y.detach().cpu())

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"[Epoch {epoch+1}] Testing", leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_preds.append(pred.cpu())
                test_targets.append(y.cpu())

        train_preds = torch.cat(train_preds).numpy().reshape(-1)
        train_targets = torch.cat(train_targets).numpy().reshape(-1)
        test_preds = torch.cat(test_preds).numpy().reshape(-1)
        test_targets = torch.cat(test_targets).numpy().reshape(-1)

        train_r2 = r2_score(train_targets, train_preds)
        test_r2 = r2_score(test_targets, test_preds)

        print(f"📉 Epoch {epoch+1:02d} | Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")

        history["epoch"].append(epoch+1)
        history["train_r2"].append(train_r2)
        history["test_r2"].append(test_r2)

    # Save metrics
    pd.DataFrame(history).to_csv(log_file, index=False)
    print(f"\n📊 Metrics saved to {log_file}")

    # Save model
    model_dir = Path("/home/bsmse2/CNN_project/models/ecg/model_1")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{dataset_type}_{test_fold}_final.pth"
    torch.save(model.state_dict(), model_file)
    print(f"\n✅ Model saved to {model_file}")

# ================================ #
#              Run                 #
# ================================ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Study", help="Study or Baseline")
    parser.add_argument("--test_fold", type=str, default="fold_4", help="fold_1 to fold_5")
    args = parser.parse_args()

    train_and_validate(args.dataset, args.test_fold)
