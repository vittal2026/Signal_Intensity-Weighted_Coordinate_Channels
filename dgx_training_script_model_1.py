# ================================ #
#          Imports & Setup         #
# ================================ #

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from tqdm import tqdm

# ================================ #
#          Configuration           #
# ================================ #
NUM_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42

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
#           Dataset Class          #
# ================================ #
class FoldwiseLazyDataset(Dataset):
    def __init__(self, image_dirs, centre_dirs):
        self.image_files = []
        self.centre_files = []
        for img_dir, ctr_dir in zip(image_dirs, centre_dirs):
            img_files = sorted(Path(img_dir).glob("*.pt"))
            ctr_files = sorted(Path(ctr_dir).glob("*_nuc.pt"))

            base_to_img = {p.stem: p for p in img_files}
            base_to_ctr = {p.stem[:-4]: p for p in ctr_files}  # strip '_nuc'

            for key in sorted(set(base_to_img) & set(base_to_ctr)):
                self.image_files.append(base_to_img[key])
                self.centre_files.append(base_to_ctr[key])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = torch.load(self.image_files[idx], map_location='cpu')
        centre = torch.load(self.centre_files[idx], map_location='cpu')
        return image, centre

# ================================ #
#             Model                #
# ================================ #
class LakshyaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.learnable_pool = nn.Conv2d(128, 128, kernel_size=32, groups=128)
        self.fc = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.learnable_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x) * 255.0

# ================================ #
#           Loss Function          #
# ================================ #
class StandardizedMSELoss(nn.Module):
    def __init__(self, target_mean, target_std):
        super().__init__()
        self.register_buffer('mean', target_mean)
        self.register_buffer('std', target_std)
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_std = (pred - self.mean) / self.std
        target_std = (target - self.mean) / self.std
        return self.mse(pred_std, target_std)

# ================================ #
#          Data Loaders            #
# ================================ #
def get_foldwise_dataloaders(dataset_type, test_fold):
    base_data_path = Path("/home/bsmse2/CNN_project/data/sipakmed_data")
    base_ctr_path = base_data_path / "centres"

    train_folds = [f"fold_{i}" for i in range(1, 6) if f"fold_{i}" != test_fold]
    test_folds = [test_fold]

    train_img_dirs = [base_data_path / dataset_type / fold for fold in train_folds]
    train_ctr_dirs = [base_ctr_path / fold for fold in train_folds]
    test_img_dirs = [base_data_path / dataset_type / fold for fold in test_folds]
    test_ctr_dirs = [base_ctr_path / fold for fold in test_folds]

    train_dataset = FoldwiseLazyDataset(train_img_dirs, train_ctr_dirs)
    test_dataset = FoldwiseLazyDataset(test_img_dirs, test_ctr_dirs)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, train_dataset

# ================================ #
#         Train + Validate         #
# ================================ #
def train_and_validate(dataset_type, test_fold):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Training on {dataset_type}, test fold: {test_fold}")
    print(f"📦 Using device: {device}\n")

    train_loader, test_loader, train_dataset = get_foldwise_dataloaders(dataset_type, test_fold)

    sample_centres = [train_dataset[i][1] for i in range(min(10000, len(train_dataset)))]
    centres_tensor = torch.stack(sample_centres)
    target_mean = centres_tensor.mean(dim=0).to(device)
    target_std = centres_tensor.std(dim=0).to(device)
    target_std[target_std < 1e-6] = 1.0

    model = LakshyaNet().to(device)
    criterion = StandardizedMSELoss(target_mean, target_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_dir = Path("/home/bsmse2/CNN_project/logs/sipakmed_logs/model_1")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset_type}_{test_fold}_log.csv"

    history = {"epoch": [], "train_r2_h": [], "train_r2_k": [], "test_r2_h": [], "test_r2_k": []}

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

        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()

        train_r2_h = r2_score(train_targets[:, 0], train_preds[:, 0])
        train_r2_k = r2_score(train_targets[:, 1], train_preds[:, 1])
        test_r2_h = r2_score(test_targets[:, 0], test_preds[:, 0])
        test_r2_k = r2_score(test_targets[:, 1], test_preds[:, 1])

        print(f"📉 Epoch {epoch+1:02d} | Train R² h/k: {train_r2_h:.4f}/{train_r2_k:.4f} | "
              f"Test R² h/k: {test_r2_h:.4f}/{test_r2_k:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_r2_h"].append(train_r2_h)
        history["train_r2_k"].append(train_r2_k)
        history["test_r2_h"].append(test_r2_h)
        history["test_r2_k"].append(test_r2_k)

    # Save metrics
    pd.DataFrame(history).to_csv(log_file, index=False)
    print(f"\n📊 R² metrics saved to: {log_file}")

    # Save trained model
    model_dir = Path("/home/bsmse2/CNN_project/models/sipakmed/model_1")
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{dataset_type}_{test_fold}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}\n")

# ================================ #
#              Run                 #
# ================================ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="control_set",
                        help="Dataset type: control_set or study_set")
    parser.add_argument("--test_fold", type=str, default="fold_5",
                        help="Test fold: fold_1 to fold_5")
    args = parser.parse_args()

    train_and_validate(args.dataset, args.test_fold)
