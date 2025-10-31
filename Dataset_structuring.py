import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

# ------------------------- #
# ----- USER SETTINGS ----- #
# ------------------------- #
aug_dir = r"C:\Users\User\Desktop\danger_windows_augmented"
log_csv = os.path.join(aug_dir, "augmented_windows_log.csv")
output_dir = r"C:\Users\User\Desktop\ECG_5fold_dataset"
os.makedirs(output_dir, exist_ok=True)

n_folds = 5
fs = 250  # sampling frequency
# ------------------------- #

# Load log CSV
log = pd.read_csv(log_csv)

# Group augmented files by original file
groups = log.groupby('original_file')['aug_file'].apply(list).to_dict()

# Randomly assign each original file to a fold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
original_files = list(groups.keys())
fold_assignments = {}
for fold_idx, (_, test_idx) in enumerate(kf.split(original_files), 1):
    for idx in test_idx:
        orig_file = original_files[idx]
        for aug_file in groups[orig_file]:
            fold_assignments[aug_file] = fold_idx

# Save fold mapping CSV
fold_map_csv = os.path.join(output_dir, "fold_mapping.csv")
pd.DataFrame.from_dict(fold_assignments, orient='index', columns=['fold']) \
    .rename_axis('aug_file') \
    .reset_index() \
    .to_csv(fold_map_csv, index=False)
print("✅ Fold mapping CSV saved at:", fold_map_csv)

# ------------------------- #
# --- Prepare datasets ---- #
# ------------------------- #

study_dir = os.path.join(output_dir, "Study")
baseline_dir = os.path.join(output_dir, "Baseline")
os.makedirs(study_dir, exist_ok=True)
os.makedirs(baseline_dir, exist_ok=True)

# Prepare empty lists for each fold
study_folds = {i: [] for i in range(1, n_folds+1)}
baseline_folds = {i: [] for i in range(1, n_folds+1)}

for idx, row in log.iterrows():
    aug_file = row['aug_file']
    rel_time = row['relative_time_s']
    fold = fold_assignments[aug_file]

    # Load augmented signal (5000 x 2)
    seg = np.load(os.path.join(aug_dir, aug_file))  # shape: (5000, 2)

    # --- Study coordinate channel ---
    weighted_coord = np.mean(seg, axis=1, keepdims=True) * np.linspace(0, 20, seg.shape[0])[:, None]
    study_input = np.concatenate([seg, weighted_coord], axis=1)  # shape: (5000, 3)

    # --- Baseline coordinate channel ---
    baseline_coord = np.linspace(0, 20, seg.shape[0])[:, None]
    baseline_input = np.concatenate([seg, baseline_coord], axis=1)

    # Create dictionary
    study_dict = {'input': torch.tensor(study_input, dtype=torch.float32),
                  'output': torch.tensor(rel_time, dtype=torch.float32)}
    baseline_dict = {'input': torch.tensor(baseline_input, dtype=torch.float32),
                     'output': torch.tensor(rel_time, dtype=torch.float32)}

    # Append to respective fold
    study_folds[fold].append(study_dict)
    baseline_folds[fold].append(baseline_dict)

# Save each fold as .pt
for fold in range(1, n_folds+1):
    torch.save(study_folds[fold], os.path.join(study_dir, f"fold_{fold}.pt"))
    torch.save(baseline_folds[fold], os.path.join(baseline_dir, f"fold_{fold}.pt"))

print("✅ 5-fold datasets saved in Study and Baseline folders.")
