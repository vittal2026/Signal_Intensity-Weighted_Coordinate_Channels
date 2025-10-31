import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

# === Input and output paths ===
dat_dir = r"D:\CNN project\Current\geometrically_augmented_dat"
save_dir = r"D:\CNN project\Current\centres"
os.makedirs(save_dir, exist_ok=True)

# === Get all .dat files ===
dat_files = sorted(glob(os.path.join(dat_dir, "*.dat")))

# === Loop with progress bar ===
for dat_path in tqdm(dat_files, desc="Computing and saving centres"):
    filename = os.path.splitext(os.path.basename(dat_path))[0]  # e.g., '2_093_09_aug00'
    
    # Load x, y coordinates from .dat file
    coords = np.loadtxt(dat_path)

    # Remove last point if it's a duplicate of the first
    if coords.shape[0] > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    # Compute center (mean of x and y)
    center = coords.mean(axis=0)  # shape (2,)
    center_tensor = torch.tensor(center, dtype=torch.float32)  # tensor([h, k])

    # Save to individual .pt file
    torch.save(center_tensor, os.path.join(save_dir, filename + ".pt"))

print(f"✅ Saved {len(dat_files)} centre .pt files to:\n{save_dir}")

input('Hit ENTER to close:')
