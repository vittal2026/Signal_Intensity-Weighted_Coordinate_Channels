import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

# Paths
npy_dir = r"D:\CNN project\Current\geometrically_augmented"
output_dir = r"D:\CNN project\Current\control_set"
os.makedirs(output_dir, exist_ok=True)

# Get all .npy files
npy_files = sorted(glob(os.path.join(npy_dir, "*.npy")))

# Create coordinate channels (shape: 256×256)
H, W = 256, 256
x_channel = np.tile(np.arange(W), (H, 1)).astype(np.float32)
y_channel = np.tile(np.arange(H).reshape(H, 1), (1, W)).astype(np.float32)

# Process all images
for npy_path in tqdm(npy_files, desc="Tensorizing with coord channels"):
    base = os.path.splitext(os.path.basename(npy_path))[0]
    
    # Load image (256×256×3)
    img = np.load(npy_path).astype(np.float32)
    if img.shape != (H, W, 3):
        print(f"Skipping {base}: unexpected shape {img.shape}")
        continue

    # Stack coordinate channels (no normalization)
    stacked = np.stack([img[..., 0], img[..., 1], img[..., 2], x_channel, y_channel], axis=0)  # shape: 5×256×256

    # Save as .pt tensor
    tensor = torch.tensor(stacked)
    torch.save(tensor, os.path.join(output_dir, f"{base}.pt"))

input('Hit ENTER to close:')
