import os
import numpy as np
import cv2
import json
from glob import glob
from tqdm import tqdm

# === Paths ===
npy_dir = r"D:\CNN project\Current\AllSipakMedFiles"
json_dir = r"D:\CNN project\Current\geometric_jsons"
output_dir = os.path.join(npy_dir, "augmented")
os.makedirs(output_dir, exist_ok=True)

# === Get all .npy image files ===
npy_files = sorted(glob(os.path.join(npy_dir, "*.npy")))

# === Loop through each image ===
for npy_path in tqdm(npy_files, desc="Geometric augmentation"):
    base = os.path.splitext(os.path.basename(npy_path))[0]  # e.g., 1_001_01

    # Load image
    img = np.load(npy_path)
    if img.shape[0] == 3 and img.shape[1] != 3:
        img = np.transpose(img, (1, 2, 0))  # convert from (3, H, W) to (H, W, 3)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    for i in range(20):
        suffix = f"_aug{i:02d}"
        aug_name = f"{base}{suffix}"
        json_file = os.path.join(json_dir, f"{aug_name}.json")

        if not os.path.exists(json_file):
            print(f"Missing JSON: {json_file}")
            continue

        with open(json_file, 'r') as f:
            params = json.load(f)[aug_name]

        tx = params['translation']['t_x']
        ty = params['translation']['t_y']
        theta = params['rotation']['theta']
        alpha = params['scaling']['alpha']

        # Build affine matrix
        M = cv2.getRotationMatrix2D(center, theta, alpha)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply transform to each channel
        transformed = np.zeros_like(img)
        for ch in range(3):
            transformed[:, :, ch] = cv2.warpAffine(
                img[:, :, ch], M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0  # Fill outside region with 0 (strictly)
            )

        # Save transformed image
        out_path = os.path.join(output_dir, f"{aug_name}.npy")
        np.save(out_path, transformed)

input('Hit ENTER to close:')
