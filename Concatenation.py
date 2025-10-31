import os
import pandas as pd
import torch
from tqdm import tqdm

# Paths
control_set_dir = r"D:\CNN project\Current\data\centres"
fold_csv_path = r"D:\CNN project\Current\fold_splits\filename_to_subset.csv"
output_dir = r"D:\CNN project\Current\foldwise_data\centres"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV
df = pd.read_csv(fold_csv_path)

# Choose the fold you want to process
fold_name = 'fold_5'

# Filter filenames belonging to that fold
base_filenames = df[df['subset'] == fold_name]['filename'].dropna().tolist()
print(f"\n🔎 Fold {fold_name} has {len(base_filenames)} base samples → expecting {len(base_filenames) * 20} with augmentations")

# Prepare to save
fold_data = []
total_samples = len(base_filenames) * 20

with tqdm(total=total_samples, desc=fold_name, unit="sample") as pbar:
    for base_name in base_filenames:
        for aug in range(20):
            aug_name = f"{base_name}_aug{aug:02d}_nuc"
            file_path = os.path.join(control_set_dir, f"{aug_name}.pt")

            if os.path.exists(file_path):
                try:
                    centre_tensor = torch.load(file_path)
                    fold_data.append({
                        'filename': aug_name,
                        'centre': centre_tensor
                    })
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
            else:
                print(f"⚠️ Missing file: {file_path}")

            pbar.update(1)

# Save the fold's data
output_path = os.path.join(output_dir, f"{fold_name}.pt")
torch.save(fold_data, output_path)
print(f"\n✅ Saved {len(fold_data)} samples to {output_path}")
