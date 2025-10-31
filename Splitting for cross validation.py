import os
import random
import pandas as pd

# === Paths ===
root_dir = r'D:\CNN project\Current\standardised data'
invalid_csv = r'D:\CNN project\Current\invalid_samples.csv'
output_dir = r'D:\CNN project\Current\fold_splits'
os.makedirs(output_dir, exist_ok=True)

# === Step 1: Get base filenames ===
all_files = os.listdir(root_dir)
base_names = sorted({f.split('.')[0].replace('_nuc', '') for f in all_files if f.endswith('.bmp')})

# === Step 2: Load invalid filenames (can be .bmp or _nuc.dat) ===
invalid_df = pd.read_excel(invalid_csv) if invalid_csv.endswith('.ods') else pd.read_csv(invalid_csv)

invalid_names = set(
    invalid_df.iloc[:, 0]
    .astype(str)
    .str.replace('.bmp', '', regex=False)
    .str.replace('_nuc.dat', '', regex=False)
    .str.strip()
)


# === Step 3: Filter valid base filenames ===
valid_base_names = [name for name in base_names if name not in invalid_names]
total_valid = len(valid_base_names)

# === Step 4: Shuffle and split into 5 folds ===
random.seed(42)
random.shuffle(valid_base_names)
folds = [valid_base_names[i::5] for i in range(5)]  # round-robin split

# === Step 5: Create wide-format DataFrame ===
max_len = max(len(f) for f in folds)
wide_df = pd.DataFrame({f"fold_{i+1}": pd.Series(folds[i]) for i in range(5)})

# === Step 6: Create filename → subset DataFrame ===
filename_to_subset = []
fold_sizes = {}
for i, fold in enumerate(folds):
    fold_name = f"fold_{i+1}"
    fold_sizes[fold_name] = len(fold)
    for filename in fold:
        filename_to_subset.append((filename, fold_name))
df_filename_to_subset = pd.DataFrame(filename_to_subset, columns=["filename", "subset"])

# === Step 7: Save CSVs ===
wide_df.to_csv(os.path.join(output_dir, 'subset_to_filenames.csv'), index=False)
df_filename_to_subset.to_csv(os.path.join(output_dir, 'filename_to_subset.csv'), index=False)

# === Step 8: Save fold size summary ===
summary_path = os.path.join(output_dir, 'fold_sizes.txt')
with open(summary_path, 'w') as f:
    f.write("Fold Sizes Summary\n\n")
    for fold_name, count in fold_sizes.items():
        f.write(f"{fold_name}: {count} samples\n")
    f.write(f"\nTotal valid samples: {total_valid}\n")

print("✅ All files created:")
print(" - subset_to_filenames.csv (wide format)")
print(" - filename_to_subset.csv (lookup)")
print(" - fold_sizes.txt (summary)")
