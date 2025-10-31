import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==== CONFIG ====
base_dir = "D:/CNN project/Current"
fold_csv = os.path.join(base_dir, "fold_splits", "filename_to_subset.csv")

img_src = os.path.join(base_dir, "data", "study_set")

dst_root = "D:/CNN project/foldwise_data"
img_dst_root = os.path.join(dst_root, "study_set")

# ==== PREPARE FOLDER STRUCTURE ====
for fold in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
    os.makedirs(os.path.join(img_dst_root, fold), exist_ok=True)

# ==== LOAD CSV ====
df = pd.read_csv(fold_csv)

# ==== MOVE FILES ====
for base_name, fold in tqdm(df.itertuples(index=False), total=len(df)):
    for i in range(20):
        suffix = f"{base_name}_aug{str(i).zfill(2)}"

        img_file = f"{suffix}.pt"

        src_img_path = os.path.join(img_src, img_file)

        dst_img_path = os.path.join(img_dst_root, fold, img_file)

        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
        else:
            print(f"Missing file(s) for base {suffix}")
