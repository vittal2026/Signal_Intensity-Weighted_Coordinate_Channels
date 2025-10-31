import os
import json
import random
import pandas as pd

# --- Settings ---
csv_path = "D:/CNN project/Current/fold_splits/filename_to_subset.csv"
output_dir = "D:/CNN project/Current/geometric_jsons"
os.makedirs(output_dir, exist_ok=True)

# --- Load filenames from CSV ---
df = pd.read_csv(csv_path)
filenames = df.iloc[:, 0].tolist()  # Use first column as before

# --- Geometric parameter generator ---
def generate_geometric_params():
    return {
        "translation": {
            "t_x": round(random.uniform(-20, 20), 2),
            "t_y": round(random.uniform(-20, 20), 2)
        },
        "rotation": {
            "theta": round(random.uniform(-20, 20), 2)
        },
        "scaling": {
            "alpha": round(random.uniform(0.75, 1.25), 3)
        }
    }

# --- Generate JSONs ---
for fname in filenames:
    base = os.path.splitext(fname)[0]  # e.g., 1_001_01
    for i in range(20):
        aug_name = f"{base}_aug{i:02d}"
        param_dict = {aug_name: generate_geometric_params()}
        out_path = os.path.join(output_dir, f"{aug_name}.json")
        with open(out_path, 'w') as f:
            json.dump(param_dict, f, indent=2)
