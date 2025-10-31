'''
import os
import shutil

# Base path where class folders are located
base_dir = r"D:\CNN project\Current\SipakMed"
# Destination folder for all renamed files
dest_dir = r"D:\CNN project\Current\AllSipakMedFiles"

# Create destination folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Map of folder names to their prefix
class_prefixes = {
    "Dyskeratotic": "1_",
    "Koilocytotic": "2_",
    "Metaplastic": "3_",
    "Parabasal": "4_",
    "Superficial-Intermediate": "5_"
}

# Process each class folder
for class_name, prefix in class_prefixes.items():
    class_path = os.path.join(base_dir, class_name)
    for filename in os.listdir(class_path):
        src_path = os.path.join(class_path, filename)
        if os.path.isfile(src_path):
            # Ensure filename does not already have prefix
            new_name = prefix + filename
            dest_path = os.path.join(dest_dir, new_name)

            # If same name somehow still exists, make unique
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                new_name = f"{prefix}{name}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, new_name)
                counter += 1

            shutil.move(src_path, dest_path)

print("✅ All files moved and renamed successfully.")
'''
'''
import os

# Folder containing all combined files
combined_folder = r"D:\CNN project\Current\AllSipakMedFiles"

# Loop through files and delete those ending with '_cyt.dat'
for filename in os.listdir(combined_folder):
    if filename.endswith('_cyt.dat'):
        file_path = os.path.join(combined_folder, filename)
        os.remove(file_path)

print("🗑️ All '_cyt.dat' files deleted successfully.")
'''
'''
import os
import csv

# Paths
csv_path = r"D:\CNN project\Current\invalid_samples.csv"
target_folder = r"D:\CNN project\Current\AllSipakMedFiles"

# Step 1: Read 8-character IDs from CSV (first column, skip header)
with open(csv_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    invalid_ids = {row[0][:8] for row in reader if row}  # Take first 8 chars

# Step 2: Browse target folder and delete matching files
for filename in os.listdir(target_folder):
    if filename[:8] in invalid_ids:
        file_path = os.path.join(target_folder, filename)
        os.remove(file_path)

print("🗑️ Files with matching 8-character prefixes deleted.")
'''

import os
import cv2
import numpy as np

# Directory containing both .bmp and .dat files
folder = r"D:\CNN project\Current\AllSipakMedFiles"

# Loop through each .bmp file
for filename in os.listdir(folder):
    if filename.lower().endswith('.bmp'):
        bmp_path = os.path.join(folder, filename)
        base_name = os.path.splitext(filename)[0]

        # Read image
        img = cv2.imread(bmp_path)
        orig_h, orig_w = img.shape[:2]

        # Determine scaling factor
        scale = 256 / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Resize image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save resized image (overwrite original)
        cv2.imwrite(bmp_path, resized_img)

        # Process corresponding .dat file if it exists
        dat_path = os.path.join(folder, base_name + '_nuc.dat')
        if os.path.exists(dat_path):
            # Load and scale coordinates
            coords = np.loadtxt(dat_path, delimiter=',')
            if coords.ndim == 1:
                coords = coords[np.newaxis, :]  # For single-point files

            coords[:, 0] *= scale  # x
            coords[:, 1] *= scale  # y

            # Save updated coordinates (overwrite original)
            np.savetxt(dat_path, coords, fmt='%.3f')

print("✅ All images resized and contour coordinates adjusted.")
