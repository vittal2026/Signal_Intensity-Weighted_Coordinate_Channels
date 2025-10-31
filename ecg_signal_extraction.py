import wfdb
import numpy as np
import pandas as pd
import os

# --- Parameters ---
records = [418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430,
           602, 605, 607, 609, 610, 611, 612, 614, 615]

data_dir = r"C:\Users\User\Desktop\mit-bih-malignant-ventricular-ectopy-database-1.0.0\mit-bih-malignant-ventricular-ectopy-database-1.0.0"
output_dir = r"C:\Users\User\Desktop\danger_windows_refined_padded"
os.makedirs(output_dir, exist_ok=True)

danger_labels = ['VF', 'VFIB', 'VFL', 'VT', 'ASYS', 'HGEA']
pre_sec = 10
post_sec = 10
window_len_sec = pre_sec + post_sec  # 20 sec total

log = []

for rec in records:
    rec_path = os.path.join(data_dir, str(rec))
    try:
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f"Error reading record {rec}: {e}")
        continue

    sig = record.p_signal
    fs = record.fs  # sampling frequency
    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)
    window_len = pre_samples + post_samples  # 20 sec in samples

    # --- Collect dangerous rhythm change points (partial match) ---
    danger_points = [s for s, note in zip(ann.sample, ann.aux_note)
                     if note and any(dr in note for dr in danger_labels)]
    danger_points = sorted(danger_points)

    print(f"Record {rec}: found {len(danger_points)} danger points")
    if not danger_points:
        continue

    last_danger_point = -np.inf  # for spacing logic

    for i, s in enumerate(danger_points):
        # Check if previous danger point is close (<10 sec)
        if i == 0 or s - danger_points[i-1] >= 10 * fs:
            # Normal window around this danger point
            start = max(0, s - pre_samples)
            end = min(s + post_samples, sig.shape[0])
            rel_time = min(pre_samples, s - start)
        else:
            # Danger point too close to previous → extend previous window
            prev_s = danger_points[i-1]
            start = prev_s
            end = min(prev_s + window_len, sig.shape[0])
            rel_time = s - start

        # Extract segment
        segment = sig[int(start):int(end), :]

        # Pad if segment too short
        if segment.shape[0] < window_len:
            pad_width = window_len - segment.shape[0]
            segment = np.pad(segment, ((0, pad_width), (0, 0)), 'constant')

        # Save window
        filename = f"{rec}_episode{i}.npy"
        np.save(os.path.join(output_dir, filename), segment)

        # Log
        log.append({
            'record': rec,
            'episode_index': i,
            'window_file': filename,
            'absolute_time_s': s / fs,
            'relative_time_s': rel_time / fs
        })

        # Update last danger point
        last_danger_point = s

# --- Save CSV ---
csv_path = os.path.join(output_dir, 'episodes_log.csv')
pd.DataFrame(log).to_csv(csv_path, index=False)
print("✅ Done. Windows saved at:", output_dir)
print("CSV log saved at:", csv_path)
