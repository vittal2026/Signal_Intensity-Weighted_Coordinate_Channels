import os
import numpy as np
import pandas as pd
from scipy import signal

# --- Parameters ---
input_dir = r"C:\Users\User\Desktop\danger_windows_refined_padded"
output_dir = r"C:\Users\User\Desktop\danger_windows_augmented"
os.makedirs(output_dir, exist_ok=True)

csv_log_file = os.path.join(output_dir, "augmented_windows_log.csv")
episodes_log_file = os.path.join(input_dir, "episodes_log.csv")
n_augment = 100
fs = 250  # Hz
max_shift_sec = 5
max_zero_sec = 2
noise_bounds = (0.01, 0.05)       # fraction of signal amplitude
wander_bounds = (0.01, 0.05)      # fraction of signal amplitude
scaling_bounds = (0.8, 1.2)       # multiplicative factor

# --- Load original episodes log ---
episodes_df = pd.read_csv(episodes_log_file)

# --- Helper functions ---
def augment_ecg_window(x, idx, fs):
    """
    x: np.ndarray, shape (samples, channels)
    idx: index of change point in samples
    returns augmented x and new idx
    """
    n_samples, n_ch = x.shape
    # --- Step 1: Uniform shift ---
    shift_sec = np.random.uniform(-max_shift_sec, max_shift_sec)
    shift_samples = int(shift_sec * fs)
    
    if shift_samples > 0:
        pad_shape = (shift_samples, n_ch)
        x = np.concatenate([np.zeros(pad_shape), x[:-shift_samples]])
    elif shift_samples < 0:
        pad_shape = (-shift_samples, n_ch)
        x = np.concatenate([x[-shift_samples:], np.zeros(pad_shape)])
    idx = np.clip(idx + shift_samples, 0, n_samples-1)
    
    # --- Step 2: Random patch zeroing (1-2 s) ---
    n_zero = int(np.random.uniform(1, max_zero_sec) * fs)
    start_zero = np.random.randint(0, n_samples - n_zero)
    # Avoid zeroing the change point
    if not (idx >= start_zero and idx < start_zero + n_zero):
        x[start_zero:start_zero+n_zero, :] = 0
    
    # --- Step 3: High frequency Gaussian noise ---
    amp = np.max(np.abs(x))
    noise_std = np.random.uniform(*noise_bounds) * amp
    x += np.random.normal(0, noise_std, x.shape)
    
    # --- Step 4: Baseline wander ---
    wander_amp = np.random.uniform(*wander_bounds) * amp
    t = np.arange(n_samples)/fs
    freq = np.random.uniform(0.1, 0.5)  # Hz
    x += wander_amp * np.sin(2*np.pi*freq*t)[:, None]
    
    # --- Step 5: Frequency domain filter ---
    # Randomly apply lowpass/highpass/bandpass
    ftype = np.random.choice(['low', 'high', 'band'])
    nyq = fs/2
    if ftype == 'low':
        cutoff = np.random.uniform(30, 50)/nyq
        b, a = signal.butter(3, cutoff, btype='low')
    elif ftype == 'high':
        cutoff = np.random.uniform(0.5, 2)/nyq
        b, a = signal.butter(3, cutoff, btype='high')
    else:  # bandpass
        low = np.random.uniform(0.5, 2)/nyq
        high = np.random.uniform(30, 50)/nyq
        if high <= low: high = low + 0.1
        b, a = signal.butter(3, [low, high], btype='band')
    x = signal.filtfilt(b, a, x, axis=0)
    
    # --- Step 6: Scaling ---
    scale = np.random.uniform(*scaling_bounds)
    x *= scale
    
    return x, idx

# --- Augmentation loop ---
augmented_rows = []
for i, row in episodes_df.iterrows():
    fname = row['window_file']
    change_sec = row['relative_time_s']
    path = os.path.join(input_dir, fname)
    x = np.load(path)
    
    idx = int(change_sec * fs)
    
    for n in range(1, n_augment+1):
        x_aug, idx_aug = augment_ecg_window(np.copy(x), idx, fs)
        aug_fname = fname.replace('.npy', f'_aug{n:03d}.npy')
        np.save(os.path.join(output_dir, aug_fname), x_aug)
        augmented_rows.append({
            'original_file': fname,
            'aug_file': aug_fname,
            'aug_index': n,
            'relative_time_s': idx_aug/fs
        })

# --- Save augmented CSV log ---
pd.DataFrame(augmented_rows).to_csv(csv_log_file, index=False)
print("✅ Done. Augmented windows saved in:", output_dir)
