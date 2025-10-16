import os
import sys
import numpy as np
import pandas as pd

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# === Default values ===
TARGET_CLASS = "DoS"
N_SAMPLES = 5

# === Parse command line arguments ===
if len(sys.argv) > 1:
    TARGET_CLASS = sys.argv[1]
if len(sys.argv) > 2:
    N_SAMPLES = int(sys.argv[2])

# === Load Test Data ===
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)

# === Check available classes ===
unique_classes = np.unique(y_test)
print(f"[INFO] Available classes: {unique_classes}")

if TARGET_CLASS not in unique_classes:
    raise ValueError(f"[ERROR] Class '{TARGET_CLASS}' not found in test set.")

# === Select samples of chosen class ===
sample_indices = np.where(y_test == TARGET_CLASS)[0][:N_SAMPLES]

if len(sample_indices) == 0:
    raise ValueError(f"[ERROR] No samples found for class '{TARGET_CLASS}'.")

sample_df = pd.DataFrame(X_test[sample_indices])

# === Save to new_traffic.csv ===
output_path = os.path.join(DATASET_DIR, "new_traffic.csv")
sample_df.to_csv(output_path, index=False)

print(f"[INFO] Saved {len(sample_df)} samples of '{TARGET_CLASS}' traffic to {output_path}")
