import os
import numpy as np
import pandas as pd

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# === Load Test Data ===
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)

# === Select Samples of Class "DoS" ===
sample_indices = np.where(y_test == "DoS")[0][:5]  # First 5 DoS rows
sample_df = pd.DataFrame(X_test[sample_indices])

# === Save to new_traffic.csv ===
output_path = os.path.join(DATASET_DIR, "new_traffic.csv")
sample_df.to_csv(output_path, index=False)

print(f"[INFO] Sample 'new_traffic.csv' saved to: {output_path}")
