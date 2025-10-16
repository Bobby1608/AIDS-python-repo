import os
import numpy as np
from load_data import (
    load_all_csvs,
    clean_data,
    encode_labels,
    extract_flow_features,
    normalize_features,
    train_test_split_data
)

# ✅ Get correct paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

# ✅ Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_numpy_arrays(X_train, X_test, y_train, y_test):
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    print(f"[INFO] Saved .npy files to '{OUTPUT_DIR}/'")

def main():
    # 1. Load + Clean + Preprocess
    df, payload_files = load_all_csvs(DATA_DIR)
    df = clean_data(df)
    df = encode_labels(df)
    df = extract_flow_features(df)
    df = normalize_features(df)

    # 2. Train/test split
    df, X_train, X_test, y_train, y_test = train_test_split_data(df)

    # 3. Convert to NumPy arrays
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    # 4. Save arrays
    save_numpy_arrays(X_train_np, X_test_np, y_train_np, y_test_np)

if __name__ == "__main__":
    main()
