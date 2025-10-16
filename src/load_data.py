import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ✅ Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load UNSW-NB15 data
def load_unsw_data():
    print("\n--- Loading UNSW-NB15 dataset ---")

    # Files in dataset
    train_file = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_file = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("UNSW-NB15 train/test CSVs not found in dataset/")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"[INFO] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"[INFO] Combined shape: {df.shape}")
    return df

# ✅ Clean data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[INFO] After cleaning: {df.shape}")
    return df

# ✅ Encode attack labels into simpler categories
def encode_labels(df):
    attack_map = {
        'Normal': 'Normal',
        'Reconnaissance': 'Reconnaissance',
        'Backdoor': 'Backdoor',
        'DoS': 'DoS',
        'Exploits': 'Exploits',
        'Analysis': 'Analysis',
        'Fuzzers': 'Fuzzers',
        'Worms': 'Worms',
        'Shellcode': 'Shellcode',
        'Generic': 'Generic'
    }
    if 'attack_cat' in df.columns:
        df['Label'] = df['attack_cat'].map(attack_map).fillna('Normal')
    elif 'Label' not in df.columns:
        raise RuntimeError("No attack label column found in UNSW dataset.")
    print(f"[INFO] Unique labels after mapping: {df['Label'].unique()}")
    return df

# ✅ Encode categorical columns
def encode_categoricals(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    encoders = {}

    for col in cat_cols:
        if col == "Label":  # skip final label
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Encoded categorical column: {col} -> {len(le.classes_)} classes")

    # Save encoders for later
    joblib.dump(encoders, os.path.join(MODEL_DIR, "categorical_encoders.pkl"))
    return df

# ✅ Normalize numeric features
def normalize_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')

    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"[INFO] Normalized {len(numeric_cols)} numeric features")
    return df

# ✅ Train/Test split
def train_test_split_data(df):
    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    df = load_unsw_data()
    df = clean_data(df)
    df = encode_labels(df)
    df = encode_categoricals(df)   # ✅ categorical fix
    df = normalize_features(df)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Save processed data
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"[INFO] Saved processed data to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
