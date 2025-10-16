# src/check_attacks.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # MyNetworkIDSProject/src
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --------------------------
# Load Data
# --------------------------
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)

print(f"[INFO] Loaded test data: {X_test.shape}, Labels: {y_test.shape}")

# --------------------------
# Load Models & Label Encoder
# --------------------------
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
mlp = joblib.load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
classes = label_encoder.classes_

print(f"[INFO] Models & encoder loaded. Classes: {list(classes)}")

# --------------------------
# Helper Function
# --------------------------
def evaluate_model(name, model, X, y_true):
    print(f"\n===== {name} =====")
    preds = model.predict(X)
    decoded_preds = label_encoder.inverse_transform(preds)

    # Classification report
    report = classification_report(
        y_true, decoded_preds, target_names=classes, zero_division=0
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, decoded_preds, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print("\nConfusion Matrix:")
    print(cm_df)

# --------------------------
# Run Evaluations
# --------------------------
evaluate_model("Random Forest", rf, X_test, y_test)
evaluate_model("XGBoost", xgb, X_test, y_test)
evaluate_model("MLP", mlp, X_test, y_test)
