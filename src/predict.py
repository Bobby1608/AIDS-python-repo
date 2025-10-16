import os
import joblib
import numpy as np
import pandas as pd

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
NEW_FILE = os.path.join(BASE_DIR, "dataset", "new_traffic.csv")

# --------------------------
# Load Models & Label Encoder
# --------------------------
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
mlp = joblib.load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
class_names = label_encoder.classes_

print(f"[INFO] Loaded models and label encoder.")

# --------------------------
# Load New Data
# --------------------------
if not os.path.exists(NEW_FILE):
    raise FileNotFoundError(f"[ERROR] {NEW_FILE} not found. Please place your traffic CSV there.")

new_df = pd.read_csv(NEW_FILE)

# Drop non-feature columns if exist (like 'id', 'attack_cat', 'label')
drop_cols = [col for col in ["attack_cat", "label", "id"] if col in new_df.columns]
new_df = new_df.drop(columns=drop_cols, errors="ignore")

# Ensure correct feature alignment
X_new = new_df.to_numpy()
print(f"[INFO] Loaded new traffic: {X_new.shape}")

# --------------------------
# Run Predictions
# --------------------------
results = []
for model_name, model in [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]:
    preds = model.predict(X_new)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)
    else:
        probs = np.eye(len(class_names))[preds]

    decoded_preds = label_encoder.inverse_transform(preds)

    print(f"\n[INFO] Predictions with {model_name}:")
    for i, (cls, prob) in enumerate(zip(decoded_preds, probs)):
        print(f"Sample {i+1}: {cls} (confidence {max(prob):.2f})")

        results.append({
            "Model": model_name,
            "Sample": i+1,
            "Predicted": cls,
            "Confidence": max(prob)
        })

# --------------------------
# Save Predictions
# --------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(BASE_DIR, "results", "predictions.csv"), index=False)

print(f"\n[INFO] Predictions saved to results/predictions.csv")
