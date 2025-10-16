# src/webapp/extract_stream_balanced.py
import os
import time
import json
import joblib
import numpy as np
from flask import Flask, Response

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

# --------------------------
# Load models
# --------------------------
rf = safe_load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb = safe_load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
mlp = safe_load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
label_encoder = safe_load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

def decode(preds):
    arr = np.asarray(preds)
    if arr.dtype.kind in ("i", "u", "f"):
        if label_encoder is not None:
            return list(label_encoder.inverse_transform(arr.astype(int)))
        else:
            return [str(int(x)) for x in arr]
    else:
        return [str(x) for x in arr]

# --------------------------
# Sample rows
# --------------------------
def sample_rows(n=10):
    x_path = os.path.join(PROCESSED_DIR, "X_test.npy")
    if os.path.exists(x_path):
        X = np.load(x_path)
        idx = np.random.choice(len(X), size=n, replace=False)
        return X[idx]
    else:
        feat = rf.n_features_in_ if rf is not None else 45
        return np.random.rand(n, feat)

# --------------------------
# SSE Stream
# --------------------------
app = Flask(__name__)

@app.route("/stream")
def stream():
    def gen():
        while True:
            X = sample_rows(10)  # 10 samples per batch
            preds = []
            for i in range(len(X)):
                row_preds = {}
                for name, model in [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]:
                    if model is None:
                        row_preds[name] = "Simulated"
                        continue
                    try:
                        p = model.predict([X[i]])
                        decoded = decode(p)[0]
                        row_preds[name] = decoded
                    except Exception:
                        row_preds[name] = "Error"
                preds.append(row_preds)

            yield f"data: {json.dumps(preds)}\n\n"
            time.sleep(3)  # refresh every 3s
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    print("[INFO] Starting extract_stream_balanced (10 predictions every 3s)...")
    app.run(debug=True, port=5001)  # separate port from app.py
