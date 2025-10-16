# src/webapp/app_combined2.py
import os
import time
import json
import joblib
import random
import numpy as np
from flask import Flask, Response, render_template, jsonify
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------
# Utility: find folders robustly
# ------------------------------
def find_folder_upwards(start_path, folder_name, max_up=6):
    cur = os.path.abspath(start_path)
    for _ in range(max_up + 1):
        cand = os.path.join(cur, folder_name)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # also check cwd
    cand2 = os.path.join(os.getcwd(), folder_name)
    if os.path.isdir(cand2):
        return os.path.abspath(cand2)
    return None

# ------------------------------
# Project locations (auto-detect)
# ------------------------------
THIS_FILE = os.path.abspath(__file__)
WEBAPP_DIR = os.path.dirname(THIS_FILE)            # .../src/webapp
SRC_DIR = os.path.dirname(WEBAPP_DIR)              # .../src
PROJECT_DIR = os.path.dirname(SRC_DIR)             # .../MyNetworkIDSProject

# try common locations
models_dir = find_folder_upwards(WEBAPP_DIR, "models")
processed_dir = find_folder_upwards(WEBAPP_DIR, "processed")

if models_dir is None:
    cand = os.path.join(PROJECT_DIR, "models")
    if os.path.isdir(cand):
        models_dir = cand

if processed_dir is None:
    cand2 = os.path.join(PROJECT_DIR, "processed")
    if os.path.isdir(cand2):
        processed_dir = cand2

print("Using paths:")
print(" PROJECT_DIR:", PROJECT_DIR)
print(" models_dir:", models_dir)
print(" processed_dir:", processed_dir)

# ------------------------------
# Safe loader
# ------------------------------
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None

rf = xgb = mlp = label_encoder = None
if models_dir:
    rf = safe_load(os.path.join(models_dir, "rf_model.pkl"))
    xgb = safe_load(os.path.join(models_dir, "xgb_model.pkl"))
    mlp = safe_load(os.path.join(models_dir, "mlp_model.pkl"))
    label_encoder = safe_load(os.path.join(models_dir, "label_encoder.pkl"))
else:
    print("[WARN] models directory not found. App will run in simulation mode.")

# ------------------------------
# Decode helper
# ------------------------------
def decode_preds(preds):
    arr = np.asarray(preds)
    if arr.dtype.kind in ("i", "u", "f"):
        try:
            ints = arr.astype(int)
            if label_encoder is not None:
                return list(label_encoder.inverse_transform(ints))
            else:
                return [str(int(x)) for x in ints]
        except Exception:
            return [str(x) for x in arr]
    else:
        return [str(x) for x in arr]

# ------------------------------
# Load test data if available
# ------------------------------
X_test = None
y_test = None
if processed_dir:
    xp = os.path.join(processed_dir, "X_test.npy")
    yp = os.path.join(processed_dir, "y_test.npy")
    if os.path.exists(xp):
        try:
            X_test = np.load(xp, allow_pickle=False)
            print("[INFO] Loaded X_test:", X_test.shape)
            if os.path.exists(yp):
                y_test = np.load(yp, allow_pickle=True)
                print("[INFO] Loaded y_test:", y_test.shape)
        except Exception as e:
            print("[WARN] Failed loading processed arrays:", e)

# ------------------------------
# Feature count for dummy data
# ------------------------------
def get_feature_count():
    for m in (rf, xgb, mlp):
        if m is not None:
            try:
                return int(getattr(m, "n_features_in_"))
            except Exception:
                pass
    if X_test is not None:
        return X_test.shape[1]
    return 45

FEATURE_COUNT = get_feature_count()
print("[INFO] Using feature count:", FEATURE_COUNT)

# ------------------------------
# Sampling rows
# ------------------------------
def sample_rows(n=10):
    if X_test is not None:
        idx = np.random.choice(len(X_test), size=n, replace=False)
        Xs = X_test[idx]
        ys = y_test[idx] if y_test is not None else [None] * n
        return Xs, list(ys)
    else:
        Xs = np.random.rand(n, FEATURE_COUNT)
        return Xs, [None] * n

# ------------------------------
# Batch prediction
# ------------------------------
def predict_batch(X_batch):
    results = []
    model_items = [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]
    for i in range(X_batch.shape[0]):
        sample_entry = {"sample_index": i, "predictions": {}}
        for name, model in model_items:
            if model is None:
                pred_label = random.choice(list(label_encoder.classes_) if label_encoder else ["Normal", "DoS", "Generic"])
                sample_entry["predictions"][name] = {"pred": pred_label, "conf": round(random.uniform(0.5, 0.95), 3)}
                continue
            try:
                x_row = X_batch[i].reshape(1, -1)
                pred = model.predict(x_row)
                decoded = decode_preds(pred)[0]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(x_row)
                    conf = float(np.max(probs[0])) if probs is not None else 1.0
                else:
                    conf = 1.0
                sample_entry["predictions"][name] = {"pred": decoded, "conf": round(conf, 3)}
            except Exception as e:
                print(f"[WARN] Predict failed for {name} on sample {i}: {e}")
                sample_entry["predictions"][name] = {"pred": "Error", "conf": 0.0}
        results.append(sample_entry)
    return results

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_models():
    if X_test is None or y_test is None:
        return {"error": "Processed X_test.npy or y_test.npy not found."}

    report_summary = {}
    for name, model in [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]:
        if model is None:
            report_summary[name] = {"error": "Model not loaded"}
            continue
        try:
            y_pred = model.predict(X_test)
            decoded = decode_preds(y_pred)
            y_true = [str(x) for x in y_test]
            acc = accuracy_score(y_true, decoded)
            rep = classification_report(y_true, decoded, output_dict=True, zero_division=0)
            report_summary[name] = {"accuracy": float(acc), "report": rep}
        except Exception as e:
            report_summary[name] = {"error": str(e)}
    results_dir = os.path.join(PROJECT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = int(time.time())
    out_path = os.path.join(results_dir, f"evaluation_{ts}.json")
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(report_summary, f, indent=2)
    return {"saved_to": out_path, "summary": report_summary}

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    loaded = []
    if rf is not None: loaded.append("RandomForest")
    if xgb is not None: loaded.append("XGBoost")
    if mlp is not None: loaded.append("MLP")
    return render_template("index.html", models=", ".join(loaded) or "none")

@app.route("/stream")
def stream():
    def generator():
        while True:
            Xs, ys = sample_rows(n=10)
            samples = predict_batch(Xs)
            for i, s in enumerate(samples):
                s["true"] = ys[i] if ys is not None else None
            payload = {"ts": int(time.time()), "samples": samples}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(3)
    return Response(generator(), mimetype="text/event-stream")

@app.route("/evaluate")
def evaluate_route():
    result = evaluate_models()
    return jsonify(result)

if __name__ == "__main__":
    print("Starting combined app 2. Models:",
          "RF" if rf else "-", "XGB" if xgb else "-", "MLP" if mlp else "-")
    app.run(debug=True, host="127.0.0.1", port=5000)
