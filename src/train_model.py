import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ✅ Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("[WARNING] XGBoost not installed. Skipping XGBClassifier.")

# ✅ Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Load processed data
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"), allow_pickle=True)
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)

print(f"[INFO] Loaded data -> Train: {X_train.shape}, Test: {X_test.shape}")

# ✅ Encode labels (strings → integers)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Save label encoder for later inference
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print("[INFO] Label encoder saved.")

# --------------------------
# 🔹 Train Random Forest
# --------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"  # ✅ handle imbalance
)
rf.fit(X_train, y_train_enc)
rf_preds = rf.predict(X_test)
print(f"[RF] Accuracy: {accuracy_score(y_test_enc, rf_preds):.4f}")
print(classification_report(y_test_enc, rf_preds, target_names=le.classes_))
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

# --------------------------
# 🔹 Train XGBoost
# --------------------------
if xgb_available:
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist"  # ✅ fast training
    )
    xgb.fit(X_train, y_train_enc)
    xgb_preds = xgb.predict(X_test)
    print(f"[XGB] Accuracy: {accuracy_score(y_test_enc, xgb_preds):.4f}")
    print(classification_report(y_test_enc, xgb_preds, target_names=le.classes_))
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

# --------------------------
# 🔹 Train MLP Neural Network
# --------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=20,          # keep small for first test
    random_state=42,
    verbose=True
)
mlp.fit(X_train, y_train_enc)
mlp_preds = mlp.predict(X_test)
print(f"[MLP] Accuracy: {accuracy_score(y_test_enc, mlp_preds):.4f}")
print(classification_report(y_test_enc, mlp_preds, target_names=le.classes_))
joblib.dump(mlp, os.path.join(MODEL_DIR, "mlp_model.pkl"))

print(f"[INFO] All models trained & saved in {MODEL_DIR}")
