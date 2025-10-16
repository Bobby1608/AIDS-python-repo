import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# ========================
# Setup paths
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ========================
# Load test data
# ========================
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"), allow_pickle=True)

# ========================
# Clean non-ASCII labels
# ========================
def clean_labels(y):
    return np.array([str(label).encode('ascii', 'ignore').decode('ascii') for label in y])

y_test = clean_labels(y_test)

# ========================
# Encode labels for ROC
# ========================
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)
class_names = le.classes_
y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))

# ========================
# Load models
# ========================
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
mlp_model = joblib.load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

# ========================
# Plot Confusion Matrix
# ========================
def plot_conf_matrix(model, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved: {save_path}")
    plt.show()

# Plot confusion matrices for all models
plot_conf_matrix(rf_model, "Random Forest")
plot_conf_matrix(xgb_model, "XGBoost")
plot_conf_matrix(mlp_model, "MLP Classifier")

# ========================
# Plot ROC Curve
# ========================
def plot_roc_curve(model, model_name, color):
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        print(f"[WARNING] {model_name} does not support predict_proba(). Skipping.")
        return

    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})", linewidth=2, color=color)

# ========================
# Create ROC Curve Plot
# ========================
plt.figure(figsize=(8, 6))
plot_roc_curve(rf_model, "Random Forest", 'blue')
plot_roc_curve(xgb_model, "XGBoost", 'green')
plot_roc_curve(mlp_model, "MLP Classifier", 'red')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

roc_save_path = os.path.join(BASE_DIR, "roc_curve_comparison.png")
plt.savefig(roc_save_path)
print(f"[INFO] Saved: {roc_save_path}")
plt.show()
