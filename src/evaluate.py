import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

# --------------------------
# Paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------
# Load Data
# --------------------------
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), allow_pickle=True)

# Load Label Encoder
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
class_names = label_encoder.classes_
y_test_enc = label_encoder.transform(y_test)

print(f"[INFO] Loaded test data: {X_test.shape}, Labels: {y_test.shape}")
print(f"[INFO] Classes: {list(class_names)}")

# --------------------------
# Utility functions
# --------------------------
def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix - {model_name}"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Print numbers inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()

def plot_roc(y_true, y_prob, classes, model_name):
    # Binarize labels for multiclass ROC
    y_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        plt.plot(fpr[i], tpr[i],
                 label=f"{class_name} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{model_name}.png"))
    plt.close()

# --------------------------
# Evaluate each model
# --------------------------
models = {
    "RandomForest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
    "MLP": "mlp_model.pkl"
}

for name, filename in models.items():
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        print(f"[WARNING] {filename} not found, skipping...")
        continue

    print(f"\n[INFO] Evaluating {name}...")
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        # Fallback: one-hot for models without probas
        y_prob = np.eye(len(class_names))[y_pred]

    # Report
    report = classification_report(y_test_enc, y_pred, target_names=class_names)
    print(report)

    # Save report to file
    with open(os.path.join(RESULTS_DIR, f"report_{name}.txt"), "w") as f:
        f.write(report)

    # Plots
    plot_confusion_matrix(y_test_enc, y_pred, class_names, name)
    plot_roc(y_test_enc, y_prob, class_names, name)

print(f"\n[INFO] Evaluation complete. Results saved to {RESULTS_DIR}")
