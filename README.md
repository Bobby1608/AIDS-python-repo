# 🛡️ AI-Based Intrusion Detection System (IDS)

A machine learning-based Intrusion Detection System built on the CICIDS2017 dataset to detect and classify malicious network traffic.

---

## ✅ Objective

To detect and classify network intrusions such as DoS, DDoS, brute-force, reconnaissance, and botnet attacks using real-world NetFlow-based data and ML models.

---

## 🧠 Technologies Used

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, MLPClassifier, Matplotlib, Seaborn, imbalanced-learn
- **Tools:** VS Code, Jupyter, joblib
- **Dataset:** [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

---

## 📁 Folder Structure

MyNetworkIDSProject/
├── dataset/ # Raw CSV data from CICIDS2017
├── processed/ # Preprocessed .npy arrays for model training
├── models/ # Saved trained models (pkl files)
├── src/ # Source scripts
│ ├── load_data.py
│ ├── preprocess.py
│ ├── train_model.py
│ ├── evaluate.py
│ └── visualize.py
├── README.md # Project documentation

---

## 🧪 Phase 1: Data Preparation

- ✅ Loaded CICIDS2017 CSVs (NetFlow + Payload)
- ✅ Cleaned missing values, dropped duplicates
- ✅ Extracted relevant flow-based features
- ✅ Encoded multiclass labels (e.g., DoS, DDoS, Web, BruteForce, etc.)
- ✅ Normalized features with RobustScaler
- ✅ Saved final arrays as `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`

---

## 🤖 Phase 2: Model Training & Evaluation

### ✅ Models Trained

| Model             | Accuracy   | Notes                                              |
| ----------------- | ---------- | -------------------------------------------------- |
| **Random Forest** | **99.68%** | Best overall, high precision/recall                |
| XGBoost           | 99.68%     | Great with small class handling (after resampling) |
| MLP Classifier    | 99.58%     | Deep neural net, slightly slower training          |

- Trained on 2M+ samples from CICIDS2017.
- Saved as `rf_model.pkl`, `xgb_model.pkl`, and `mlp_model.pkl`.

### 🧪 Evaluation Metrics

- High precision/recall on common attacks (Normal, DoS, DDoS)
- Lower performance on rare attacks (Botnet, XSS, SQL Injection) — to be improved with SMOTE/class weighting

### 📊 Visualizations

Confusion matrices and ROC curves for all models are saved as:

random_forest_confusion_matrix.png
xgboost_confusion_matrix.png
mlp_classifier_confusion_matrix.png
roc_curve_comparison.png

---

## 🔍 Current Best Model

**✅ Random Forest** is currently chosen as the primary classifier due to its:

- Excellent accuracy
- Fast inference speed
- Robustness across classes

---

## 🚧 To Do (Future Work / Phase 3+)

- ✅ Add SMOTE resampling to improve rare-class detection
- ⏳ Implement live inference module (`predict.py`)
- ⏳ Add CLI or GUI for user interaction
- ⏳ Convert model to ONNX or serve via REST API
- ⏳ Deploy to cloud or edge device

---

## 📌 Credits

- Dataset: [CICIDS 2017 by Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- Model structure & pipeline: Self-built from scratch using Scikit-learn + XGBoost

---

> 💡 Project Status: ✅ Phase 1 + 2 Completed  
> Waiting for real-time prediction & advanced improvements.
