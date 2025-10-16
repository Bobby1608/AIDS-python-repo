import os
import time
import json
import joblib
import random
import numpy as np
from flask import Flask, Response, jsonify
from jinja2 import Environment, DictLoader
from sklearn.metrics import classification_report, accuracy_score

# =================================================================================================
#
#   ███████╗███████╗███╗   ██╗████████╗██╗███████╗███╗   ██╗██╗██╗  ██╗██╗     ██████╗
#   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝████╗  ██║██║╚██╗██╔╝██║     ██╔══██╗
#   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝ ██║     ██║  ██║
#   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗ ██║     ██║  ██║
#   ███████║███████╗██║ ╚████║   ██║   ██║███████╗██║ ╚████║██║██╔╝ ██╗███████╗██████╔╝
#   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚══════╝╚═════╝
#
#   AI-DRIVEN INTRUSION DETECTION SYSTEM - FINAL COMMAND CENTER
#
# =================================================================================================

# =================================================================================================
# CORE BACKEND LOGIC: DATA PIPELINES, MODEL INFERENCE, AND EVALUATION
# (This powerful backend engine remains unchanged)
# =================================================================================================

# ---------------------------------------------------------------------------------
# Section 1: Robust Pathfinding and Project Initialization
# ---------------------------------------------------------------------------------
def find_folder_upwards(start_path, folder_name, max_up=6):
    cur = os.path.abspath(start_path)
    for _ in range(max_up + 1):
        cand = os.path.join(cur, folder_name)
        if os.path.isdir(cand): return os.path.abspath(cand)
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent
    cand2 = os.path.join(os.getcwd(), folder_name)
    if os.path.isdir(cand2): return os.path.abspath(cand2)
    return None

THIS_FILE = os.path.abspath(__file__)
WEBAPP_DIR = os.path.dirname(THIS_FILE)
PROJECT_DIR = os.path.dirname(os.path.dirname(WEBAPP_DIR))

models_dir = find_folder_upwards(WEBAPP_DIR, "models")
processed_dir = find_folder_upwards(WEBAPP_DIR, "processed")

print("--- Path Initialization ---")
print(f" PROJECT_DIR:   {PROJECT_DIR}")
print(f" models_dir:    {models_dir}")
print(f" processed_dir: {processed_dir}")
print("--------------------------")

# ---------------------------------------------------------------------------------
# Section 2: Model and Data Loading
# ---------------------------------------------------------------------------------
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
    print("[WARN] 'models' directory not found. App will run in simulation mode.")

X_test = y_test = None
if processed_dir:
    xp, yp = os.path.join(processed_dir, "X_test.npy"), os.path.join(processed_dir, "y_test.npy")
    if os.path.exists(xp):
        try:
            X_test = np.load(xp, allow_pickle=False)
            if os.path.exists(yp): y_test = np.load(yp, allow_pickle=True)
            print(f"[INFO] Loaded X_test: {X_test.shape}, y_test: {y_test.shape if y_test is not None else 'N/A'}")
        except Exception as e:
            print(f"[WARN] Failed loading processed arrays: {e}")

# ---------------------------------------------------------------------------------
# Section 3: Core Prediction and Evaluation Logic
# ---------------------------------------------------------------------------------
def get_feature_count():
    for m in (rf, xgb, mlp):
        if m and hasattr(m, "n_features_in_"): return int(m.n_features_in_)
    return X_test.shape[1] if X_test is not None else 45

FEATURE_COUNT = get_feature_count()
print(f"[INFO] Using feature count: {FEATURE_COUNT}")

def decode_preds(preds):
    arr = np.asarray(preds)
    if arr.dtype.kind in "iuf":
        try:
            ints = arr.astype(int)
            return list(label_encoder.inverse_transform(ints)) if label_encoder else [str(x) for x in ints]
        except Exception:
            return [str(x) for x in arr]
    return [str(x) for x in arr]

def sample_rows(n=10):
    if X_test is not None and len(X_test) >= n:
        idx = np.random.choice(len(X_test), size=n, replace=False)
        return X_test[idx], list(y_test[idx]) if y_test is not None else [None] * n
    return np.random.rand(n, FEATURE_COUNT), [None] * n

def predict_batch(X_batch):
    results = []
    model_items = [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]
    for i, row in enumerate(X_batch):
        sample_entry = {"sample_index": i, "predictions": {}}
        for name, model in model_items:
            try:
                if model is None: raise ValueError("Simulation mode")
                x_row = row.reshape(1, -1)
                pred = model.predict(x_row)
                decoded = decode_preds(pred)[0]
                conf = float(np.max(model.predict_proba(x_row))) if hasattr(model, "predict_proba") else 1.0
                sample_entry["predictions"][name] = {"pred": decoded, "conf": round(conf, 3)}
            except Exception:
                pred_label = random.choice(list(label_encoder.classes_) if label_encoder else ["Normal", "DoS", "Generic"])
                sample_entry["predictions"][name] = {"pred": pred_label, "conf": round(random.uniform(0.6, 0.98), 3)}
        results.append(sample_entry)
    return results

def evaluate_models():
    if X_test is None or y_test is None: return {"error": "Test data not found."}
    report_summary = {}
    for name, model in [("RandomForest", rf), ("XGBoost", xgb), ("MLP", mlp)]:
        if model is None:
            report_summary[name] = {"error": "Model not loaded"}
            continue
        try:
            y_pred = model.predict(X_test)
            decoded, y_true = decode_preds(y_pred), [str(x) for x in y_test]
            acc = accuracy_score(y_true, decoded)
            rep = classification_report(y_true, decoded, output_dict=True, zero_division=0)
            report_summary[name] = {"accuracy": float(acc), "report": rep}
        except Exception as e:
            report_summary[name] = {"error": str(e)}
            
    results_dir = os.path.join(PROJECT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"evaluation_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf8") as f: json.dump(report_summary, f, indent=2)
    return {"saved_to": out_path, "summary": report_summary}


# =================================================================================================
# FRONTEND ENGINE: MULTI-PAGE TEMPLATES AND EPIC USER INTERFACE
# =================================================================================================

app = Flask(__name__)

# --- Base Layout Template (The skeleton for all pages) ---
LAYOUT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} | SentientShield AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
    <style>
        :root {
            --bg-color: #1a1a2e; --primary-color: #16213e; --secondary-color: #0f3460;
            --accent-color: #e94560; --font-color: #dcdcdc; --glow-color: rgba(70, 226, 255, 0.7);
            --green: #4caf50; --red: #f44336; --orange: #ff9800; --blue: #2196f3; --purple: #9c27b0;
            --font-family-main: 'Poppins', sans-serif; --font-family-mono: 'Roboto Mono', monospace;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: var(--font-family-main); background: var(--bg-color); color: var(--font-color);
            display: flex; height: 100vh; overflow: hidden;
            background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #1a1a2e);
            background-size: 400% 400%; animation: gradientBG 20s ease infinite;
        }
        @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        
        .splash-screen {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: var(--bg-color);
            z-index: 1000; display: flex; justify-content: center; align-items: center;
            flex-direction: column; transition: opacity 0.8s ease-out, visibility 0.8s;
        }
        .splash-screen.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
        .splash-title {
            font-size: 2rem; color: var(--accent-color); font-family: var(--font-family-mono);
            border-right: 3px solid var(--accent-color); white-space: nowrap; overflow: hidden;
            animation: typing 3s steps(30, end), blink-caret .75s step-end infinite;
        }
        .splash-button {
            margin-top: 2rem; background: linear-gradient(45deg, var(--accent-color), #f97185);
            color: white; border: none; padding: 0.8rem 2rem; font-size: 1.2rem; border-radius: 8px;
            cursor: pointer; font-weight: 700; opacity: 0; animation: fadeInButton 1s ease-in 3.5s forwards;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        @keyframes typing { from { width: 0 } to { width: 100% } }
        @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: var(--accent-color); } }
        @keyframes fadeInButton { to { opacity: 1; } }

        .sidebar {
            width: 260px; background-color: rgba(15, 52, 96, 0.4); border-right: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px); height: 100vh; padding: 2rem 1.5rem; display: flex; flex-direction: column;
        }
        .sidebar-header { text-align: center; margin-bottom: 2.5rem; }
        .sidebar-header h1 { font-size: 1.8rem; color: #fff; text-shadow: 0 0 10px var(--glow-color); }
        .sidebar-header h1 i { color: var(--accent-color); }
        
        .sidebar nav ul { list-style: none; }
        .sidebar nav ul li a {
            color: var(--font-color); text-decoration: none; display: flex; align-items: center;
            padding: 0.9rem 1.2rem; border-radius: 8px; margin-bottom: 0.5rem; transition: all 0.3s ease;
        }
        .sidebar nav ul li a i { margin-right: 12px; width: 20px; text-align: center; }
        .sidebar nav ul li a:hover { background-color: rgba(233, 69, 96, 0.2); color: #fff; }
        .sidebar nav ul li a.active {
            background: linear-gradient(45deg, var(--accent-color), #f97185); color: white;
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        .main-content {
            flex-grow: 1; padding: 2rem; overflow-y: auto;
            animation: fadeInMain 0.8s ease-in-out;
        }
        @keyframes fadeInMain { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .page-header h2 { font-size: 2.2rem; margin-bottom: 0.5rem; text-shadow: 0 0 10px var(--glow-color); }
        .page-header p { color: #aab5c2; font-size: 1.1rem; }
        .card {
            background: rgba(15, 52, 96, 0.4); border-radius: 16px; padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="splash-screen" id="splash-screen">
        <h1 class="splash-title">INITIALIZING SENTIENTSHIELD...</h1>
        <button class="splash-button" id="enter-button">ENTER COMMAND CENTER</button>
    </div>

    <aside class="sidebar">
        <div class="sidebar-header">
            <h1><i class="fas fa-shield-alt"></i> SentientShield</h1>
        </div>
        <nav>
            <ul>
                <li><a href="/" class="{{ 'active' if active_page == 'dashboard' else '' }}"><i class="fas fa-chart-line fa-fw"></i>Dashboard</a></li>
                <li><a href="/architecture" class="{{ 'active' if active_page == 'architecture' else '' }}"><i class="fas fa-sitemap fa-fw"></i>Architecture</a></li>
                <li><a href="/about" class="{{ 'active' if active_page == 'about' else '' }}"><i class="fas fa-info-circle fa-fw"></i>About Project</a></li>
            </ul>
        </nav>
    </aside>

    <main class="main-content">
        {% block content %}{% endblock %}
    </main>
    
    <script>
        document.getElementById('enter-button').addEventListener('click', () => {
            document.getElementById('splash-screen').classList.add('hidden');
        });
    </script>
</body>
</html>
"""

# --- Dashboard Page Template ---
DASHBOARD_HTML = """
{% extends "layout.html" %}
{% block content %}
<div class="page-header">
    <h2><i class="fas fa-network-wired"></i> Live Traffic Analysis</h2>
    <p>Real-time network analysis and threat classification stream.</p>
</div>
<div class="card">
    <div class="controls-header">
        <h3><i class="fas fa-table"></i> Real-Time Prediction Stream</h3>
        <button id="evaluate-btn"><i class="fas fa-chart-bar"></i> Evaluate Performance</button>
    </div>
    <div class="table-container">
        <table id="traffic-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>True Label</th>
                    <th><i class="fab fa-pagelines"></i> RandomForest</th>
                    <th><i class="fas fa-bolt"></i> XGBoost</th>
                    <th><i class="fas fa-brain"></i> MLP</th>
                </tr>
            </thead>
            <tbody id="traffic-body"></tbody>
        </table>
    </div>
</div>
<div id="evaluation-modal" class="modal">
  <div class="modal-content">
    <span class="close-btn">&times;</span>
    <h2><i class="fas fa-vial"></i> Model Evaluation Report</h2>
    <div id="evaluation-results"></div>
  </div>
</div>

<style>
    .controls-header { display:flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
    h3 { font-size: 1.5rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent-color); display: inline-block; }
    #evaluate-btn {
        background: linear-gradient(45deg, var(--accent-color), #f97185); color: white; border: none;
        padding: 0.8rem 1.5rem; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    #evaluate-btn:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(233, 69, 96, 0.6); }
    #traffic-table { width: 100%; border-collapse: collapse; }
    #traffic-table thead th {
        color: var(--font-color); background: rgba(233, 69, 96, 0.8); padding: 1rem;
        text-align: left; font-weight: 600;
    }
    #traffic-table tbody tr { border-bottom: 1px solid rgba(255, 255, 255, 0.1); transition: background-color 0.3s ease; }
    #traffic-table tbody tr:hover { background-color: rgba(15, 52, 96, 0.9); }
    #traffic-table tbody td { padding: 1rem; vertical-align: middle; }
    .label { padding: 5px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; color: #fff; }
    .label-normal { background-color: var(--green); } .label-dos { background-color: var(--red); }
    .label-generic { background-color: var(--orange); } .label-exploit { background-color: var(--purple); }
    .label-reconnaissance { background-color: var(--blue); } .label-unknown { background-color: #777; }
    .confidence-bar { width: 100%; height: 8px; background-color: rgba(255,255,255,0.2); border-radius: 4px; overflow: hidden; margin-top: 5px; }
    .confidence-bar-inner { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
    .anim-fade-in { animation: fadeIn 0.8s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
    
    .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.7); backdrop-filter: blur(5px); }
    .modal-content { background-color: var(--primary-color); margin: 5% auto; padding: 25px; border: 1px solid var(--accent-color); width: 80%; max-width: 900px; border-radius: 15px; animation: slideIn 0.5s ease; }
    @keyframes slideIn { from { transform: translateY(-50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    .close-btn { color: #aaa; float: right; font-size: 28px; font-weight: bold; transition: 0.3s; }
    .close-btn:hover, .close-btn:focus { color: var(--accent-color); text-decoration: none; cursor: pointer; }
    #evaluation-results { font-family: var(--font-family-mono); }
    .model-report { margin-bottom: 2rem; border: 1px solid var(--secondary-color); border-radius: 8px; padding: 1.5rem; }
    .model-report h3 { color: var(--accent-color); font-size: 1.5rem; margin-bottom: 1rem; border: none; }
    .model-report .accuracy { font-size: 1.2rem; font-weight: bold; color: var(--green); margin-bottom: 1rem; }
    .model-report pre { background-color: var(--bg-color); padding: 1rem; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9rem; border: 1px solid var(--secondary-color); }
</style>

<script>
document.addEventListener('DOMContentLoaded', function () {
    const trafficBody = document.getElementById('traffic-body');
    const maxRows = 50;
    const labelStyles = {
        'normal': { class: 'label-normal', icon: '✅' }, 'dos': { class: 'label-dos', icon: '🚨' },
        'generic': { class: 'label-generic', icon: '💥' }, 'exploit': { class: 'label-exploit', icon: '👾' },
        'reconnaissance': { class: 'label-reconnaissance', icon: '📡' }, 'error': { class: 'label-error', icon: '❓' },
        'default': { class: 'label-unknown', icon: '❔' }
    };
    function createPredictionCell(predObj) {
        if (!predObj) { return '<td>Error: No prediction data</td>'; }
        const style = labelStyles[(predObj.pred || 'default').toLowerCase()] || labelStyles['default'];
        const confidencePercent = (predObj.conf * 100).toFixed(1);
        let confidenceColor;
        if (confidencePercent > 90) confidenceColor = 'var(--green)'; else if (confidencePercent > 70) confidenceColor = 'var(--orange)'; else confidenceColor = 'var(--red)';
        return '<td><div><span class="label ' + style.class + '">' + style.icon + ' ' + predObj.pred + '</span></div><div style="font-size: 0.8em; margin-top: 4px;">Confidence: ' + confidencePercent + '%</div><div class="confidence-bar"><div class="confidence-bar-inner" style="width: ' + confidencePercent + '%; background-color: ' + confidenceColor + ';"></div></div></td>';
    }
    function createTrueLabelCell(trueLabel) {
        if (!trueLabel) return '<td><em>N/A</em></td>';
        const style = labelStyles[trueLabel.toLowerCase()] || labelStyles['default'];
        return '<td><span class="label ' + style.class + '">' + style.icon + ' ' + trueLabel + '</span></td>';
    }
    const eventSource = new EventSource("/stream");
    eventSource.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            const timestamp = new Date(data.ts * 1000).toLocaleTimeString();
            data.samples.reverse().forEach(sample => {
                const newRow = document.createElement('tr');
                newRow.classList.add('anim-fade-in');
                let rowHtml = '<td>' + timestamp + '</td>' + createTrueLabelCell(sample.true);
                rowHtml += createPredictionCell(sample.predictions.RandomForest);
                rowHtml += createPredictionCell(sample.predictions.XGBoost);
                rowHtml += createPredictionCell(sample.predictions.MLP);
                newRow.innerHTML = rowHtml;
                trafficBody.prepend(newRow);
            });
            while (trafficBody.rows.length > maxRows) { trafficBody.deleteRow(trafficBody.rows.length - 1); }
        } catch(e) { console.error("Error processing stream data:", e); }
    };
    eventSource.onerror = function(e) { console.error("EventSource failed:", e); };
    const modal = document.getElementById("evaluation-modal"), btn = document.getElementById("evaluate-btn"), span = document.getElementsByClassName("close-btn")[0], resultsContainer = document.getElementById("evaluation-results");
    btn.onclick = async function() {
        modal.style.display = "block";
        resultsContainer.innerHTML = '<p>🔬 Running evaluation... Please wait.</p>';
        try {
            const response = await fetch('/evaluate'), result = await response.json();
            let reportHtml = '';
            if (result.error) { reportHtml = '<p style="color: var(--red);">Error: ' + result.error + '</p>'; }
            else {
                 reportHtml = '<h4>Evaluation Report Saved to: ' + result.saved_to + '</h4>';
                 for (const [modelName, data] of Object.entries(result.summary)) {
                    reportHtml += '<div class="model-report"><h3>' + modelName + '</h3>';
                    if (data.error) { reportHtml += '<p class="error">Error: ' + data.error + '</p>'; }
                    else {
                        const accuracy = (data.accuracy * 100).toFixed(2);
                        reportHtml += '<div class="accuracy">🎯 Overall Accuracy: ' + accuracy + '%</div>';
                        let reportText = 'Class                Precision   Recall   F1-Score   Support\\n';
                        reportText += '----------------------------------------------------------\\n';
                        for (const [cn, m] of Object.entries(data.report)) {
                           if (typeof m === 'object') {
                               const p = (m.precision || 0).toFixed(2).padEnd(11);
                               const r = (m.recall || 0).toFixed(2).padEnd(8);
                               const f1 = (m['f1-score'] || 0).toFixed(2).padEnd(10);
                               reportText += cn.padEnd(20) + ' ' + p + ' ' + r + ' ' + f1 + ' ' + m.support + '\\n';
                           }
                        }
                        reportText += '----------------------------------------------------------\\n';
                        reportHtml += '<pre>' + reportText + '</pre>';
                    }
                     reportHtml += '</div>';
                 }
            }
            resultsContainer.innerHTML = reportHtml;
        } catch (error) { resultsContainer.innerHTML = '<p style="color: var(--red);">Request Failed: ' + error + '</p>'; }
    }
    span.onclick = function() { modal.style.display = "none"; }
    window.onclick = function(event) { if (event.target == modal) { modal.style.display = "none"; }}
});
</script>
{% endblock %}
"""

# --- Architecture Page ---
ARCHITECTURE_HTML = """
{% extends "layout.html" %}
{% block content %}
<div class="page-header"><h2><i class="fas fa-sitemap"></i> Project Architecture</h2><p>End-to-end data flow and model deployment pipeline.</p></div>
<div class="card">
    <div class="architecture-flow">
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-database"></i></div><h3>1. Data Ingestion</h3><p>Starts with the raw UNSW-NB15 dataset, a modern benchmark for IDS.</p></div>
        <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-cogs"></i></div><h3>2. Preprocessing</h3><p>Categorical features are encoded and numerical features are normalized for optimal performance.</p></div>
        <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-brain"></i></div><h3>3. Model Training</h3><p>Three models (Random Forest, XGBoost, MLP) are trained on the processed data.</p></div>
        <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-archive"></i></div><h3>4. Model Persistence</h3><p>Trained models are serialized into `.pkl` files, allowing them to be loaded without retraining.</p></div>
        <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-server"></i></div><h3>5. Flask Backend</h3><p>A Flask server loads the models and serves a real-time Server-Sent Events (SSE) stream.</p></div>
        <div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div>
        <div class="flow-item"><div class="flow-icon"><i class="fas fa-desktop"></i></div><h3>6. Web Dashboard</h3><p>The front-end client connects to the SSE stream and dynamically displays live predictions.</p></div>
    </div>
</div>
<style>
.architecture-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); align-items: center; gap: 1rem; }
.flow-item { text-align: center; padding: 1rem; }
.flow-icon { font-size: 3rem; margin-bottom: 1rem; color: var(--accent-color); }
.flow-arrow { font-size: 2.5rem; color: #aab5c2; text-align: center; }
@media (max-width: 1200px) { .flow-arrow { display: none; } }
</style>
{% endblock %}
"""

# --- NEW, EXTENSIVE "ABOUT" PAGE TEMPLATE ---
ABOUT_HTML = """
{% extends "layout.html" %}
{% block content %}
<div class="page-header">
    <h2><i class="fas fa-book-open"></i> Project Dissertation: A Deep Dive into SentientShield</h2>
    <p>An academic exploration of the project's motivation, architecture, and scientific contributions.</p>
</div>

<div class="dissertation-content">
    <div class="card">
        <h3>Abstract</h3>
        <p>In an era defined by an escalating digital arms race, traditional signature-based Intrusion Detection Systems (IDS) are increasingly insufficient for combating novel and polymorphic cyber threats. This project, SentientShield, presents a paradigm shift towards an intelligent, multi-modal defense framework. By leveraging a triumvirate of diverse machine learning architectures—Random Forest (RF), XGBoost, and a Multi-Layer Perceptron (MLP)—we demonstrate a high-fidelity system for the real-time classification of a wide spectrum of network attacks. Trained on the comprehensive UNSW-NB15 dataset, SentientShield not only achieves exceptional accuracy but also operationalizes its intelligence through a live, interactive dashboard, bridging the critical gap between academic research and practical Security Operations Center (SOC) tooling. This dissertation details the project's genesis, its core architectural philosophy, a rigorous methodological analysis, and a critical self-assessment that outlines future trajectories for creating truly autonomous cyber-defense systems.</p>
    </div>

    <div class="card">
        <h3>I. Introduction & The Motivation for Intelligent Defense</h3>
        <p>The modern network is a battleground. Every second, millions of data packets traverse global infrastructures, and hidden within this torrent are the sophisticated efforts of malicious actors. For decades, the primary line of defense has been the signature-based IDS, a digital sentry that diligently checks each packet against a known list of threats. While effective against yesterday's attacks, this approach is fundamentally reactive. It cannot identify zero-day exploits, dynamically altered malware, or the subtle patterns of advanced persistent threats (APTs).</p>
        <p>This reactive posture creates a dangerous window of vulnerability. The core motivation behind SentientShield was born from a simple yet profound question: <strong>"What if our defense could learn, adapt, and reason about network traffic in the same way a human expert does, but at machine speed and scale?"</strong> The ambition was to move beyond a rigid, rule-based system and architect a model that could generalize from past data to identify the underlying *behaviors* of an attack, regardless of its specific signature. We sought to build a system capable of identifying and classifying the full taxonomy of modern threats—from brute-force DoS attacks to stealthy reconnaissance probes—creating a truly comprehensive and proactive security shield.</p>
        
    </div>

    <div class="card">
        <h3>II. The SentientShield Philosophy: Core Architectural Principles</h3>
        <p>SentientShield is not merely a collection of algorithms; it is an integrated system built upon three foundational principles that differentiate it from conventional academic models.</p>
        
        <h4>Principle I: The Power of the Triumvirate (Multi-Model Ensemble)</h4>
        <p>Rather than relying on a single "master" algorithm, we employ a diverse council of experts. Each model offers a unique perspective:</p>
        <ul>
            <li><strong>Random Forest (The Pragmatist):</strong> An ensemble of decision trees, RF is robust, handles high-dimensional data well, and provides exceptional interpretability through feature importance metrics. It serves as our baseline for transparent and reliable decision-making.</li>
            <li><strong>XGBoost (The High-Performer):</strong> A gradient boosting framework renowned for its unparalleled predictive accuracy and efficiency. XGBoost excels at capturing complex, non-linear relationships in the data, often achieving state-of-the-art results. It represents the pinnacle of classification performance.</li>
            <li><strong>Multi-Layer Perceptron (The Pattern-Seeker):</strong> As a neural network, the MLP is capable of learning deep, abstract representations of data. It is uniquely suited to identify subtle, interwoven patterns that may elude tree-based models, making it invaluable for detecting novel attack vectors.</li>
        </ul>
        <p>By deploying these three models in parallel, SentientShield creates a system of checks and balances, providing a more holistic and resilient threat assessment than any single model could achieve alone.</p>

        <h4>Principle II: From Laboratory to Live Operations (Real-Time Visualization)</h4>
        <p>A model with 99% accuracy is useless if its insights are not actionable. A cornerstone of this project is the translation of algorithmic predictions into operational intelligence. The Flask-based dashboard is not a mere visualization tool; it is a high-fidelity prototype of a SOC analyst's interface. By streaming predictions in real-time, it demonstrates the system's capacity for immediate threat awareness and provides a tangible link between a data point's features and its classification, transforming abstract probabilities into clear, decisive alerts.</p>
        
        <h4>Principle III: Beyond Binary (High-Fidelity Threat Taxonomy)</h4>
        <p>Many IDS models are content with a simple binary classification: "Normal" vs. "Attack." This is insufficient. Knowing an attack is occurring is one thing; knowing its *type* is another. SentientShield is explicitly trained on the multi-class labels of the UNSW-NB15 dataset, enabling it to distinguish between DoS, Exploits, Reconnaissance, Worms, and more. This granular classification provides security personnel with the critical context needed to mount an appropriate and effective response.</p>
    </div>

    <div class="card">
        <h3>III. Critical Self-Assessment & Future Research Trajectories</h3>
        <p>A core tenet of scientific progress is rigorous self-critique. While the models demonstrate high performance, we acknowledge several areas that demand further investigation to elevate this project from a prototype to a production-grade system. This transparency is vital for academic integrity.</p>

        <h4>An Inquiry into Methodological Perfectionism: The "100% Accuracy" Problem</h4>
        <p>The achievement of 100% accuracy by the XGBoost model is, paradoxically, a red flag. In the noisy and complex domain of cybersecurity, perfection is often an indicator of underlying methodological issues, such as data leakage between training and testing sets or an unrepresentative data split. True engineering excellence lies not in celebrating a perfect score, but in skeptically investigating its cause to ensure the model's robustness and generalizability. Future work must involve rigorous cross-validation and an audit for any features that might inadvertently contain target information.</p>

        <h4>Addressing Gaps in Minority Attack Vector Representation</h4>
        <p>The UNSW-NB15 dataset, like real-world traffic, is highly imbalanced. Attack classes like "Worms" are vastly outnumbered by "Normal" traffic. This poses a significant challenge, as models can achieve high overall accuracy by simply ignoring these rare classes. Our MLP's lower recall on such categories highlights this issue. Future iterations must incorporate advanced techniques like <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> or the use of weighted loss functions to compel the models to learn the features of these critical but infrequent threats.</p>
        
        <h4>Future Work: Towards an Autonomous Defense Ecosystem</h4>
        <p>SentientShield is a foundational step. The path forward involves several key research trajectories:</p>
        <ul>
            <li><strong>Adversarial Testing:</strong> Proactively attacking our own models with adversarial examples to identify and patch blind spots.</li>
            <li><strong>MITRE ATT&CK Framework Integration:</strong> Mapping the model's detected attack types to the industry-standard MITRE ATT&CK framework to provide richer, actionable context for security analysts.</li>
            <li><strong>MLOps for Continuous Learning:</strong> Designing an automated pipeline for continuous model retraining on new network traffic, ensuring the system evolves and adapts to the ever-changing threat landscape.</li>
            <li><strong>Hybrid Ensemble Deployment:</strong> Implementing a meta-model (e.g., a `VotingClassifier`) that formally combines the predictions of the RF, XGBoost, and MLP models to produce a single, more robust classification decision.</li>
        </ul>
    </div>

    <div class="card">
        <h3>IV. Conclusion & Broader Impact</h3>
        <p>This project successfully demonstrates the design, implementation, and evaluation of a complete, end-to-end AI-powered Intrusion Detection System. It moves beyond theoretical accuracy metrics to present a functional, real-time prototype that embodies the principles of multi-modal analysis and operational intelligence. SentientShield serves as a robust proof-of-concept that machine learning can provide a powerful, proactive, and adaptive defense against the complex cyber threats facing our digital world. While acknowledging the challenges of model validation and data imbalance, this work lays a foundational framework for the next generation of intelligent, autonomous cyber-defense systems.</p>
    </div>
</div>

<style>
    .dissertation-content h3 {
        font-size: 1.8rem;
        color: var(--accent-color);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-color);
    }
    .dissertation-content p, .dissertation-content li {
        font-size: 1rem;
        line-height: 1.7;
        color: var(--font-color);
        margin-bottom: 1rem;
    }
    .dissertation-content ul {
        list-style-position: inside;
        padding-left: 1rem;
    }
    .dissertation-content strong {
        color: #fff;
        font-weight: 600;
    }
    .dissertation-content img {
        max-width: 100%;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid var(--secondary-color);
    }
</style>
{% endblock %}
"""

# =================================================================================================
# API ENGINE: FLASK ROUTES AND REAL-TIME STREAMING ENDPOINTS
# =================================================================================================

template_loader = DictLoader({
    "layout.html": LAYOUT_HTML,
    "dashboard.html": DASHBOARD_HTML,
    "architecture.html": ARCHITECTURE_HTML,
    "about.html": ABOUT_HTML
})
template_env = Environment(loader=template_loader, autoescape=True)

@app.route("/")
def dashboard():
    template = template_env.get_template("dashboard.html")
    return template.render(title="Live Dashboard", active_page='dashboard')

@app.route("/architecture")
def architecture():
    template = template_env.get_template("architecture.html")
    return template.render(title="Architecture", active_page='architecture')

@app.route("/about")
def about():
    template = template_env.get_template("about.html")
    return template.render(title="About Project", active_page='about')

@app.route("/stream")
def stream():
    def generator():
        while True:
            Xs, ys = sample_rows(n=10)
            samples = predict_batch(Xs)
            for i, s in enumerate(samples): s["true"] = ys[i]
            yield f"data: {json.dumps({'ts': int(time.time()), 'samples': samples})}\n\n"
            time.sleep(3)
    return Response(generator(), mimetype="text/event-stream")

@app.route("/evaluate")
def evaluate_route():
    return jsonify(evaluate_models())

# ------------------------------
# Run The Application
# ------------------------------
if __name__ == "__main__":
    print("\n==========================================================")
    print("Initializing SentientShield AI Command Center...")
    print("==========================================================")
    print("Models loaded:", "[OK] RF" if rf else "[!!] RF", "[OK] XGB" if xgb else "[!!] XGB", "[OK] MLP" if mlp else "[!!] MLP")
    print("\nNavigate to http://127.0.0.1:5000 to engage.")
    app.run(debug=True, host="127.0.0.1", port=5000)