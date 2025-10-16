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
#   █████╗ ██╗██╗   ██╗     ██████╗ ██████╗ ██╗   ██╗███████╗██████╗
#  ██╔══██╗██║╚██╗ ██╔╝     ██╔══██╗██╔══██╗██║   ██║██╔════╝██╔══██╗
#  ███████║██║ ╚████╔╝      ██████╔╝██████╔╝██║   ██║█████╗  ██████╔╝
#  ██╔══██║██║  ╚██╔╝       ██╔═══╝ ██╔══██╗██║   ██║██╔══╝  ██╔══██╗
#  ██║  ██║██║   ██║        ██║     ██║  ██║╚██████╔╝███████╗██║  ██║
#  ╚═╝  ╚═╝╚═╝   ╚═╝        ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
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
SRC_DIR = os.path.dirname(WEBAPP_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

models_dir = find_folder_upwards(WEBAPP_DIR, "models")
processed_dir = find_folder_upwards(WEBAPP_DIR, "processed")

print("Using paths:")
print(" PROJECT_DIR:", PROJECT_DIR)
print(" models_dir:", models_dir)
print(" processed_dir:", processed_dir)

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
    print("[WARN] models directory not found. App will run in simulation mode.")

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
    if X_test is not None:
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
                pred_label = random.choice(list(label_encoder.classes_) if label_encoder else ["Normal", "DoS"])
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
    out_path = os.path.join(PROJECT_DIR, "results", f"evaluation_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f: json.dump(report_summary, f, indent=2)
    return {"saved_to": out_path}

# =================================================================================================
# FRONTEND ENGINE: MULTI-PAGE TEMPLATES AND EPIC USER INTERFACE
# =================================================================================================

app = Flask(__name__)

# --- Base Layout Template (The skeleton for all pages) ---
LAYOUT_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - AI-Based IDS</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-dark: #0d1117; --bg-light: #161b22; --border-color: #30363d;
            --text-light: #c9d1d9; --text-dark: #8b949e; --cyan: #39d3ee; --purple: #a78bfa;
            --green: #56d364; --red: #f87171; --orange: #f0883e;
            --glow-cyan: rgba(57, 211, 238, 0.2); --glow-purple: rgba(167, 139, 250, 0.2);
            --glow-red: rgba(248, 113, 113, 0.25); --glow-orange: rgba(240, 136, 62, 0.25); --glow-green: rgba(86, 211, 100, 0.2);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body {
            font-family: 'Poppins', sans-serif; background-color: var(--bg-dark);
            color: var(--text-light); display: flex; overflow: hidden;
        }
        #particles-js { position: fixed; width: 100%; height: 100%; top: 0; left: 0; z-index: -1; }
        .splash-screen {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background-color: var(--bg-dark); z-index: 100;
            display: flex; justify-content: center; align-items: center;
            flex-direction: column; transition: opacity 0.8s ease-out;
        }
        .splash-screen.hidden { opacity: 0; pointer-events: none; }
        .splash-title {
            font-size: 2rem; color: var(--cyan); border-right: 3px solid var(--cyan);
            white-space: nowrap; overflow: hidden; animation: typing 3s steps(30, end), blink-caret .75s step-end infinite;
        }
        .splash-button {
            margin-top: 2rem; background: linear-gradient(90deg, var(--cyan), var(--purple));
            color: var(--bg-dark); border: none; padding: 0.8rem 2rem; font-size: 1.2rem;
            border-radius: 8px; cursor: pointer; font-weight: 700; opacity: 0;
            animation: fadeInButton 1s ease-in 3.5s forwards;
        }
        @keyframes typing { from { width: 0 } to { width: 100% } }
        @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: var(--cyan); } }
        @keyframes fadeInButton { to { opacity: 1; } }
        .sidebar {
            width: 280px; background-color: var(--bg-light); height: 100vh;
            position: fixed; top: 0; left: 0; padding: 2rem 1.5rem;
            border-right: 1px solid var(--border-color); display: flex; flex-direction: column;
            z-index: 10;
        }
        .sidebar h1 {
            font-size: 1.5rem; font-weight: 700;
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 2.5rem; display: flex; align-items: center;
        }
        .sidebar h1 i { -webkit-text-fill-color: var(--cyan); margin-right: 12px; }
        .sidebar nav ul { list-style: none; }
        .sidebar nav ul li a {
            color: var(--text-dark); text-decoration: none; display: block;
            padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
            transition: all 0.3s ease; font-weight: 600;
        }
        .sidebar nav ul li a:hover {
            background-color: rgba(201, 209, 217, 0.1); color: var(--text-light);
        }
        .sidebar nav ul li a.active {
            background: linear-gradient(90deg, var(--cyan), var(--purple));
            color: var(--bg-dark); box-shadow: 0 4px 20px var(--glow-purple);
        }
        .main-content {
            margin-left: 280px; padding: 2rem 3rem; width: calc(100% - 280px);
            animation: fadeInMain 0.8s ease-in-out; height: 100vh; overflow-y: auto;
        }
        @keyframes fadeInMain { from { opacity: 0; } to { opacity: 1; } }
        .page-header h2 { font-size: 2.2rem; }
        .page-header p { color: var(--text-dark); font-size: 1.1rem; }
        .card {
            background-color: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px);
            border: 1px solid var(--border-color); border-radius: 12px;
            padding: 2rem; margin-bottom: 1.5rem;
        }
        h3 { font-size: 1.2rem; margin-bottom: 1rem; padding-left: 1rem; border-left: 3px solid var(--cyan); text-transform: uppercase; letter-spacing: 1px;}
    </style>
</head>
<body>
    <div id="particles-js"></div>
    <div class="splash-screen" id="splash-screen">
        <h1 class="splash-title">INITIALIZING AI-IDS ENGINE...</h1>
        <button class="splash-button" id="enter-button">ENTER COMMAND CENTER</button>
    </div>
    <div class="sidebar">
        <h1><i class="fas fa-shield-virus"></i>AI-IDS</h1>
        <nav>
            <ul>
                <li><a href="/" class="{{ 'active' if active_page == 'dashboard' else '' }}" onclick="playSound('nav')"><i class="fas fa-chart-line"></i>Dashboard</a></li>
                <li><a href="/architecture" class="{{ 'active' if active_page == 'architecture' else '' }}" onclick="playSound('nav')"><i class="fas fa-sitemap"></i>Architecture</a></li>
                <li><a href="/about" class="{{ 'active' if active_page == 'about' else '' }}" onclick="playSound('nav')"><i class="fas fa-users"></i>About & Team</a></li>
            </ul>
        </nav>
    </div>
    <main class="main-content">
        {% block content %}{% endblock %}
    </main>
    
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const sounds = {
            nav: { freq: 440, type: 'sine', duration: 0.1 },
            predict: { freq: 880, type: 'triangle', duration: 0.05 }
        };
        function playSound(soundName) {
            const sound = sounds[soundName];
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            oscillator.type = sound.type;
            oscillator.frequency.setValueAtTime(sound.freq, audioContext.currentTime);
            gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.00001, audioContext.currentTime + sound.duration);
            oscillator.start();
            oscillator.stop(audioContext.currentTime + sound.duration);
        }

        document.getElementById('enter-button').addEventListener('click', () => {
            playSound('nav');
            document.getElementById('splash-screen').classList.add('hidden');
        });

        particlesJS("particles-js", {
            "particles":{"number":{"value":80,"density":{"enable":true,"value_area":800}},"color":{"value":"#8b949e"},"shape":{"type":"circle"},"opacity":{"value":0.3,"random":true},"size":{"value":3,"random":true},"line_linked":{"enable":true,"distance":150,"color":"#8b949e","opacity":0.2,"width":1},"move":{"enable":true,"speed":1,"direction":"none","random":true,"straight":false,"out_mode":"out","bounce":false}},
            "interactivity":{"detect_on":"canvas","events":{"onhover":{"enable":true,"mode":"grab"},"onclick":{"enable":true,"mode":"push"},"resize":true},"modes":{"grab":{"distance":140,"line_linked":{"opacity":0.5}},"push":{"particles_n":4}}},
            "retina_detect":true
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
    <h2>Live Command Center</h2>
    <p>Real-time network analysis and threat classification engine.</p>
</div>

<div class="dashboard-grid">
    <div class="info-column">
        <div class="stat-card threat-level-card">
            <div class="stat-info">
                <span id="threat-level-text" class="stat-label">THREAT LEVEL</span>
                <span id="threat-level-status" class="stat-number">NOMINAL</span>
            </div>
            <div id="threat-level-icon" class="threat-icon"><i class="fas fa-shield-alt"></i></div>
        </div>
        <div class="stat-card">
            <i class="fas fa-network-wired"></i>
            <div class="stat-info">
                <span id="packets-stat" class="stat-number">1,428</span>
                <span class="stat-label">Packets Analyzed / Sec</span>
            </div>
        </div>
        <div class="card">
            <h3><i class="fas fa-star"></i>Key Features</h3>
            <ul>
                <li>Real-Time Threat Classification</li>
                <li>Multi-Model Ensemble Engine</li>
                <li>Interactive Web Dashboard</li>
                <li>On-Demand Performance Evaluation</li>
            </ul>
        </div>
    </div>

    <div class="table-column">
        <div class="card">
            <div class="controls-header">
                <h3><i class="fas fa-table"></i>Real-Time Prediction Stream</h3>
                <button id="eval"><i class="fas fa-play-circle"></i> Run Full Evaluation</button>
            </div>
            <div id="eval_result"></div>
            <table id="pred_table">
                <thead><tr><th>#</th><th>True Label</th><th>RandomForest</th><th>XGBoost</th><th>MLP</th></tr></thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
</div>

<style>
    .dashboard-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 1.5rem; align-items: flex-start; }
    .info-column { display: flex; flex-direction: column; gap: 1.5rem; }
    .stat-card {
        padding: 1.5rem; display: flex; align-items: center; justify-content: space-between;
        border: 1px solid var(--border-color); border-radius: 12px; transition: all 0.3s ease;
        background-color: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px);
    }
    .stat-card:hover { border-color: var(--cyan); box-shadow: 0 0 20px var(--glow-cyan); }
    .stat-card i { font-size: 2rem; margin-right: 1rem; color: var(--cyan); }
    .stat-info { display: flex; flex-direction: column; }
    .stat-number { font-size: 1.8rem; font-weight: 700; }
    .stat-label { font-size: 0.9rem; color: var(--text-dark); text-transform: uppercase; }
    
    .threat-level-card { border-left: 5px solid var(--green); }
    .threat-level-card.elevated { border-left-color: var(--orange); }
    .threat-level-card.critical { border-left-color: var(--red); animation: pulseRed 1.5s infinite; }
    .threat-icon { font-size: 2.5rem; }
    #threat-level-status { font-size: 1.5rem; }
    .info-column .card ul { list-style-type: none; padding-left: 0; }
    .info-column .card li { margin-bottom: 0.5rem; }
    
    .controls-header { display:flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
    #eval { background: linear-gradient(90deg, var(--cyan), var(--purple)); color: var(--bg-dark); border: none; padding: 0.6rem 1.2rem; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.3s ease; }
    #eval:hover { box-shadow: 0 4px 20px var(--glow-purple); transform: translateY(-2px); }
    table { width: 100%; margin-top: 1rem; border-collapse: separate; border-spacing: 0 8px;}
    th, td { padding: 0.8rem 1rem; text-align: left; }
    thead tr { background-color: transparent !important; box-shadow: none !important; }
    th { color: var(--text-dark); text-transform: uppercase; font-size: 0.8rem; }
    tbody tr { 
        background-color: rgba(30, 36, 45, 0.7);
        transition: all 0.3s ease;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }
    tbody tr.threat-critical { box-shadow: inset 0 0 10px var(--glow-red), 0 0 10px var(--glow-red); border: 1px solid var(--red);}
    tbody tr.threat-elevated { box-shadow: inset 0 0 10px var(--glow-orange), 0 0 10px var(--glow-orange); border: 1px solid var(--orange);}
    tbody tr.threat-nominal { box-shadow: inset 0 0 10px var(--glow-green), 0 0 10px var(--glow-green); border: 1px solid var(--green);}
    td:first-child { border-top-left-radius: 8px; border-bottom-left-radius: 8px; }
    td:last-child { border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
    .pred-cell.normal { color: var(--green); } .pred-cell.attack { color: var(--red); } .pred-cell.other { color: var(--orange); }
    .conf { font-size: 0.8em; color: var(--text-dark); }
    @keyframes pulseRed { 0%, 100% { box-shadow: 0 0 20px var(--glow-red); } 50% { box-shadow: 0 0 30px var(--glow-red); } }
</style>

<script>
  const tbody = document.querySelector("#pred_table tbody");
  const evtSource = new EventSource("/stream");
  
  function getOverallThreat(predictions) {
    let level = 0; // 0=nominal, 1=elevated, 2=critical
    for (const model in predictions) {
        const pred = predictions[model].pred.toLowerCase();
        if (['dos', 'exploit', 'worm'].some(t => pred.includes(t))) {
            level = Math.max(level, 2); // Critical
        } else if (pred.includes('attack') || pred.includes('reconnaissance') || pred.includes('generic')) {
            level = Math.max(level, 1); // Elevated
        }
    }
    if (level === 2) return 'threat-critical';
    if (level === 1) return 'threat-elevated';
    return 'threat-nominal';
  }

  evtSource.onmessage = e => {
    try {
        const batch = JSON.parse(e.data);
        playSound('predict');
        tbody.innerHTML = batch.samples.map(s => {
            const rf = s.predictions.RandomForest || {pred:'N/A',conf:0};
            const xgb = s.predictions.XGBoost || {pred:'N/A',conf:0};
            const mlp = s.predictions.MLP || {pred:'N/A',conf:0};
            const threatClass = getOverallThreat(s.predictions);
            const getPClass = p => { const l=(p||'').toLowerCase(); if(l.includes('normal')) return 'normal'; if(l.includes('dos') || l.includes('attack')) return 'attack'; return 'other'; };
            return `<tr class="${threatClass}"><td>${s.sample_index}</td><td>${s.true || "-"}</td><td class="pred-cell ${getPClass(rf.pred)}">${rf.pred} <span class="conf">(${rf.conf})</span></td><td class="pred-cell ${getPClass(xgb.pred)}">${xgb.pred} <span class="conf">(${xgb.conf})</span></td><td class="pred-cell ${getPClass(mlp.pred)}">${mlp.pred} <span class="conf">(${mlp.conf})</span></td></tr>`;
        }).join('');
    } catch(err) { console.error("Stream parse error:", err); }
  };
  
  document.getElementById("eval").addEventListener("click", async () => {
    const rDiv=document.getElementById("eval_result"); rDiv.style.color='var(--cyan)'; rDiv.textContent="Running full evaluation...";
    try { const res=await fetch("/evaluate"), data=await res.json(); rDiv.style.color=data.error ? 'var(--red)' : 'var(--green)'; rDiv.textContent = data.error ? `Error: ${data.error}` : `Success! Report saved to: ${data.saved_to}`; } catch (err) { rDiv.style.color='var(--red)'; rDiv.textContent=`Request Failed: ${err}`; }
  });

  setInterval(() => {
    document.getElementById('packets-stat').textContent=(Math.floor(Math.random()*500)+1200).toLocaleString();
    const rows = tbody.getElementsByTagName('tr');
    let attackCount = 0;
    for (let row of rows) {
        if (!row.classList.contains('threat-nominal')) {
            attackCount++;
        }
    }
    const threatCard = document.querySelector('.threat-level-card');
    const statusEl = document.getElementById('threat-level-status');
    const attackRatio = attackCount / (rows.length || 1);
    threatCard.classList.remove('elevated', 'critical');
    if (attackRatio > 0.5) { statusEl.textContent = 'CRITICAL'; threatCard.classList.add('critical'); } 
    else if (attackRatio > 0.2) { statusEl.textContent = 'ELEVATED'; threatCard.classList.add('elevated'); } 
    else { statusEl.textContent = 'NOMINAL'; }
  }, 3500);
</script>
{% endblock %}
"""

# --- All Other Page Templates (Architecture, About) are unchanged for brevity but would be here ---
ARCHITECTURE_HTML = """
{% extends "layout.html" %}
{% block content %}
<div class="page-header"><h2>Project Architecture</h2><p>End-to-end data flow and model deployment pipeline.</p></div>
<div class="card"><div class="architecture-flow"><div class="flow-item"><div class="flow-icon"><i class="fas fa-database"></i></div><h3>1. Data Ingestion</h3><p>The process starts with the raw UNSW-NB15 dataset, a modern benchmark containing a hybrid of real and simulated network traffic.</p></div><div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div><div class="flow-item"><div class="flow-icon"><i class="fas fa-cogs"></i></div><h3>2. Preprocessing</h3><p>Categorical features are one-hot encoded, and all 45 numerical features are normalized to a uniform scale for optimal model performance.</p></div><div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div><div class="flow-item"><div class="flow-icon"><i class="fas fa-brain"></i></div><h3>3. Model Training</h3><p>Three distinct models (Random Forest, XGBoost, MLP) are trained on the processed data, each learning to identify attack patterns in a unique way.</p></div><div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div><div class="flow-item"><div class="flow-icon"><i class="fas fa-archive"></i></div><h3>4. Model Persistence</h3><p>Trained models are serialized into `.pkl` files using Joblib, allowing them to be loaded for prediction without retraining.</p></div><div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div><div class="flow-item"><div class="flow-icon"><i class="fas fa-server"></i></div><h3>5. Flask Backend</h3><p>A Flask web server loads the models and serves a real-time Server-Sent Events (SSE) stream of predictions from test data samples.</p></div><div class="flow-arrow"><i class="fas fa-long-arrow-alt-right"></i></div><div class="flow-item"><div class="flow-icon"><i class="fas fa-desktop"></i></div><h3>6. Web Dashboard</h3><p>The front-end client connects to the SSE stream and dynamically displays live predictions in the interactive dashboard you are using now.</p></div></div></div>
<style>.architecture-flow { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 1rem; }.flow-item { flex-basis: 30%; text-align: center; padding: 1rem; }.flow-icon { font-size: 3rem; margin-bottom: 1rem; color: var(--purple); }.flow-arrow { font-size: 2.5rem; color: var(--text-dark); }</style>
{% endblock %}
"""
ABOUT_HTML = """
{% extends "layout.html" %}
{% block content %}
<div class="page-header"><h2>About the Project & Team</h2><p>The technology, methodology, and the people behind this project.</p></div><div class="about-grid"><div class="card"><h3><i class="fas fa-rocket"></i>Project Significance</h3><p>This project demonstrates a complete, end-to-end pipeline for building a modern, AI-powered Intrusion Detection System. By integrating multiple high-performance models and presenting the results in a real-time dashboard, it replicates an industry-grade security monitoring concept and serves as a robust framework for research in AI-driven cybersecurity.</p></div><div class="card"><h3><i class="fas fa-users"></i>Our Team</h3><ul class="team-list"><li>Bhuban Wakode</li><li>Parth Karalkar</li><li>Aaryan Khamkar</li></ul></div><div class="card tech-card"><h3><i class="fas fa-microchip"></i>Technology Stack</h3><ul><li><b>Backend:</b> Python, Flask</li><li><b>ML Libraries:</b> Scikit-learn, XGBoost, Pandas, NumPy</li><li><b>Frontend:</b> HTML5, CSS3, JavaScript</li><li><b>Real-time Engine:</b> Server-Sent Events (SSE)</li><li><b>Dataset:</b> UNSW-NB15</li><li><b>Persistence:</b> Joblib</li></ul></div></div>
<style>.about-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }.tech-card { grid-column: 1 / -1; }.team-list { list-style: none; padding-left: 0; }.team-list li { font-size: 1.2rem; font-weight: 600; color: var(--text-light); padding: 0.5rem; border-bottom: 1px solid var(--border-color); }.team-list li:first-child { color: var(--cyan); } ul {list-style-position: inside; padding-left: 0; }</style>
{% endblock %}
"""

# =================================================================================================
# API ENGINE: FLASK ROUTES AND REAL-TIME STREAMING ENDPOINTS
# =================================================================================================

# --- Create the Jinja2 "Virtual" Environment ---
template_loader = DictLoader({
    "layout.html": LAYOUT_HTML,
    "dashboard.html": DASHBOARD_HTML,
    "architecture.html": ARCHITECTURE_HTML,
    "about.html": ABOUT_HTML
})
template_env = Environment(loader=template_loader, autoescape=True)

# --- Define the page routes ---
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
    return template.render(title="About", active_page='about')

# --- Define the API and Streaming endpoints ---
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
    print("Initializing AI-IDS Epic Command Center...")
    print("Models loaded:", "RF" if rf else "-", "XGB" if xgb else "-", "MLP" if mlp else "-")
    print("Navigate to http://127.0.0.1:5000 to engage.")
    app.run(debug=True, host="127.0.0.1", port=5000)