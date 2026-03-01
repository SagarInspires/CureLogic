"""
╔══════════════════════════════════════════════════════════════════╗
║   CureLogic — Flask Backend v2.0                                ║
║   L&T CreaTech Hackathon | Problem Statement 01                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Self-contained: trains SVM models on first launch if no .pkl   ║
║                                                                  ║
║  Routes:                                                         ║
║   GET  /                        → Serve Dashboard HTML           ║
║   POST /api/sensor_data         → Receive IoT push from NodeMCU  ║
║   GET  /api/predict             → Current live prediction        ║
║   POST /api/predict             → Custom payload prediction      ║
║   GET  /api/scenarios           → Ranked scenario table          ║
║   GET  /api/live                → SSE stream for dashboard       ║
║   GET  /api/status              → System health check            ║
║   POST /api/simulator/start     → Start IoT simulator            ║
║   POST /api/simulator/stop      → Stop IoT simulator             ║
╚══════════════════════════════════════════════════════════════════╝

Requirements:
    pip install flask numpy pandas scikit-learn joblib

Run:
    python app.py

Curl test:
    curl -X POST http://localhost:5000/api/sensor_data \
      -H "Content-Type: application/json" \
      -H "X-Device-Key: CL-SECRET-2024" \
      -d '{"device_id":"CL-NODE-01","elapsed_hours":8.5,
           "ambient_temp":35.0,"temp_surface":38.2,
           "temp_mid":46.5,"temp_core":58.1,
           "humidity":62.0,"maturity_index":480.0,
           "curing_method":"Steam","season":"Summer",
           "cement_content":400,"w_c_ratio":0.48}'
"""

# ─── Standard Library ─────────────────────────────────────────
import os
import sys
import json
import time
import math
import random
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

# ─── Third-Party ──────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    sys.exit("ERROR: Flask not installed. Run: pip install flask numpy pandas scikit-learn joblib")

try:
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
except ImportError:
    sys.exit("ERROR: ML libraries missing. Run: pip install numpy pandas scikit-learn joblib")


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CureLogic")

# -- ADD THESE LINES --
from dotenv import load_dotenv

# This automatically finds and loads the variables from your .env file
load_dotenv() 

USE_LIVE_WEATHER = True        
# os.getenv pulls the value securely; if not found, it defaults to None
OW_API_KEY       = os.getenv("OW_API_KEY") 
CITY             = os.getenv("CITY", "Pune,IN")

# ══════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Device-Key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════
TARGET_STRENGTH = 25.0      # MPa — M30 demoulding threshold
MOULD_COST_HR   = 450.0     # INR per hour mould is occupied
T_DATUM         = -10.0     # Nurse-Saul datum temperature °C
MAX_HISTORY     = 120       # Rolling buffer size
MODEL_DIR       = Path(__file__).parent / "models"

CURE_COSTS = {"Wet": 12.5, "Steam": 22.0, "Membrane": 8.0}
SEASON_TEMP = {"Summer": 38, "Monsoon": 30, "Winter": 18, "Spring": 28}
METHOD_ENC  = {"Wet": 0, "Steam": 1, "Membrane": 2}
SEASON_ENC  = {"Summer": 0, "Monsoon": 1, "Winter": 2, "Spring": 3}

VALID_DEVICE_KEYS = {"CL-SECRET-2024", ""}   # empty = internal simulator

FEATURE_COLS = [
    "ambient_temp", "humidity", "temp_core",
    "w_c_ratio", "cement_content", "log_maturity",
    "thermal_delta", "heat_stress", "method_enc", "season_enc",
]


# ══════════════════════════════════════════════════════════════
# SYNTHETIC TRAINING DATA  (runs once if no .pkl found)
# ══════════════════════════════════════════════════════════════
def _generate_training_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate physically plausible training samples.
    Strength follows Nurse-Saul maturity law:
        f(M) = a + b * ln(M)   where a=-12.4, b=7.8  (typical M30)
    """
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        method  = rng.choice(list(CURE_COSTS))
        season  = rng.choice(list(SEASON_TEMP))
        cement  = rng.choice([350, 370, 400, 420, 450])
        w_c     = rng.uniform(0.35, 0.60)
        ambient = SEASON_TEMP[season] + rng.normal(0, 3)
        humidity= rng.uniform(40, 95)
        elapsed = rng.uniform(6, 72)

        # Temperature model (curing heat curve)
        peak    = 18 * math.exp(-((elapsed - 10) ** 2) / 60)
        core    = ambient + peak + cement * 0.025
        mid     = core * 0.87 + ambient * 0.13
        surface = core * 0.72 + ambient * 0.28
        avg_t   = (core + mid + surface) / 3.0

        # Maturity (Nurse-Saul) with method modifier
        method_factor = {"Wet": 1.0, "Steam": 1.30, "Membrane": 0.90}[method]
        maturity = max(avg_t - T_DATUM, 0) * elapsed * method_factor
        maturity = max(maturity + rng.normal(0, 10), 1)

        # Strength
        strength = -12.4 + 7.8 * math.log(maturity)
        strength = float(np.clip(strength + rng.normal(0, 1.2), 0, 60))

        # Time-to-target
        target_mat = math.exp((TARGET_STRENGTH + 12.4) / 7.8)
        rate = max(avg_t - T_DATUM, 1) * method_factor
        time_to_target = max(target_mat / rate, 6.0) + rng.normal(0, 1.5)
        time_to_target = max(time_to_target, 6.0)

        rows.append({
            "ambient_temp":   round(ambient, 2),
            "humidity":       round(humidity, 1),
            "temp_core":      round(core, 2),
            "w_c_ratio":      round(w_c, 3),
            "cement_content": cement,
            "log_maturity":   math.log1p(maturity),
            "thermal_delta":  round(core - ambient, 2),
            "heat_stress":    round(ambient * humidity / 100.0, 2),
            "method_enc":     METHOD_ENC[method],
            "season_enc":     SEASON_ENC[season],
            "strength":       round(strength, 3),
            "time_hrs":       round(time_to_target, 2),
        })
    return pd.DataFrame(rows)


def train_and_save_models():
    """Train SVR models and persist to disk."""
    MODEL_DIR.mkdir(exist_ok=True)
    log.info("Training SVM models on synthetic data — please wait...")

    df = _generate_training_data(2000)
    X  = df[FEATURE_COLS].values
    y_strength = df["strength"].values
    y_time     = df["time_hrs"].values

    def _make_pipeline():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.5)),
        ])

    strength_model = _make_pipeline()
    strength_model.fit(X, y_strength)
    joblib.dump(strength_model, MODEL_DIR / "model_strength.pkl")

    time_model = _make_pipeline()
    time_model.fit(X, y_time)
    joblib.dump(time_model, MODEL_DIR / "model_time.pkl")

    log.info("✓  SVM models trained and saved to ./models/")
    return strength_model, time_model


def load_or_train_models():
    sp = MODEL_DIR / "model_strength.pkl"
    tp = MODEL_DIR / "model_time.pkl"
    if sp.exists() and tp.exists():
        try:
            sm = joblib.load(sp)
            tm = joblib.load(tp)
            log.info("✓  SVM models loaded from disk")
            return sm, tm
        except Exception as e:
            log.warning(f"Model load failed ({e}) — retraining...")
    return train_and_save_models()


strength_model, time_model = load_or_train_models()


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def build_features(payload: dict) -> np.ndarray:
    """Convert raw sensor/API payload → SVM feature vector (1 × 10)."""
    ambient  = float(payload.get("ambient_temp", 32.0))
    humidity = float(payload.get("humidity", 60.0))
    core     = float(payload.get("temp_core", ambient + 15))
    w_c      = float(payload.get("w_c_ratio", 0.48))
    cement   = float(payload.get("cement_content", 400))
    maturity = float(payload.get("maturity_index", 1.0))
    method   = payload.get("curing_method", "Wet")
    season   = payload.get("season", "Summer")

    return np.array([[
        ambient,
        humidity,
        core,
        w_c,
        cement,
        math.log1p(max(maturity, 0)),
        core - ambient,                    # thermal_delta
        ambient * humidity / 100.0,        # heat_stress
        float(METHOD_ENC.get(method, 0)),
        float(SEASON_ENC.get(season, 0)),
    ]], dtype=np.float64)


# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════
def run_prediction(payload: dict) -> dict:
    """Run both SVM models; return enriched prediction dict."""
    X = build_features(payload)

    pred_strength = float(np.clip(strength_model.predict(X)[0], 0, 60))
    pred_time     = float(max(time_model.predict(X)[0], 6.0))

    method        = payload.get("curing_method", "Wet")
    elapsed       = float(payload.get("elapsed_hours", 0.0))
    method_cost   = CURE_COSTS.get(method, 12.5)

    cumulative_cost = (method_cost + MOULD_COST_HR) * elapsed
    projected_cost  = (method_cost + MOULD_COST_HR) * pred_time
    strength_pct    = min((pred_strength / TARGET_STRENGTH) * 100.0, 100.0)

    # ETA via Nurse-Saul
    ambient  = float(payload.get("ambient_temp", 32.0))
    core     = float(payload.get("temp_core", ambient + 15))
    avg_temp = (ambient + core) / 2.0
    mat_rate = max(avg_temp - T_DATUM, 1.0)
    cur_mat  = float(payload.get("maturity_index", 0.0))
    tgt_mat  = math.exp((TARGET_STRENGTH + 12.4) / 7.8)
    eta      = max((tgt_mat - cur_mat) / mat_rate, 0.0)

    return {
        "pred_strength_mpa": round(pred_strength, 3),
        "pred_time_hrs":     round(pred_time, 2),
        "strength_pct":      round(strength_pct, 1),
        "eta_hours":         round(eta, 2),
        "cumulative_cost":   round(cumulative_cost, 2),
        "projected_cost":    round(projected_cost, 2),
        "demoulding_ready":  pred_strength >= TARGET_STRENGTH,
        "model_version":     "SVR-RBF-v2",
    }


# ══════════════════════════════════════════════════════════════
# SCENARIO ENGINE
# ══════════════════════════════════════════════════════════════
def compute_scenarios(live_weather: dict = None) -> list:
    """Evaluate 36 curing strategy combos; return ranked list."""
    if live_weather is None:
        live_weather = {"ambient_temp": 32.0, "humidity": 60.0}

    scenarios = []
    for method, method_cost in CURE_COSTS.items():
        for season, season_temp in SEASON_TEMP.items():
            for cement in [370, 400, 430]:
                eff_temp  = (season_temp + float(live_weather.get("ambient_temp", season_temp))) / 2.0
                humidity  = float(live_weather.get("humidity", 60.0))
                core_temp = eff_temp + cement * 0.04

                payload = {
                    "ambient_temp":   eff_temp,
                    "humidity":       humidity,
                    "temp_core":      core_temp,
                    "w_c_ratio":      0.48,
                    "cement_content": cement,
                    "maturity_index": max((eff_temp - T_DATUM) * 24, 1),
                    "curing_method":  method,
                    "season":         season,
                    "elapsed_hours":  0.0,
                }
                result     = run_prediction(payload)
                total_cost = (method_cost + MOULD_COST_HR) * result["pred_time_hrs"]

                # Savings vs wet curing baseline
                wet_payload = dict(payload, curing_method="Wet")
                wet_result  = run_prediction(wet_payload)
                wet_cost    = (CURE_COSTS["Wet"] + MOULD_COST_HR) * wet_result["pred_time_hrs"]
                savings     = wet_cost - total_cost

                # Composite score: lower = better
                score = (result["pred_time_hrs"] / 48.0) * 0.5 + \
                        (total_cost / max(result["pred_strength_mpa"], 1) / 500.0) * 0.5

                if result["pred_strength_mpa"] >= TARGET_STRENGTH:
                    scenarios.append({
                        "method":        method,
                        "season":        season,
                        "cement_kg_m3":  cement,
                        "strength_mpa":  round(result["pred_strength_mpa"], 2),
                        "cure_time_hrs": round(result["pred_time_hrs"], 1),
                        "total_cost":    round(total_cost),
                        "savings_inr":   round(savings),
                        "score":         round(score, 4),
                    })

    scenarios.sort(key=lambda x: x["score"])
    for i, s in enumerate(scenarios):
        s["rank"] = i + 1
    return scenarios


# ══════════════════════════════════════════════════════════════
# SHARED LIVE STATE
# ══════════════════════════════════════════════════════════════
state_lock     = threading.Lock()
sensor_history = deque(maxlen=MAX_HISTORY)

live_state = {
    "device_id":         "CL-NODE-01",
    "batch_id":          "CLB-2025-001",
    "elapsed_hours":     0.0,
    "temp_surface":      28.0,
    "temp_mid":          34.0,
    "temp_core":         41.0,
    "maturity_index":    0.0,
    "strength_mpa":      0.0,
    "strength_pct":      0.0,
    "eta_hours":         None,
    "cumulative_cost":   0.0,
    "projected_cost":    0.0,
    "demoulded":         False,
    "prediction_source": "svm",
    "last_updated":      datetime.now(timezone.utc).isoformat(),
    "_demould_notified": False,
}

import requests
_last_weather_fetch = 0
_cached_weather = None
WEATHER_CACHE_TTL = 300   # seconds (5 minutes)

def fetch_live_weather():
    global _last_weather_fetch, _cached_weather
    if not OW_API_KEY:
       print("Weather API key not configured.")
       return None

    if not USE_LIVE_WEATHER or not OW_API_KEY:
        return None

    now = time.time()

    # Return cached if still valid
    if _cached_weather and (now - _last_weather_fetch) < WEATHER_CACHE_TTL:
        return _cached_weather

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={CITY}&appid={OW_API_KEY}&units=metric"
        )

        r = requests.get(url, timeout=5)
        data = r.json()

        _cached_weather = {
            "ambient_temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"]
        }
        _last_weather_fetch = now

        return _cached_weather

    except Exception as e:
        log.warning(f"Live weather fetch failed: {e}")
        return _cached_weather

def _ingest_sensor(payload: dict):
    """Process sensor reading, run SVM, update live_state."""
    # ─── Inject Live Weather Here ───
    live_weather = fetch_live_weather()
    if live_weather:
       payload["external_ambient"] = live_weather["ambient_temp"]
       payload["external_humidity"] = live_weather["humidity"]

    prediction = run_prediction(payload)

    with state_lock:
        live_state.update({
            "device_id":       payload.get("device_id", "UNKNOWN"),
            "elapsed_hours":   float(payload.get("elapsed_hours", 0)),
            "temp_surface":    float(payload.get("temp_surface", 0)),
            "temp_mid":        float(payload.get("temp_mid", 0)),
            "temp_core":       float(payload.get("temp_core", 0)),
            "maturity_index":  float(payload.get("maturity_index", 0)),
            "strength_mpa":    prediction["pred_strength_mpa"],
            "strength_pct":    prediction["strength_pct"],
            "eta_hours":       prediction["eta_hours"],
            "cumulative_cost": prediction["cumulative_cost"],
            "projected_cost":  prediction["projected_cost"],
            "demoulded":       prediction["demoulding_ready"],
            "last_updated":    datetime.now(timezone.utc).isoformat(),
            "external_ambient": live_weather["ambient_temp"] if live_weather else None,
            "external_humidity": live_weather["humidity"] if live_weather else None,
        })

        sensor_history.append({
            "ts":       live_state["elapsed_hours"],
            "core":     live_state["temp_core"],
            "mid":      live_state["temp_mid"],
            "surface":  live_state["temp_surface"],
            "strength": live_state["strength_mpa"],
            "maturity": live_state["maturity_index"],
            "cost":     live_state["cumulative_cost"],
        })

        # One-shot demould notification
        if prediction["demoulding_ready"] and not live_state["_demould_notified"]:
            live_state["_demould_notified"] = True
            log.info(
                f"✓ DEMOULDING READY | "
                f"Strength={prediction['pred_strength_mpa']}MPa | "
                f"Cost=₹{prediction['cumulative_cost']:.0f} | "
                f"Elapsed={payload.get('elapsed_hours', 0):.1f}hrs"
            )

    if float(payload.get("temp_core", 0)) > 70:
        log.warning(
            f"THERMAL ALERT: core={payload['temp_core']:.1f}°C "
            f"on {payload.get('device_id', '?')}"
        )

    log.info(
        f"[IoT] {payload.get('device_id', '?')} | "
        f"T_core={payload.get('temp_core', 0):.1f}°C | "
        f"Maturity={payload.get('maturity_index', 0):.0f} | "
        f"Strength={prediction['pred_strength_mpa']:.2f}MPa | "
        f"ETA={prediction['eta_hours']:.1f}hr"
    )


# ══════════════════════════════════════════════════════════════
# IoT SIMULATOR
# ══════════════════════════════════════════════════════════════
class IoTSimulator:
    """
    Mimics NodeMCU firmware posting every ~60 simulated seconds.
    Runs in a daemon thread; injects data directly via _ingest_sensor().
    """

    def __init__(self):
        self.running  = False
        self.thread   = None
        self.tick     = 0
        self.elapsed  = 0.0
        self.maturity = 0.0
        self.season   = "Summer"
        self.method   = "Steam"
        self.cement   = 400

    def _step(self):
        self.tick    += 1
        self.elapsed += 1.0 / 60.0          # 1 simulated minute per tick

        ambient_base = SEASON_TEMP.get(self.season, 32.0)
        ambient      = ambient_base + random.gauss(0, 1.5)

        # Hydration heat curve
        peak       = 22.0 * math.exp(-((self.elapsed - 10.0) ** 2) / 50.0)
        core       = ambient + peak + self.cement * 0.025 + random.gauss(0, 0.6)
        mid        = core * 0.87 + ambient * 0.13 + random.gauss(0, 0.5)
        surface    = core * 0.71 + ambient * 0.29 + random.gauss(0, 0.8)

        avg_t      = (core + mid + surface) / 3.0
        self.maturity += max(avg_t - T_DATUM, 0.0) / 60.0   # per-minute accumulation

        _ingest_sensor({
            "device_id":      "CL-NODE-SIM",
            "elapsed_hours":  round(self.elapsed, 4),
            "ambient_temp":   round(ambient, 2),
            "humidity":       round(55 + random.gauss(0, 5), 1),
            "temp_surface":   round(surface, 2),
            "temp_mid":       round(mid, 2),
            "temp_core":      round(core, 2),
            "maturity_index": round(self.maturity, 2),
            "curing_method":  self.method,
            "season":         self.season,
            "cement_content": self.cement,
            "w_c_ratio":      0.48,
            "simulated":      True,
        })

    def _loop(self):
        while self.running:
            try:
                self._step()
            except Exception as e:
                log.error(f"Simulator step error: {e}")
            time.sleep(3)   # 3s wall-clock = 1 simulated minute

    def start(self, season="Summer", method="Steam", cement=400):
        if self.running:
            self.stop()
        self.season   = season
        self.method   = method
        self.cement   = int(cement)
        self.tick     = 0
        self.elapsed  = 0.0
        self.maturity = 0.0
        self.running  = True
        self.thread   = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        log.info(f"IoT Simulator started | Season={season} Method={method} Cement={cement}kg")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        log.info("IoT Simulator stopped")


simulator = IoTSimulator()


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

# ── Dashboard HTML ────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the dashboard HTML file if it exists next to app.py."""
    for name in ["curelogic_v3.html", "sagar_dashboard_v2.html", "index.html"]:
        path = Path(__file__).parent / name
        if path.exists():
            return path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}
    return (
        "<h2>CureLogic backend is running ✓</h2>"
        "<p>Place <code>curelogic_v3.html</code> next to <code>app.py</code> "
        "to serve the dashboard at this URL.</p>"
        "<ul>"
        "<li><a href='/api/status'>/api/status</a></li>"
        "<li><a href='/api/predict'>/api/predict</a></li>"
        "<li><a href='/api/scenarios'>/api/scenarios</a></li>"
        "<li><a href='/api/live'>/api/live  (SSE)</a></li>"
        "</ul>",
        200,
        {"Content-Type": "text/html"},
    )


# ── OPTIONS pre-flight (CORS) ─────────────────────────────────
@app.route("/api/sensor_data", methods=["OPTIONS"])
@app.route("/api/predict",     methods=["OPTIONS"])
@app.route("/api/simulator/start", methods=["OPTIONS"])
@app.route("/api/simulator/stop",  methods=["OPTIONS"])
def options_handler():
    return jsonify({}), 200


# ── IoT Data Ingestion ────────────────────────────────────────
@app.route("/api/sensor_data", methods=["POST"])
def receive_sensor_data():
    """
    NodeMCU (or curl test) posts JSON here.
    Validates X-Device-Key header, runs SVM, returns prediction.

    Expected fields:
        device_id, elapsed_hours, ambient_temp, temp_surface,
        temp_mid, temp_core, humidity, maturity_index,
        curing_method, season, cement_content, w_c_ratio
    """
    # Auth check
    key = request.headers.get("X-Device-Key", "")
    if key not in VALID_DEVICE_KEYS:
        log.warning(f"Rejected request with bad device key: '{key}'")
        return jsonify({"error": "Unauthorized — invalid X-Device-Key"}), 401

    # Parse body
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Empty or invalid JSON body"}), 400

    # Validate required fields
    required = ["elapsed_hours", "temp_core", "maturity_index"]
    missing  = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 422

    try:
        _ingest_sensor(payload)

        with state_lock:
            resp = {
                "status":    "ok",
                "timestamp": live_state["last_updated"],
                "device_id": payload.get("device_id", "unknown"),
                "prediction": {
                    "strength_mpa":  live_state["strength_mpa"],
                    "strength_pct":  live_state["strength_pct"],
                    "eta_hours":     live_state["eta_hours"],
                    "cost_inr":      live_state["cumulative_cost"],
                    "demould_ready": live_state["demoulded"],
                    "model":         "SVR-RBF-v2",
                },
            }
        return jsonify(resp), 200

    except Exception as e:
        log.exception("sensor_data error")
        return jsonify({"error": str(e)}), 500


# ── On-Demand Prediction ──────────────────────────────────────
@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    """
    GET  → returns current live prediction state
    POST → runs SVM on any custom payload you provide
    """
    if request.method == "GET":
        with state_lock:
            data = dict(live_state)
            data["history"] = list(sensor_history)
        return jsonify(data), 200

    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = run_prediction(payload)
        return jsonify({"status": "ok", "prediction": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Scenario Rankings ─────────────────────────────────────────
@app.route("/api/scenarios", methods=["GET"])
def scenarios():
    """Return all 36 ranked curing strategies based on live conditions."""
    try:
        with state_lock:
            weather = fetch_live_weather() or {
    "ambient_temp": live_state.get("ambient_temp", 32.0),
    "humidity": live_state.get("humidity", 60.0),
}
        ranked = compute_scenarios(weather)
        return jsonify({
            "status":    "ok",
            "count":     len(ranked),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenarios": ranked,
        }), 200
    except Exception as e:
        log.exception("scenarios error")
        return jsonify({"error": str(e)}), 500


# ── SSE Live Stream ───────────────────────────────────────────
@app.route("/api/live")
def live_stream():
    """
    Server-Sent Events endpoint.
    Dashboard JS connects here and receives state every 2 seconds.
    """
    def generate():
        last_sent = None
        while True:
            with state_lock:
                snapshot = dict(live_state)
                # Send last 30 history points for chart rendering
                snapshot["history_tail"] = list(sensor_history)[-30:]

            ts = snapshot.get("last_updated")
            if ts != last_sent:
                last_sent = ts
                try:
                    yield f"data: {json.dumps(snapshot)}\n\n"
                except GeneratorExit:
                    return
            time.sleep(2)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        },
    )


# ── Simulator Control ─────────────────────────────────────────
@app.route("/api/simulator/start", methods=["POST"])
def start_simulator():
    body   = request.get_json(force=True, silent=True) or {}
    season = body.get("season", "Summer")
    method = body.get("method", "Steam")
    cement = int(body.get("cement", 400))

    if season not in SEASON_TEMP:
        return jsonify({"error": f"Invalid season. Choose from: {list(SEASON_TEMP)}"}), 422
    if method not in CURE_COSTS:
        return jsonify({"error": f"Invalid method. Choose from: {list(CURE_COSTS)}"}), 422

    simulator.start(season=season, method=method, cement=cement)
    return jsonify({
        "status":  "simulator_started",
        "season":  season,
        "method":  method,
        "cement":  cement,
    }), 200


@app.route("/api/simulator/stop", methods=["POST"])
def stop_simulator():
    simulator.stop()
    return jsonify({"status": "simulator_stopped"}), 200


# ── System Status ─────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def status():
    with state_lock:
        readings = len(sensor_history)
        strength = live_state.get("strength_mpa", 0.0)
        elapsed  = live_state.get("elapsed_hours", 0.0)

    return jsonify({
        "status":           "operational",
        "version":          "2.0.0",
        "models_loaded":    True,
        "model_version":    "SVR-RBF-v2",
        "readings":         readings,
        "simulator_on":     simulator.running,
        "current_strength": round(strength, 3),
        "elapsed_hours":    round(elapsed, 2),
        "target_mpa":       TARGET_STRENGTH,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "curing_ambient":   live_state.get("ambient_temp", 32.0),
        "external_ambient": live_state.get("external_ambient"),
        "external_humidity": live_state.get("external_humidity"),
    }), 200


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    bar = "═" * 58
    log.info(bar)
    log.info("  CureLogic Backend v2.0")
    log.info("  L&T CreaTech Hackathon | Problem Statement 01")
    log.info(bar)
    log.info("  Dashboard  →  http://localhost:5000/")
    log.info("  IoT POST   →  http://localhost:5000/api/sensor_data")
    log.info("  Predict    →  http://localhost:5000/api/predict")
    log.info("  Scenarios  →  http://localhost:5000/api/scenarios")
    log.info("  Live SSE   →  http://localhost:5000/api/live")
    log.info("  Status     →  http://localhost:5000/api/status")
    log.info(bar)

    # Auto-start simulator so the dashboard has data immediately
    # simulator.start(season="Summer", method="Steam", cement=400)
    log.info("  IoT Simulator auto-started (demo mode)")
    log.info(bar)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)