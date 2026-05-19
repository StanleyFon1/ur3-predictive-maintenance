"""
dashboard/app.py – UR3 Predictive Maintenance Dashboard
Launch: streamlit run dashboard/app.py
"""

import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UR3 Predictive Maintenance",
    page_icon="🤖",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_COLS = [
    "current_j1", "current_j2", "current_j3",
    "temperature_j1", "temperature_j2",
    "vibration",
]
THRESHOLD = 0.5

# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model        = joblib.load("models/rf_model.pkl")
    scaler       = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artefacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def simulate_reading(anomaly: bool = False) -> dict:
    """Generate one synthetic sensor reading."""
    rng = np.random.default_rng()
    if anomaly:
        return dict(
            current_j1    = float(rng.normal(3.20, 0.40)),
            current_j2    = float(rng.normal(3.60, 0.50)),
            current_j3    = float(rng.normal(2.80, 0.35)),
            temperature_j1= float(rng.normal(62.0,  5.0)),
            temperature_j2= float(rng.normal(65.0,  6.0)),
            vibration     = float(rng.normal(0.18, 0.05)),
        )
    return dict(
        current_j1    = float(rng.normal(2.10, 0.15)),
        current_j2    = float(rng.normal(2.40, 0.18)),
        current_j3    = float(rng.normal(1.80, 0.12)),
        temperature_j1= float(rng.normal(45.0,  2.0)),
        temperature_j2= float(rng.normal(47.0,  2.5)),
        vibration     = float(rng.normal(0.05, 0.01)),
    )

def build_feature_row(reading: dict, history: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Combine raw reading with rolling stats to match training feature set."""
    base = {col: reading[col] for col in BASE_COLS}
    tail = history[BASE_COLS].tail(window) if len(history) >= window else pd.DataFrame([reading] * window)
    for col in BASE_COLS:
        base[f"{col}_roll_mean"] = tail[col].mean()
        base[f"{col}_roll_std"]  = tail[col].std() if len(tail) > 1 else 0.0
    return pd.DataFrame([base])[feature_names]

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=BASE_COLS + ["anomaly_score", "prediction"]
    )
if "running" not in st.session_state:
    st.session_state.running = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()

    inject_anomaly = st.toggle("💥 Inject Anomaly", value=False,
                               help="Forces readings into anomalous ranges")
    speed = st.slider("Update Interval (s)", 0.5, 3.0, 1.0, 0.5)
    max_points = st.slider("History Window (points)", 20, 200, 60, 10)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start" if not st.session_state.running else "⏹ Stop",
                     use_container_width=True):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = pd.DataFrame(
                columns=BASE_COLS + ["anomaly_score", "prediction"]
            )

    st.divider()
    st.caption("**Model:** Random Forest (100 trees)")
    st.caption("**Dataset:** UR3 CobotOps (synthetic)")
    st.caption("**Author:** Stanley Fon")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 UR3 Robotic Arm — Predictive Maintenance")
st.caption("Real-time health monitoring · Anomaly detection · Industry 4.0")

if not model_loaded:
    st.error("⚠️ Model not found. Run `python src/train.py` first, then relaunch the dashboard.")
    st.stop()

# ── Ingest one reading when running ──────────────────────────────────────────
if st.session_state.running:
    reading  = simulate_reading(anomaly=inject_anomaly)
    feat_row = build_feature_row(reading, st.session_state.history)
    scaled   = scaler.transform(feat_row)
    prob     = float(model.predict_proba(scaled)[0][1])
    pred     = int(prob >= THRESHOLD)

    new_row = {**reading, "anomaly_score": prob, "prediction": pred}
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([new_row])],
        ignore_index=True,
    ).tail(max_points)

# ── Dashboard layout ──────────────────────────────────────────────────────────
hist  = st.session_state.history
total = len(hist)

# KPI row
k1, k2, k3, k4 = st.columns(4)
n_anomalies  = int(hist["prediction"].sum()) if total else 0
latest_score = float(hist["anomaly_score"].iloc[-1]) if total else 0.0
health_pct   = max(0, 100 - int(latest_score * 100))

k1.metric("📊 Total Readings",     total)
k2.metric("🚨 Anomalies Detected", n_anomalies)
k3.metric("📈 Anomaly Score",       f"{latest_score:.3f}")
status_icon = "🟢" if health_pct > 70 else ("🟡" if health_pct > 40 else "🔴")
k4.metric("💚 Health Index",        f"{status_icon}  {health_pct} %")

st.divider()

# Charts – row 1
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Anomaly Score Over Time")
    if total:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=hist["anomaly_score"], mode="lines+markers",
            line=dict(color="#ef4444", width=2), marker=dict(size=4),
            name="Score",
        ))
        fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold ({THRESHOLD})")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(range=[0, 1], title="Probability"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet — press ▶ Start.")

with c_right:
    st.subheader("Joint Currents (A)")
    if total:
        fig2 = go.Figure()
        colours = ["#3b82f6", "#10b981", "#f59e0b"]
        for col, clr in zip(["current_j1", "current_j2", "current_j3"], colours):
            fig2.add_trace(go.Scatter(y=hist[col], mode="lines",
                                      name=col, line=dict(color=clr, width=1.8)))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# Charts – row 2
c_b1, c_b2 = st.columns(2)

with c_b1:
    st.subheader("Joint Temperatures (°C)")
    if total:
        fig3 = go.Figure()
        for col, clr in zip(["temperature_j1", "temperature_j2"], ["#ec4899", "#8b5cf6"]):
            fig3.add_trace(go.Scatter(y=hist[col], mode="lines",
                                      name=col, line=dict(color=clr, width=1.8)))
        fig3.add_hline(y=60.0, line_dash="dot", line_color="red",
                       annotation_text="Overheat limit (60 °C)")
        fig3.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

with c_b2:
    st.subheader("Vibration (g)")
    if total:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=hist["vibration"], mode="lines",
                                  fill="tozeroy",
                                  line=dict(color="#06b6d4", width=1.5)))
        fig4.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)

# Raw data table
if total:
    st.subheader("Latest Readings")
    display_cols = BASE_COLS + ["anomaly_score", "prediction"]

    def highlight_anomaly(row):
        return ["background-color: #fecaca" if row["prediction"] == 1 else "" for _ in row]

    st.dataframe(
        hist[display_cols].tail(10).style.apply(highlight_anomaly, axis=1),
        use_container_width=True,
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.running:
    time.sleep(speed)
    st.rerun()
else:
    if not total:
        st.info("▶ Press **Start** in the sidebar to begin the live simulation.")