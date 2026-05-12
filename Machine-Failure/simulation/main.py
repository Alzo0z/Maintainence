"""Predictive Maintenance Live Console — Streamlit demo.

Single page with three stacked panels:
  1. Animated industrial motor SVG (top, ~300px tall).
  2. Status cards: P(failure), STATUS, recommended action + what-if strip.
  3. 2D decision-boundary heatmap with the current operating point.

Controls (sidebar):
  - RPM slider, Torque slider
  - Algorithm picker (LogReg / SVM / Neural Network)
  - Auto-pilot: Play / Pause / Reset (30-second shift profile)

Run:
    streamlit run main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from components.autopilot import (
    PROFILE_DURATION_S, TICK_S, TRAIL_LIMIT, phase_for, profile_at,
)
from components.boundary import FEATURES, render_boundary
from components.motor import RPM_MAX, RPM_MIN, render_motor
from components.status import render_status_panel

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Live Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }
      h1, h2, h3 { color: #102A55; }
      div[data-testid="stSidebar"] { background: #F4F7FB; }
      .pm-header {
        background: linear-gradient(135deg, #102A55 0%, #1565C0 100%);
        color: #ffffff; border-radius: 12px;
        padding: 20px 28px; margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(16, 42, 85, 0.18);
      }
      .pm-header h1 { color: #ffffff; margin: 0; font-size: 26px; letter-spacing: 0.5px; }
      .pm-header .sub { color: #BBDEFB; font-size: 14px; margin-top: 4px;
                       letter-spacing: 2px; text-transform: uppercase; }
      div[data-testid="stMetricValue"] { color: #1565C0; font-weight: 700; }
      .stButton button {
        background: #1565C0; color: white; border-radius: 8px; border: none;
        font-weight: 600;
      }
      .stButton button:hover { background: #0D47A1; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "SVM": "svm.joblib",
    "Neural Network": "neural_network.joblib",
}


@st.cache_resource
def load_models():
    return {name: joblib.load(ROOT / "models" / fn)
            for name, fn in MODEL_FILES.items()}


@st.cache_data
def load_train():
    return pd.read_parquet(ROOT / "models" / "train.parquet")


models = load_models()
train_df = load_train()


# ---------------------------------------------------------------------------
# Session state for auto-pilot
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("rpm", 1500.0)
ss.setdefault("torque", 40.0)
ss.setdefault("algo", "Neural Network")
ss.setdefault("ap_playing", False)
ss.setdefault("ap_t", 0.0)
ss.setdefault("ap_trail", [])


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Controls")

algo = st.sidebar.radio(
    "Algorithm",
    list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index(ss.algo),
    help="Switch live — the decision-boundary map will update.",
)
ss.algo = algo

st.sidebar.markdown("### Operating point")
ap_active = ss.ap_playing
rpm = st.sidebar.slider(
    "Rotational speed [rpm]", float(RPM_MIN), float(RPM_MAX),
    float(ss.rpm), 1.0,
    disabled=ap_active,
    help="Disabled during auto-pilot playback.",
)
torque = st.sidebar.slider(
    "Torque [Nm]", 3.8, 76.6,
    float(ss.torque), 0.1,
    disabled=ap_active,
    help="Disabled during auto-pilot playback.",
)
if not ap_active:
    ss.rpm = rpm
    ss.torque = torque

st.sidebar.markdown("---")
st.sidebar.markdown("### Auto-pilot (30s shift)")
ap_c1, ap_c2, ap_c3 = st.sidebar.columns(3)
if ap_c1.button("▶ Play", use_container_width=True, disabled=ap_active):
    ss.ap_playing = True
    if ss.ap_t >= PROFILE_DURATION_S:
        # restart from beginning if previous run finished
        ss.ap_t = 0.0
        ss.ap_trail = []
    st.rerun()
if ap_c2.button("❚❚ Pause", use_container_width=True, disabled=not ap_active):
    ss.ap_playing = False
    st.rerun()
if ap_c3.button("⟲ Reset", use_container_width=True):
    ss.ap_playing = False
    ss.ap_t = 0.0
    ss.ap_trail = []
    ss.rpm = 1500.0
    ss.torque = 40.0
    st.rerun()

# Progress + phase label
phase = phase_for(ss.ap_t)
st.sidebar.progress(
    min(1.0, ss.ap_t / PROFILE_DURATION_S),
    text=f"{ss.ap_t:0.0f}s / {PROFILE_DURATION_S}s  •  Phase: **{phase}**",
)


# ---------------------------------------------------------------------------
# Auto-pilot tick
# ---------------------------------------------------------------------------
if ss.ap_playing:
    # Refresh every TICK_S seconds while playing
    st_autorefresh(interval=int(TICK_S * 1000), key="autopilot_tick")
    ss.ap_t = min(PROFILE_DURATION_S, ss.ap_t + TICK_S)
    rpm_now, torque_now = profile_at(ss.ap_t)
    ss.rpm = rpm_now
    ss.torque = torque_now
    ss.ap_trail.append((rpm_now, torque_now))
    if len(ss.ap_trail) > TRAIL_LIMIT:
        ss.ap_trail = ss.ap_trail[-TRAIL_LIMIT:]
    if ss.ap_t >= PROFILE_DURATION_S:
        ss.ap_playing = False  # stop at end of shift


# ---------------------------------------------------------------------------
# Predictions for current operating point
# ---------------------------------------------------------------------------
x_now = pd.DataFrame([[ss.rpm, ss.torque]], columns=FEATURES)
chosen = models[algo]
p_fail = float(chosen.predict_proba(x_now)[0, 1])
all_probs = {
    name: float(m.predict_proba(x_now)[0, 1]) for name, m in models.items()
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="pm-header">
      <h1>⚙️ Predictive Maintenance Live Console</h1>
      <div class="sub">AI4I 2020 · Live failure-probability monitor · model: {algo}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Panel 1 — Motor + status cards (side by side)
top_left, top_right = st.columns([1.15, 1.0], gap="medium")
with top_left:
    st.markdown("##### Live machine")
    render_motor(ss.rpm, ss.torque, p_fail, height=310)
with top_right:
    st.markdown("##### Current readout")
    render_status_panel(p_fail, algo, all_probs)

st.markdown("---")

# Panel 2 — Decision-boundary heatmap
st.markdown("##### Decision-boundary map")
st.caption(
    "Background colour = the chosen model's P(Failure) across the operating "
    "space. Green dots = normal training examples, red dots = real failures. "
    "Yellow trail (during auto-pilot) = the path the operating point has taken."
)
render_boundary(
    chosen, algo, train_df,
    rpm=ss.rpm, torque=ss.torque, p_failure=p_fail,
    trail=ss.ap_trail if (ss.ap_trail and (ss.ap_playing or ss.ap_t > 0)) else None,
)

# Footer — test-set scores from the existing report so the demo connects to it
st.markdown("---")
st.markdown("##### Test-set performance (from the project report)")
report_metrics = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "SVM (RBF)", "Neural Network"],
        "Accuracy": [0.969, 0.970, 0.971],
        "Precision": [0.917, 0.800, 0.792],
        "Recall": [0.108, 0.157, 0.186],
        "Specificity": [1.000, 0.999, 0.998],
        "F1-Score": [0.193, 0.262, 0.302],
        "AUC": [0.797, 0.480, 0.877],
    }
).set_index("Model")

st.dataframe(
    report_metrics.style.format("{:.3f}")
    .highlight_max(axis=0, props="background-color: #C8E6C9;")
    .highlight_min(axis=0, props="background-color: #FFCDD2;"),
    use_container_width=True,
)
st.caption(
    "Best per metric is green, worst is red. Note the Neural Network leads on "
    "Accuracy, Recall, F1 and AUC — consistent with the verdict in the report."
)
