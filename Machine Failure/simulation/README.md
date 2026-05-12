# Predictive Maintenance Live Console

Interactive Streamlit simulator built on top of the *Machine Failure Prediction* report.
It uses the same three models (Logistic Regression, SVM-RBF, Neural Network) and the
same AI4I 2020 dataset, and exposes them through a live console with:

- **Animated industrial-motor SVG** (spins faster as RPM rises, vibrates and glows red as failure risk grows).
- **Status cards** showing `P(Failure)`, machine status (`NORMAL` / `WARNING` / `CRITICAL`), recommended action, plus a what-if strip comparing all three models at the current operating point.
- **2D decision-boundary heatmap** — the centerpiece — showing the chosen model's failure probability across the operating space, with training data overlaid and the live operating point pinned on top.
- **Auto-pilot mode** — Play / Pause / Reset a 30-second shift profile that takes the machine through *idle → ramp-up → stress test → cool-down*, animating the operating point across the heatmap and tracing its path.

## Quick start

```bash
pip install -r requirements.txt
python train_models.py                 # one-time: trains and saves the 3 models
streamlit run main.py                  # launches on http://localhost:8501
```

## Layout

```
simulation/
├── main.py                  # Streamlit entry — assembles the layout
├── train_models.py          # Trains LogReg, SVM, MLP; saves joblibs + train/test parquet
├── requirements.txt
├── README.md
├── .streamlit/config.toml   # Pins the app to light theme
├── components/
│   ├── motor.py             # Animated industrial-motor SVG (CSS spin/shake/glow)
│   ├── boundary.py          # 2D decision-boundary heatmap (matplotlib)
│   ├── status.py            # Status cards + what-if comparison strip
│   └── autopilot.py         # 30-second shift profile + state machine
└── models/                  # Filled by train_models.py
    ├── logistic_regression.joblib
    ├── svm.joblib
    ├── neural_network.joblib
    ├── train.parquet
    └── test.parquet
```

## How the visuals respond

| Risk level | P(Failure) | Motor visual | Status |
|---|---|---|---|
| Normal   | < 25 % | Smooth spin, soft green glow | NORMAL  ✓ |
| Warning  | 25–60 % | Mild vibration, amber glow | WARNING ⚠ |
| Critical | ≥ 60 % | Strong vibration, pulsing red halo | CRITICAL ■ |

The spin speed of the shaft and cooling fan is tied directly to the RPM slider so the motor "feels" alive as you operate it.

## Why this dataset suits a 2D visualization

The AI4I-2020 task in the report uses only **two numeric features** — rotational speed and torque. Most ML projects can't visualize their decision boundary because they have many features; this one can plot it directly. The heatmap therefore is not a stylized abstraction — it is the *actual* decision surface of the trained classifier.

That makes one feature of the demo especially clear: when you switch between Logistic Regression (linear boundary), SVM-RBF (smooth non-linear ring), and Neural Network (more flexible non-linear region), the heatmap visibly changes shape — illustrating why the Neural Network achieves the highest AUC and F1 in the report.
