"""2D decision-boundary heatmap for the chosen model.

Renders Rotational speed (x) × Torque (y) as a contour-filled heatmap of the
model's P(failure), overlays training points (failures vs normal), and marks
the user's current operating point.
"""
from __future__ import annotations

import io
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["Rotational speed [rpm]", "Torque [Nm]"]

# Slightly inset from the dataset min/max so the boundary visual is clean
X_MIN, X_MAX = 1100, 2950
Y_MIN, Y_MAX = 0, 80

GRID_RES = 200  # resolution of the heatmap grid


@lru_cache(maxsize=8)
def _grid(grid_res: int = GRID_RES):
    xx, yy = np.meshgrid(
        np.linspace(X_MIN, X_MAX, grid_res),
        np.linspace(Y_MIN, Y_MAX, grid_res),
    )
    return xx, yy


@st.cache_data(show_spinner=False)
def _prob_grid(_model_id: str, _model_hash: int, grid_res: int = GRID_RES):
    """Compute P(failure) over the grid. Cached per model identity."""
    # Re-load the model from disk via streamlit-caching upstream; here we just
    # take a marker string + an identity hash so the cache key changes when the
    # model object changes.
    raise RuntimeError("Use _compute_prob_grid(model) instead.")


def _compute_prob_grid(model, grid_res: int = GRID_RES):
    xx, yy = _grid(grid_res)
    grid_df = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=FEATURES,
    )
    proba = model.predict_proba(grid_df)[:, 1]
    return xx, yy, proba.reshape(xx.shape)


def render_boundary(model, model_name: str, train_df: pd.DataFrame,
                    rpm: float, torque: float, p_failure: float,
                    trail: list[tuple[float, float]] | None = None) -> None:
    """Render the 2D decision-boundary heatmap with the current op point.

    Args:
        model:      fitted sklearn pipeline with predict_proba
        model_name: shown in the title
        train_df:   training data (with TARGET column) for the scatter overlay
        rpm, torque, p_failure: current operating point and its prediction
        trail: optional list of recent (rpm, torque) for auto-pilot trail
    """
    xx, yy, zz = _compute_prob_grid(model)

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Filled contour heatmap (P(failure) 0..1) — light-theme friendly palette
    levels = np.linspace(0, 1, 21)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="RdYlGn_r", alpha=0.85)
    cbar = fig.colorbar(cf, ax=ax, pad=0.015, ticks=np.linspace(0, 1, 6))
    cbar.set_label("P(Failure)", fontsize=11, color="#102A55")
    cbar.ax.tick_params(labelsize=9, colors="#102A55")

    # Decision-boundary contour line at P=0.5
    cs = ax.contour(xx, yy, zz, levels=[0.5], colors="#102A55",
                    linewidths=1.8, linestyles="--")
    ax.clabel(cs, fmt={0.5: "P = 0.5"}, fontsize=9, inline=True)

    # Training data overlay
    normal = train_df[train_df["Machine failure"] == 0].sample(
        n=min(1200, (train_df["Machine failure"] == 0).sum()),
        random_state=42,
    )
    fail = train_df[train_df["Machine failure"] == 1]
    ax.scatter(normal[FEATURES[0]], normal[FEATURES[1]],
               s=8, color="#2E7D32", alpha=0.18, label="Normal (training)",
               edgecolors="none")
    ax.scatter(fail[FEATURES[0]], fail[FEATURES[1]],
               s=22, color="#C62828", alpha=0.85, label="Failure (training)",
               edgecolors="#7f0000", linewidths=0.4)

    # Auto-pilot trail
    if trail:
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], color="#FFB300", lw=1.8, alpha=0.55,
                zorder=4)

    # Current operating point — pulsing-style marker
    point_color = ("#2E7D32" if p_failure < 0.25
                   else ("#F57C00" if p_failure < 0.60 else "#C62828"))
    ax.scatter([rpm], [torque], s=420, marker="o", facecolor=point_color,
               edgecolor="#102A55", linewidth=2.5, zorder=5,
               label="Operating point")
    ax.scatter([rpm], [torque], s=900, marker="o", facecolor="none",
               edgecolor=point_color, linewidth=1.4, alpha=0.5, zorder=5)

    # Annotations
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Rotational speed [rpm]", fontsize=11, color="#102A55")
    ax.set_ylabel("Torque [Nm]", fontsize=11, color="#102A55")
    ax.set_title(f"{model_name} — failure-probability map",
                 fontsize=13, fontweight="bold", color="#102A55")
    ax.tick_params(colors="#102A55")
    for sp in ax.spines.values():
        sp.set_color("#d6dff0")
    ax.grid(True, alpha=0.25, color="#102A55", linestyle=":")

    leg = ax.legend(loc="upper right", framealpha=0.95, fontsize=9,
                    edgecolor="#d6dff0")
    leg.get_frame().set_facecolor("#ffffff")
    for txt in leg.get_texts():
        txt.set_color("#102A55")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)
