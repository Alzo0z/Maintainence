"""Status cards: P(Failure), machine status, recommended action, model agreement."""
from __future__ import annotations

import streamlit as st

LOW_THRESH = 0.25
HIGH_THRESH = 0.60


def status_for(p_failure: float) -> tuple[str, str, str, str]:
    """Return (label, color, icon, recommended_action) for the given probability."""
    if p_failure < LOW_THRESH:
        return ("NORMAL", "#2E7D32", "✓",
                "Operating safely. Continue normal duty.")
    if p_failure < HIGH_THRESH:
        return ("WARNING", "#F57C00", "⚠",
                "Risk rising — reduce torque or speed; schedule inspection.")
    return ("CRITICAL", "#C62828", "■",
            "Imminent failure risk — stop the machine and inspect immediately.")


def render_status_panel(p_failure: float, model_name: str,
                        all_model_probs: dict[str, float] | None = None) -> None:
    """Render the three status cards in a row."""
    label, color, icon, action = status_for(p_failure)

    cards_css = """
    <style>
      .pm-row {{
        display: grid;
        grid-template-columns: 1.0fr 1.0fr 1.3fr;
        gap: 14px;
        margin: 4px 0 8px 0;
      }}
      .pm-card {{
        background: #ffffff;
        border: 1px solid #d6dff0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 12px rgba(16, 42, 85, 0.06);
        position: relative;
        overflow: hidden;
      }}
      .pm-card .lbl {{
        font-size: 11px;
        color: #6c7a91;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 6px;
      }}
      .pm-card .val {{
        font-size: 38px;
        font-weight: 800;
        color: #102A55;
        line-height: 1.0;
      }}
      .pm-card .sub {{
        font-size: 13px;
        color: #102A55;
        margin-top: 6px;
      }}
      .pm-card.status .icon {{
        position: absolute; top: 12px; right: 16px;
        font-size: 22px; font-weight: 900; color: var(--accent);
      }}
      .pm-bar {{
        height: 8px; background: #eef3fb; border-radius: 4px; overflow: hidden;
        margin-top: 10px;
      }}
      .pm-bar > span {{
        display: block; height: 100%; border-radius: 4px;
      }}
    </style>
    """

    pct = max(0.0, min(1.0, p_failure))
    bar_color = color
    fail_pct = f"{pct * 100:.1f}%"

    html = cards_css + f"""
    <div class="pm-row">
      <div class="pm-card" style="--accent:{color}">
        <div class="lbl">P(Failure)</div>
        <div class="val" style="color:{color}">{fail_pct}</div>
        <div class="pm-bar"><span style="width:{pct * 100:.1f}%;background:{bar_color}"></span></div>
        <div class="sub">model: <strong>{model_name}</strong></div>
      </div>

      <div class="pm-card status" style="--accent:{color}">
        <div class="lbl">Status</div>
        <div class="val" style="color:{color}">{label}</div>
        <div class="icon">{icon}</div>
        <div class="sub">thresholds: warn @ {LOW_THRESH:.0%}, critical @ {HIGH_THRESH:.0%}</div>
      </div>

      <div class="pm-card" style="--accent:{color}">
        <div class="lbl">Recommended action</div>
        <div class="sub" style="font-size:16px;line-height:1.4;color:#102A55;font-weight:600;margin-top:4px;">
          {action}
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if all_model_probs:
        # Inline what-if comparison strip
        cmp_html = """
        <style>
          .pm-cmp {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 4px;
          }
          .pm-cmp-card {
            background: #F4F7FB;
            border: 1px solid #d6dff0;
            border-radius: 8px;
            padding: 10px 14px;
          }
          .pm-cmp-card .name {
            font-size: 11px; color: #6c7a91; letter-spacing: 1.5px;
            font-weight: 700; text-transform: uppercase;
          }
          .pm-cmp-card .v {
            font-size: 22px; font-weight: 800; color: #102A55;
          }
        </style>
        <div class="pm-cmp">
        """
        for name, p in all_model_probs.items():
            _, c, _, _ = status_for(p)
            cmp_html += (
                f'<div class="pm-cmp-card">'
                f'<div class="name">{name}</div>'
                f'<div class="v" style="color:{c}">{p * 100:.1f}%</div>'
                f'</div>'
            )
        cmp_html += "</div>"
        st.markdown(
            '<div style="font-size:11px;color:#6c7a91;'
            'letter-spacing:2px;font-weight:700;margin:8px 0 6px 0;">'
            'WHAT-IF — ALL THREE MODELS AT THIS OPERATING POINT</div>',
            unsafe_allow_html=True,
        )
        st.markdown(cmp_html, unsafe_allow_html=True)
