"""Animated industrial-motor SVG renderer.

CSS animations control:
  - Shaft spin speed (duration tied to RPM)
  - Body vibration (intensity tied to failure probability)
  - Red warning glow (visible only above the CRITICAL threshold)
"""
from __future__ import annotations

import streamlit.components.v1 as components

# Operating-range constants from the dataset
RPM_MIN, RPM_MAX = 1168, 2886


def _rpm_to_duration(rpm: float) -> float:
    """Map RPM to spin animation duration in seconds (1168 -> 2.0s, 2886 -> 0.35s)."""
    t = max(0.0, min(1.0, (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN)))
    return 2.0 - t * 1.65


def render_motor(rpm: float, torque: float, p_failure: float,
                 height: int = 320) -> None:
    """Render the animated motor.

    Args:
        rpm: current rotational speed (drives spin animation duration)
        torque: current torque (cosmetic — drawn as a label)
        p_failure: 0..1 probability driving vibration and warning glow
    """
    spin_duration = _rpm_to_duration(rpm)
    risk = max(0.0, min(1.0, p_failure))

    # Vibration intensity scales with risk; below 0.25 it's still
    if risk < 0.25:
        vib_class = "v-none"
    elif risk < 0.60:
        vib_class = "v-mild"
    else:
        vib_class = "v-strong"

    # Glow color: green -> amber -> red
    if risk < 0.25:
        glow = "0 0 30px rgba(46, 125, 50, 0.18)"
        status_color = "#2E7D32"
    elif risk < 0.60:
        glow = "0 0 40px rgba(255, 152, 0, 0.30)"
        status_color = "#F57C00"
    else:
        glow = "0 0 60px rgba(198, 40, 40, 0.55)"
        status_color = "#C62828"

    html = f"""<!doctype html>
<html><head>
<style>
  html, body {{ margin:0; padding:0; background:transparent; height:100%; overflow:hidden; }}
  .stage {{
    width:100%; height:100%;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 100%);
  }}
  .motor-wrap {{
    transition: filter 0.3s ease;
    filter: drop-shadow({glow});
  }}
  .motor-body.v-none   {{ animation: none; }}
  .motor-body.v-mild   {{ animation: shake 0.12s infinite; }}
  .motor-body.v-strong {{ animation: shake 0.06s infinite; }}
  @keyframes shake {{
    0%, 100% {{ transform: translate(0, 0); }}
    25%      {{ transform: translate(-1.2px, 0.6px); }}
    50%      {{ transform: translate(1.0px, -0.8px); }}
    75%      {{ transform: translate(-0.6px, 1.0px); }}
  }}
  .shaft   {{
    transform-origin: 360px 230px;
    animation: spin {spin_duration:.2f}s linear infinite;
  }}
  .fan     {{
    transform-origin: 540px 230px;
    animation: spin {spin_duration:.2f}s linear infinite;
  }}
  @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
  .warn-halo {{
    opacity: {0 if risk < 0.60 else 1};
    transition: opacity 0.3s ease;
    animation: pulse 1.0s ease-in-out infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: {1 if risk >= 0.60 else 0}; transform: scale(1); }}
    50%      {{ opacity: {0.35 if risk >= 0.60 else 0}; transform: scale(1.04); }}
  }}
  .badge {{
    fill: {status_color}; opacity: 0.92;
  }}
</style></head>
<body>
<div class="stage">
<svg viewBox="0 0 720 360" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:100%;max-height:300px;">

  <defs>
    <linearGradient id="bodyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5b6b87"/>
      <stop offset="50%" stop-color="#3d4a63"/>
      <stop offset="100%" stop-color="#28334a"/>
    </linearGradient>
    <linearGradient id="capGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7b8aa6"/>
      <stop offset="100%" stop-color="#3d4a63"/>
    </linearGradient>
    <radialGradient id="shaftGrad" cx="0.3" cy="0.3" r="0.7">
      <stop offset="0%" stop-color="#e9eef7"/>
      <stop offset="100%" stop-color="#7a8298"/>
    </radialGradient>
    <linearGradient id="footGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3d4a63"/>
      <stop offset="100%" stop-color="#1f2a3d"/>
    </linearGradient>
    <radialGradient id="hubGrad" cx="0.4" cy="0.4" r="0.6">
      <stop offset="0%" stop-color="#9fb0cc"/>
      <stop offset="100%" stop-color="#3a4660"/>
    </radialGradient>
  </defs>

  <!-- Floor shadow -->
  <ellipse cx="380" cy="338" rx="280" ry="10" fill="rgba(40,55,90,0.18)"/>

  <!-- Warning halo behind the motor (pulses when risk >= 0.6) -->
  <circle class="warn-halo" cx="380" cy="220" r="160"
          fill="none" stroke="#C62828" stroke-width="8" opacity="0"/>

  <g class="motor-wrap">
    <g class="motor-body {vib_class}">

      <!-- Mounting feet -->
      <rect x="170" y="300" width="80" height="22" rx="3" fill="url(#footGrad)"/>
      <rect x="510" y="300" width="80" height="22" rx="3" fill="url(#footGrad)"/>
      <circle cx="190" cy="318" r="3" fill="#0d1422"/>
      <circle cx="230" cy="318" r="3" fill="#0d1422"/>
      <circle cx="530" cy="318" r="3" fill="#0d1422"/>
      <circle cx="570" cy="318" r="3" fill="#0d1422"/>

      <!-- Main cylindrical body -->
      <rect x="180" y="150" width="380" height="155" rx="14"
            fill="url(#bodyGrad)" stroke="#1f2a3d" stroke-width="1.5"/>

      <!-- Cooling fins -->
      <g fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.18)" stroke-width="0.6">
        <rect x="210" y="160" width="2" height="135"/>
        <rect x="240" y="160" width="2" height="135"/>
        <rect x="270" y="160" width="2" height="135"/>
        <rect x="300" y="160" width="2" height="135"/>
        <rect x="330" y="160" width="2" height="135"/>
        <rect x="430" y="160" width="2" height="135"/>
        <rect x="460" y="160" width="2" height="135"/>
        <rect x="490" y="160" width="2" height="135"/>
        <rect x="520" y="160" width="2" height="135"/>
      </g>

      <!-- Top junction box -->
      <rect x="350" y="125" width="60" height="32" rx="3"
            fill="url(#capGrad)" stroke="#1f2a3d" stroke-width="1"/>
      <circle cx="380" cy="141" r="4" fill="#28334a" stroke="#0d1422" stroke-width="0.5"/>

      <!-- End caps -->
      <ellipse cx="180" cy="227" rx="10" ry="78" fill="url(#capGrad)"/>
      <ellipse cx="560" cy="227" rx="10" ry="78" fill="url(#capGrad)"/>

      <!-- Brand plate / label -->
      <rect x="370" y="225" width="120" height="44" rx="3"
            fill="#1a2238" stroke="#0d1422" stroke-width="0.5"/>
      <text x="430" y="245" text-anchor="middle"
            font-family="Segoe UI, Arial" font-size="13" font-weight="700"
            fill="#9fb0cc" letter-spacing="2">AI4I-MOTOR</text>
      <text x="430" y="262" text-anchor="middle"
            font-family="Segoe UI, Arial" font-size="10"
            fill="#6c7a91">ID 2020 · 3 PHASE</text>

      <!-- Hub at shaft side -->
      <circle cx="360" cy="230" r="35" fill="url(#hubGrad)"
              stroke="#0d1422" stroke-width="1"/>
      <!-- Spinning shaft (cross + bolts inside hub) -->
      <g class="shaft">
        <rect x="354" y="200" width="12" height="60" rx="2" fill="url(#shaftGrad)"/>
        <rect x="330" y="224" width="60" height="12" rx="2" fill="url(#shaftGrad)"/>
        <circle cx="360" cy="230" r="6" fill="#28334a" stroke="#0d1422" stroke-width="0.5"/>
      </g>

      <!-- Output shaft and cooling fan on the right end -->
      <rect x="560" y="222" width="40" height="16" rx="3" fill="url(#shaftGrad)"/>
      <circle cx="540" cy="230" r="30" fill="#1f2a3d"/>
      <g class="fan">
        <!-- 6 fan blades -->
        <ellipse cx="540" cy="205" rx="6" ry="20" fill="#7b8aa6" opacity="0.9"/>
        <ellipse cx="540" cy="255" rx="6" ry="20" fill="#7b8aa6" opacity="0.9"/>
        <ellipse cx="515" cy="230" rx="20" ry="6" fill="#7b8aa6" opacity="0.9"/>
        <ellipse cx="565" cy="230" rx="20" ry="6" fill="#7b8aa6" opacity="0.9"/>
        <ellipse cx="523" cy="213" rx="6" ry="20" fill="#7b8aa6" opacity="0.6"
                 transform="rotate(45 540 230)"/>
        <ellipse cx="557" cy="247" rx="6" ry="20" fill="#7b8aa6" opacity="0.6"
                 transform="rotate(45 540 230)"/>
      </g>
      <circle cx="540" cy="230" r="5" fill="#0d1422"/>

      <!-- Status indicator LEDs on the body -->
      <circle cx="220" cy="180" r="5" class="badge"/>
      <circle cx="240" cy="180" r="5" fill="#28334a" stroke="#0d1422" stroke-width="0.5"/>
      <circle cx="260" cy="180" r="5" fill="#28334a" stroke="#0d1422" stroke-width="0.5"/>
      <text x="220" y="200" text-anchor="middle"
            font-family="Segoe UI, Arial" font-size="9"
            fill="rgba(255,255,255,0.6)" letter-spacing="1">PWR</text>
    </g>
  </g>

  <!-- Read-out tags at corners -->
  <g font-family="Segoe UI, Arial">
    <rect x="20" y="22" width="170" height="48" rx="6"
          fill="#ffffff" stroke="#d6dff0" stroke-width="1"/>
    <text x="35" y="42" font-size="11" fill="#6c7a91" letter-spacing="2">RPM</text>
    <text x="35" y="62" font-size="22" font-weight="800" fill="#102A55">{rpm:.0f}</text>

    <rect x="530" y="22" width="170" height="48" rx="6"
          fill="#ffffff" stroke="#d6dff0" stroke-width="1"/>
    <text x="545" y="42" font-size="11" fill="#6c7a91" letter-spacing="2">TORQUE (Nm)</text>
    <text x="545" y="62" font-size="22" font-weight="800" fill="#102A55">{torque:.1f}</text>
  </g>
</svg>
</div>
</body></html>"""
    components.html(html, height=height, scrolling=False)
