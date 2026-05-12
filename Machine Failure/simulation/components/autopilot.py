"""Auto-pilot: simulate a 30-second machine shift.

The shift profile takes the machine through four phases:
  0–6s    idle    — low RPM, low torque
  6–14s   ramp-up — RPM and torque climb together
  14–22s  stress  — torque pushes high while RPM stays mid (failure-zone)
  22–30s  cool-down — torque drops, RPM normalises

Each tick (1 per real-second) updates the operating point in session state and
the page reruns. The trail of recent points is kept so the boundary plot can
draw the path.
"""
from __future__ import annotations

import numpy as np

PROFILE_DURATION_S = 30
TICK_S = 1.0          # one operating-point update per real second
TRAIL_LIMIT = PROFILE_DURATION_S  # keep the whole shift in the trail


def profile_at(t: float) -> tuple[float, float]:
    """Return (rpm, torque) for time t in seconds along the shift."""
    t = max(0.0, min(PROFILE_DURATION_S, t))
    if t < 6:
        # Idle
        rpm = 1450 + 30 * np.sin(t * 1.5)
        torque = 15 + 1.5 * np.sin(t)
    elif t < 14:
        # Ramp up
        a = (t - 6) / 8
        rpm = 1450 + a * (2000 - 1450) + 25 * np.sin(t * 1.2)
        torque = 15 + a * (40 - 15) + 1.5 * np.sin(t * 0.9)
    elif t < 22:
        # Stress test: torque spikes high, RPM dips (failure region)
        a = (t - 14) / 8
        # Torque climbs through the danger zone
        torque = 40 + a * (68 - 40) + 1.8 * np.sin(t * 1.4)
        # RPM dips slightly (motor labouring under load)
        rpm = 2000 - a * (2000 - 1350) + 30 * np.sin(t * 1.1)
    else:
        # Cool-down
        a = (t - 22) / 8
        torque = 68 - a * (68 - 25) + 2.0 * np.sin(t * 1.1)
        rpm = 1350 + a * (1500 - 1350) + 35 * np.sin(t * 1.4)

    rpm = float(np.clip(rpm, 1168, 2886))
    torque = float(np.clip(torque, 3.8, 76.6))
    return rpm, torque


def phase_for(t: float) -> str:
    if t < 6:
        return "Idle"
    if t < 14:
        return "Ramp-up"
    if t < 22:
        return "Stress test"
    return "Cool-down"
