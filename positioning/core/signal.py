"""Discrete contrarian stance from the continuous composite index.

Turns the composite crowding z-index into a {+1 / 0 / -1} stance with hysteresis, so
the call only flips at genuine positioning extremes and doesn't chatter around the band.
Contrarian sign convention: crowded net-long positioning (composite >> 0) implies a
SHORT-duration stance (-1); crowded shorts imply long (+1). Mirrors the band-crossing
state machine in the repo's src/curve_signals.generate_signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def contrarian_signal(composite: pd.Series, k_entry: float, k_exit: float) -> pd.Series:
    """Hysteresis band state machine on the composite z-index.

    Enter a contrarian stance when |z| > k_entry, hold until |z| < k_exit.
    Returns a Series of {+1 long-duration, 0 flat, -1 short-duration}.
    """
    state = 0
    out = []
    for z in composite.to_numpy():
        if np.isnan(z):
            out.append(state)
            continue
        if state == 0:
            if z > k_entry:
                state = -1   # crowded long -> fade -> short duration
            elif z < -k_entry:
                state = +1   # crowded short -> fade -> long duration
        elif state == -1:
            if z < k_exit:
                state = 0
        elif state == +1:
            if z > -k_exit:
                state = 0
        out.append(state)
    return pd.Series(out, index=composite.index, name="stance")
