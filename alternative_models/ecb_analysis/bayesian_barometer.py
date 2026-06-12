"""
Gaussian conjugate Bayesian tone updater + CUSUM change-point detection.

Prior is built from historical meeting mean/variance.
Each new speech segment provides one likelihood observation.
Posterior mean maps to the barometer [0, 100].

CUSUM flags structural breaks (tone regime shifts).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Prior: derived from MD file data
# GC tone meter = +2.22 (scale -7.5 to +7.5) → barometer ≈ 64.8
# EB tone meter = +1.99 → barometer ≈ 63.3
# Historical meeting variance: broad spread across dovish/neutral/hawkish
# ---------------------------------------------------------------------------
PRIOR_MU = 57.0        # prior mean barometer (slightly above neutral; matches MD +2.22 level)
PRIOR_SIGMA2 = 180.0   # prior variance (≈ std of 13.4 barometer units = medium uncertainty)
OBS_SIGMA2 = 120.0     # observation noise (within-meeting variability)

# CUSUM thresholds
CUSUM_THRESHOLD = 15.0     # barometer units; alert when cumulative sum exceeds this
CUSUM_SLACK = 3.0          # allowance (k parameter) for CUSUM


@dataclass
class BayesianBarometerState:
    """
    Mutable state for the live Bayesian barometer.

    mu_n, sigma2_n: current posterior mean and variance
    observations: list of (segment_index, barometer_value) tuples
    cusum_pos / cusum_neg: CUSUM statistics (positive/negative drift)
    alerts: list of detected change-point alerts
    """
    mu_n: float = PRIOR_MU
    sigma2_n: float = PRIOR_SIGMA2
    obs_sigma2: float = OBS_SIGMA2
    observations: list[tuple[int, float]] = field(default_factory=list)
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    cusum_history: list[tuple[int, float, float]] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    _running_mean: float = 0.0
    _n: int = 0

    @property
    def posterior_mean(self) -> float:
        return self.mu_n

    @property
    def posterior_std(self) -> float:
        return math.sqrt(self.sigma2_n)

    @property
    def credible_interval_95(self) -> tuple[float, float]:
        z = 1.96
        lo = max(0.0, self.mu_n - z * self.posterior_std)
        hi = min(100.0, self.mu_n + z * self.posterior_std)
        return (round(lo, 1), round(hi, 1))

    def update(self, segment_index: int, barometer_value: float) -> dict:
        """
        Bayesian conjugate update (known variance, Gaussian likelihood).

        Posterior precision = prior precision + n/σ²_obs
        Posterior mean = weighted average of prior and data mean
        """
        x = float(barometer_value)
        self.observations.append((segment_index, x))
        self._n += 1
        self._running_mean += (x - self._running_mean) / self._n

        # Conjugate update
        prior_prec = 1.0 / self.sigma2_n
        obs_prec = 1.0 / self.obs_sigma2
        post_prec = prior_prec + obs_prec
        self.mu_n = (prior_prec * self.mu_n + obs_prec * x) / post_prec
        self.sigma2_n = 1.0 / post_prec

        # CUSUM update
        alert = self._update_cusum(segment_index, x)

        return {
            "segment_index": segment_index,
            "observation": round(x, 2),
            "posterior_mean": round(self.mu_n, 2),
            "posterior_std": round(self.posterior_std, 2),
            "ci_95": self.credible_interval_95,
            "cusum_pos": round(self.cusum_pos, 2),
            "cusum_neg": round(self.cusum_neg, 2),
            "alert": alert,
        }

    def _update_cusum(self, idx: int, x: float) -> Optional[dict]:
        """
        Page's CUSUM for detecting upward and downward mean shifts.
        k = CUSUM_SLACK (allowance), h = CUSUM_THRESHOLD (decision boundary).
        """
        k = CUSUM_SLACK
        h = CUSUM_THRESHOLD

        self.cusum_pos = max(0.0, self.cusum_pos + (x - self.mu_n) - k)
        self.cusum_neg = max(0.0, self.cusum_neg - (x - self.mu_n) - k)
        self.cusum_history.append((idx, self.cusum_pos, self.cusum_neg))

        alert = None
        if self.cusum_pos > h:
            alert = {
                "segment_index": idx,
                "direction": "more hawkish",
                "cusum": round(self.cusum_pos, 2),
                "message": f"Tone shift detected: MORE HAWKISH at segment {idx}",
            }
            self.alerts.append(alert)
            self.cusum_pos = 0.0  # reset after detection

        elif self.cusum_neg > h:
            alert = {
                "segment_index": idx,
                "direction": "more dovish",
                "cusum": round(self.cusum_neg, 2),
                "message": f"Tone shift detected: MORE DOVISH at segment {idx}",
            }
            self.alerts.append(alert)
            self.cusum_neg = 0.0

        return alert

    def reset(self):
        self.__init__()


def build_barometer_from_history(historical_scores: list[float]) -> BayesianBarometerState:
    """
    Initialise a barometer pre-loaded with historical meeting scores.
    Returns a state that can then be updated live.
    """
    state = BayesianBarometerState()
    for i, score in enumerate(historical_scores):
        state.update(i, score)
    # Reset CUSUM and alerts after warm-up (historical is the prior)
    state.cusum_pos = 0.0
    state.cusum_neg = 0.0
    state.cusum_history.clear()
    state.alerts.clear()
    return state


def posterior_time_series(state: BayesianBarometerState) -> list[dict]:
    """
    Replay the update sequence to produce a time series of posterior means.
    Useful for plotting how the barometer evolved over the meeting.
    """
    replay_state = BayesianBarometerState()
    series = []
    for idx, x in state.observations:
        result = replay_state.update(idx, x)
        series.append(result)
    return series


def gauge_colour(barometer_value: float) -> str:
    """Colour for the Plotly gauge (blue → amber → red)."""
    if barometer_value >= 65:
        return "#c0392b"   # hawkish red
    elif barometer_value >= 50:
        return "#e67e22"   # amber
    elif barometer_value >= 35:
        return "#27ae60"   # balanced green
    else:
        return "#2980b9"   # dovish blue


def build_gauge_trace(barometer_value: float, ci: tuple[float, float]) -> dict:
    """
    Returns a Plotly go.Indicator dict for the barometer gauge.
    Call go.Figure(go.Indicator(**build_gauge_trace(...))).
    """
    import plotly.graph_objects as go
    lo, hi = ci
    return go.Indicator(
        mode="gauge+number+delta",
        value=round(barometer_value, 1),
        number={"font": {"size": 40, "color": gauge_colour(barometer_value)}},
        delta={"reference": 50.0, "increasing": {"color": "#c0392b"}, "decreasing": {"color": "#2980b9"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555"},
            "bar": {"color": gauge_colour(barometer_value), "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 35], "color": "rgba(41,128,185,0.15)"},
                {"range": [35, 65], "color": "rgba(230,126,34,0.10)"},
                {"range": [65, 100], "color": "rgba(192,57,43,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#FF8200", "width": 3},
                "thickness": 0.75,
                "value": barometer_value,
            },
        },
        title={
            "text": (
                "<b>Lagarde Hawkishness Barometer</b><br>"
                f"<span style='font-size:12px'>95% CI: [{lo}, {hi}]</span>"
            )
        },
        domain={"x": [0, 1], "y": [0, 1]},
    )
