"""V5 — Kalman local-linear-trend read of a positioning signal.

A *state-space* re-read of crowding. Instead of another rolling-window normalization,
V5 runs a local-linear-trend Kalman filter (statsmodels `UnobservedComponents`, the same
engine as the 10s5s RV model in `src.curve_signals`) over a chosen positioning SOURCE and
decomposes it into a dynamic fair value (`level`) and its `trend`. All three headline
outputs are strictly causal (the FILTERED, not smoothed, state — value at t uses only data
<= t):

  deviation     = (source - level) / rolling-σ(filter residual, dev_window)
                  POSITIVE = source sits ABOVE its dynamic fair value = crowding building.
  slope         = filter trend (nu_t)                POSITIVE = positioning rising.
  acceleration  = trend.diff()                        POSITIVE = the build-up is accelerating.

SOURCE (`params["source"]`): a version composite (v0..v4, each built with its own defaults)
or the raw buy-side net-DV01 `signal` (categories / signal_mode applied via `resolve_signal`).

SMOOTHNESS (`params["kalman_smooth"]`):
  auto        -> MLE-fit the three variances (fixed_params=None).
  responsive  -> fixed-variance preset with a LARGER sigma2.trend (looser, faster-turning trend).
  smooth      -> fixed-variance preset with a SMALLER sigma2.trend (stiffer, slower trend).
The presets scale every variance by the SOURCE's own variance V (see `_PRESETS`) so they are
data-agnostic: rescaling the source by c scales all variances by c² and leaves the filter's
behaviour (which depends only on the variance RATIOS / signal-to-noise) unchanged.

Sign convention matches base: POSITIVE = crowded net LONG duration. Unbounded (bounded=False).
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from positioning.versions import base
from positioning.versions import params as P
from positioning.versions.base import load_inputs
from src.curve_signals import kalman_local_linear_trend

# Which module builds each version source (called with its own defaults).
_SRC_MODULES = {
    "v0": "positioning.versions.v0_production",
    "v1": "positioning.versions.v1_kaiser_pca",
    "v2": "positioning.versions.v2_cftc_bestpractice",
    "v3": "positioning.versions.v3_ewma_52w",
    "v4": "positioning.versions.v4_curve",
}

# Fixed-variance presets as MULTIPLIERS of the source variance V (data-agnostic).
# irregular/level are held constant across presets; only sigma2.trend distinguishes them
# (larger trend variance -> the slope can move faster -> more "responsive").
_PRESETS = {
    "responsive": {"sigma2.irregular": 0.30, "sigma2.level": 0.02, "sigma2.trend": 0.05},
    "smooth":     {"sigma2.irregular": 0.30, "sigma2.level": 0.02, "sigma2.trend": 5e-4},
}


def _resolve_source(inp: base.VersionInputs, p: dict) -> pd.Series:
    """Return the series the Kalman filter runs on (a version composite, or raw signal)."""
    src = p["source"]
    if src in _SRC_MODULES:
        mod = importlib.import_module(_SRC_MODULES[src])
        return mod.build(inp).composite.rename(src)          # version defaults
    return P.resolve_signal(inp, p)                          # raw buy-side net-DV01


def _fixed_params(smooth: str, variance: float) -> dict | None:
    """Map the smoothness choice to statsmodels fixed params (None => MLE)."""
    if smooth == "auto":
        return None
    mult = _PRESETS[smooth]
    v = variance if (variance and np.isfinite(variance)) else 1.0
    return {k: m * v for k, m in mult.items()}


def build(inp: base.VersionInputs, params: dict | None = None) -> base.VersionResult:
    p = P.merge("v5", params)
    source = _resolve_source(inp, p).astype(float)
    src_clean = source.dropna()

    smooth = p["kalman_smooth"]
    dev_w = int(p["dev_window"])
    out_name = p["output"]

    variance = float(np.nanvar(src_clean.values, ddof=0))
    fixed = _fixed_params(smooth, variance)

    lt = kalman_local_linear_trend(src_clean, fixed_params=fixed)
    level = lt.level                                   # dynamic fair value (filtered mu_t)
    trend = lt.trend                                   # slope (filtered nu_t)
    resid = lt.resid                                   # source - level (causal filter residual)

    # deviation: filter residual scaled by its own trailing rolling std (crowding vs fair value).
    # The denominator is floored at 1e-6 * source scale: with a very smooth source the MLE can
    # collapse sigma2.irregular to ~0 (level tracks the source exactly), leaving a residual that
    # is pure floating-point noise; dividing that noise by its own ~0 rolling std would manufacture
    # spurious ±σ spikes. The floor is negligible for any well-behaved fit (residual σ >> it) so it
    # only tames the degenerate perfect-fit case, where it yields an honest ~0 deviation.
    src_scale = float(np.nanstd(src_clean.values)) or 1.0
    resid_std = resid.rolling(dev_w, min_periods=max(dev_w // 2, 10)).std(ddof=0)
    deviation = (resid / resid_std.clip(lower=1e-6 * src_scale)).rename("deviation")
    slope = trend.rename("slope")
    acceleration = trend.diff().rename("acceleration")

    outputs = {"deviation": deviation, "slope": slope, "acceleration": acceleration}
    composite = outputs[out_name].rename("v5_kalman")

    def _latest(s: pd.Series):
        s = s.dropna()
        return None if s.empty else float(s.iloc[-1])

    meta = {
        "source": p["source"],
        "output": out_name,
        "kalman_smooth": smooth,
        "deviation_latest": _latest(deviation),
        "slope_latest": _latest(slope),
        "acceleration_latest": _latest(acceleration),
        "level_latest": _latest(level),
        "innov_std": float(lt.innov_std),
        "kalman_params": {k: float(v) for k, v in lt.params.items()},
        "dev_window": dev_w,
        "source_variance": variance,
        "note": (f"Local-linear-trend Kalman on source={p['source']} ({smooth}); headline="
                 f"{out_name}. deviation>0 = above fair value (crowding building), "
                 f"slope>0 = positioning rising, acceleration>0 = accelerating."),
    }

    return base.VersionResult(
        name="V5 Kalman (deviation-σ / slope / acceleration)",
        description="Local-linear-trend Kalman filter (statsmodels UnobservedComponents) over a "
                    "chosen positioning source -> causal dynamic fair value + trend; headline is "
                    "deviation-from-fair-value (σ), filter slope, or its acceleration.",
        composite=composite,
        bounded=False,
        curve=None,
        meta=meta,
    )


if __name__ == "__main__":
    inp = load_inputs()
    res = build(inp)                                   # source=v0, output=deviation (defaults)
    print(res.name)
    print("description:", res.description)
    print("source=v0 latest outputs:")
    print("  deviation   :", res.meta["deviation_latest"])
    print("  slope       :", res.meta["slope_latest"])
    print("  acceleration:", res.meta["acceleration_latest"])
    print("  level       :", res.meta["level_latest"])
    print("  innov_std   :", res.meta["innov_std"])
    print("  kalman_params:", res.meta["kalman_params"])
