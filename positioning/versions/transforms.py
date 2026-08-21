"""Upstream normalization transforms (angle 1: adaptive normalization / stationarity).

Applied to a source series BEFORE a version builds its signal, selected by the shared `norm_mode`
param. All causal, no look-ahead.

  frac_diff_ffd / min_ffd_d   Fixed-width-window fractional differentiation (Lopez de Prado, AFML
                              ch.5). Makes a near-integrated level stationary while retaining
                              long memory (minimal d where ADF just clears 95%). This is the fix
                              for "the 3y-z normalizes away the trend that IS the signal."
  ewma_vol_scale              Divide by trailing EWMA volatility -> constant-risk signal so a
                              "2 sigma crowd" means the same in 2008/2020/2022 (RiskMetrics).
  robust_mad_z                (x - rolling median)/(1.4826*rolling MAD) -> robust z; CFTC spikes
                              don't contaminate the center/scale (MAD breakdown point 50%).
  apply_norm                  Dispatch on norm_mode in {raw, fracdiff, volscale, robust}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def ffd_weights(d: float, thresh: float = 1e-5, max_size: int | None = None) -> np.ndarray:
    """Binomial FFD weights (newest-first index 0), truncated where |w| < thresh."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
        if max_size is not None and k >= max_size:
            break
    return np.array(w[::-1])  # oldest..newest for a trailing dot product


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-5) -> pd.Series:
    """Fixed-width fractional differentiation (causal). width = len(weights)-1 lags."""
    s = series.astype(float)
    w = ffd_weights(d, thresh)
    width = len(w) - 1
    vals = s.to_numpy()
    out = np.full(len(s), np.nan)
    for i in range(width, len(s)):
        window = vals[i - width: i + 1]
        if not np.isnan(window).any():
            out[i] = float(np.dot(w, window))
    return pd.Series(out, index=s.index, name=f"ffd_d{d:.2f}")


def min_ffd_d(series: pd.Series, d_grid=None, thresh: float = 1e-5,
              pval: float = 0.05) -> dict:
    """Sweep d and return the MINIMAL d whose FFD series passes ADF at `pval` (keeps most memory).
    Returns {d, adf_p, corr (FFD vs raw), series}. Falls back to d=1.0 if none pass."""
    from statsmodels.tsa.stattools import adfuller
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    raw = series.astype(float)
    best = None
    for d in d_grid:
        fd = frac_diff_ffd(raw, float(d), thresh).dropna()
        if len(fd) < 30:
            continue
        try:
            p = float(adfuller(fd.values, maxlag=1, regression="c", autolag=None)[1])
        except Exception:
            continue
        aligned = pd.concat([raw.rename("r"), fd.rename("f")], axis=1).dropna()
        corr = float(aligned["r"].corr(aligned["f"])) if len(aligned) > 2 else np.nan
        rec = {"d": float(d), "adf_p": p, "corr": corr, "series": frac_diff_ffd(raw, float(d), thresh)}
        if p < pval:
            return rec
        best = rec  # remember the last (largest-d) attempt as fallback
    return best or {"d": 1.0, "adf_p": np.nan, "corr": np.nan,
                    "series": raw.diff().rename("ffd_d1.00")}


def ewma_vol_scale(series: pd.Series, halflife: int = 26, use_changes: bool = True) -> pd.Series:
    """Constant-risk scaling: series divided by trailing EWMA volatility of its (changes)."""
    s = series.astype(float)
    base = s.diff() if use_changes else s
    vol = base.ewm(halflife=halflife, min_periods=halflife).std(bias=False)
    return (s / vol.replace(0.0, np.nan)).rename("volscale")


def robust_mad_z(series: pd.Series, window: int = 52) -> pd.Series:
    """Robust rolling z: (x - median)/(1.4826*MAD). Contamination-proof center/scale."""
    s = series.astype(float)
    mp = max(window // 2, 10)
    med = s.rolling(window, min_periods=mp).median()
    mad = (s - med).abs().rolling(window, min_periods=mp).median()
    return ((s - med) / (1.4826 * mad).replace(0.0, np.nan)).rename("robust_z")


def apply_norm(series: pd.Series, norm_mode: str = "raw", **kw) -> pd.Series:
    """Dispatch the shared norm_mode param onto a source series."""
    if norm_mode == "raw":
        return series
    if norm_mode == "fracdiff":
        return min_ffd_d(series)["series"]
    if norm_mode == "volscale":
        return ewma_vol_scale(series, halflife=int(kw.get("halflife", 26)))
    if norm_mode == "robust":
        return robust_mad_z(series, window=int(kw.get("window", 52)))
    raise ValueError(f"unknown norm_mode {norm_mode!r}")


if __name__ == "__main__":  # smoke test
    from statsmodels.tsa.stattools import adfuller
    idx = pd.date_range("2018-01-07", periods=422, freq="W")
    t = np.arange(422.0)
    lvl = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 422)) + 0.05 * t, index=idx)  # near-integrated
    res = min_ffd_d(lvl)
    p_raw = adfuller(lvl.values, maxlag=1, autolag=None)[1]
    print(f"raw ADF p={p_raw:.3f} (non-stationary)  ->  min-d={res['d']} ADF p={res['adf_p']:.3f} "
          f"corr={res['corr']:.2f}")
    vs = ewma_vol_scale(lvl); rz = robust_mad_z(lvl)
    print(f"volscale finite={vs.notna().sum()}  robust_z finite={rz.notna().sum()}")
    print("transforms.py OK" if res["d"] <= 1.0 and (np.isnan(res['adf_p']) or res['adf_p'] < 0.1) else "transforms.py CHECK")
