"""Regime-adaptive helpers for V10: structural-break monitoring + resampled robust deviation bands.

The idea (user): a break of structure is rare but, when it happens, the model should either retrain on
the new regime or — the headline — RESAMPLE the new regime's distribution to get ROBUST boundaries
(quantile bands with confidence intervals) instead of a fixed ±1.5σ. "Crowded/extreme" is then
calibrated to the current regime's actual (skewed, fat-tailed) distribution, with honest uncertainty.

  cusum_recursive_resid   CUSUM of standardized recursive residuals (Brown-Durbin-Evans stability
                          test) on a causal innovation series -> cumulative path + expanding boundary +
                          first-breach date. A break = the CUSUM path leaving the boundary.
  resampled_bands         From the trailing regime window of a deviation series, estimate each quantile
                          boundary (empirical / gaussian_kde / genpareto POT tail) and BOOTSTRAP it ->
                          {q: {point, lo, hi}} with CIs.
  rolling_resampled_bands Causal time series of those band point-estimates (for plotting + coverage).
  coverage_test           Breach rate of a series beyond each band vs the nominal rate (calibration).

Pure numpy/scipy (+ optional). Causal (trailing windows only). Reusable + independently testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_QUANTILES = (0.01, 0.05, 0.95, 0.99)
# Brown-Durbin-Evans CUSUM boundary coefficient a(alpha): line through (0, a√n)..(n, 3a√n).
_BDE_A = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}


# --------------------------------------------------------------------------- #
def cusum_recursive_resid(innov: pd.Series, alpha: float = 0.05) -> dict:
    """CUSUM of standardized recursive residuals + Brown-Durbin-Evans boundary.

    `innov` are the one-step (recursive) residuals — e.g. the DFM standardized innovations. The CUSUM
    path W_t = (1/s) Σ_{j<=t} w_j (s = residual std) should stay inside the straight-line boundary
    ±a·(√n + 2t/√n); leaving it signals parameter instability (a structural break). Causal.

    Returns {cusum, boundary_hi, boundary_lo, breach, first_breach, n, alpha}.
    """
    w = innov.dropna().astype(float)
    n = len(w)
    empty = {"cusum": pd.Series(dtype=float), "boundary_hi": pd.Series(dtype=float),
             "boundary_lo": pd.Series(dtype=float), "breach": pd.Series(dtype=bool),
             "first_breach": None, "n": int(n), "alpha": alpha}
    if n < 10:
        return empty
    s = float(w.std(ddof=1))
    if not np.isfinite(s) or s == 0:
        return empty
    cum = np.cumsum(w.values - w.values.mean()) / s
    a = _BDE_A.get(round(alpha, 2), 0.948)
    t = np.arange(1, n + 1, dtype=float)
    bnd = a * (np.sqrt(n) + 2.0 * t / np.sqrt(n))
    idx = w.index
    cusum = pd.Series(cum, index=idx, name="cusum")
    hi = pd.Series(bnd, index=idx, name="boundary_hi")
    lo = pd.Series(-bnd, index=idx, name="boundary_lo")
    breach = (cusum > hi) | (cusum < lo)
    fb = breach[breach].index.min() if breach.any() else None
    return {"cusum": cusum, "boundary_hi": hi, "boundary_lo": lo, "breach": breach,
            "first_breach": None if fb is None else str(fb.date()), "n": int(n), "alpha": alpha}


# --------------------------------------------------------------------------- #
def _pot_quantile(x: np.ndarray, q: float, thr_pct: float = 0.90, min_exc: int = 12) -> float:
    """Peaks-Over-Threshold quantile via a genpareto fit on the exceedances (fat-tail calibrated).
    Upper tail for q>=0.5, lower tail (mirrored) for q<0.5. Falls back to the empirical quantile
    when there are too few exceedances."""
    from scipy.stats import genpareto
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float(np.quantile(x, q)) if n else float("nan")
    upper = q >= 0.5
    y = x if upper else -x
    qq = q if upper else 1.0 - q                      # work in the upper tail of y
    u = np.quantile(y, thr_pct)
    exc = y[y > u] - u
    if len(exc) < min_exc:
        return float(np.quantile(x, q))
    try:
        c, _loc, scale = genpareto.fit(exc, floc=0.0)
        nu = len(exc)
        # POT quantile: u + (scale/c)[ ((n/nu)(1-qq))^{-c} - 1 ]  (c->0 limit = exponential)
        r = (n / nu) * (1.0 - qq)
        if abs(c) < 1e-6:
            val = u - scale * np.log(r)
        else:
            val = u + (scale / c) * (r ** (-c) - 1.0)
        return float(val if upper else -val)
    except Exception:
        return float(np.quantile(x, q))


def _quantile_point(x: np.ndarray, q: float, tail: str) -> float:
    """Point estimate of the q-quantile under the chosen tail model."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    extreme = (q <= 0.10) or (q >= 0.90)
    if tail == "gpd" and extreme:
        return _pot_quantile(x, q)
    if tail == "kde":
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(x)
            grid = np.linspace(x.min(), x.max(), 512)
            cdf = np.cumsum(kde(grid)); cdf /= cdf[-1]
            return float(np.interp(q, cdf, grid))
        except Exception:
            return float(np.quantile(x, q))
    return float(np.quantile(x, q))


def resampled_bands(dev: pd.Series, quantiles=DEFAULT_QUANTILES, window: int = 104,
                    tail: str = "gpd", n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap-CI'd quantile boundaries from the trailing `window` of `dev`.

    Returns {str(q): {'point','lo','hi'} , 'n': int, 'window': int, 'tail': str} where lo/hi are the
    2.5/97.5 percentiles of the bootstrap distribution of that boundary (uncertainty on the boundary
    itself — the 'robust expectation over the boundaries')."""
    x = dev.dropna().astype(float).values
    if window and len(x) > window:
        x = x[-window:]
    out: dict = {"n": int(len(x)), "window": int(window), "tail": tail}
    if len(x) < 12:
        for q in quantiles:
            out[str(q)] = {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return out
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(x), size=(n_boot, len(x)))    # one resample matrix, reused per q
    for q in quantiles:
        point = _quantile_point(x, q, tail)
        bq = np.array([_quantile_point(x[boot_idx[b]], q, tail) for b in range(n_boot)])
        bq = bq[np.isfinite(bq)]
        lo, hi = (float(np.percentile(bq, 2.5)), float(np.percentile(bq, 97.5))) if len(bq) else (np.nan, np.nan)
        out[str(q)] = {"point": float(point), "lo": lo, "hi": hi}
    return out


def rolling_resampled_bands(dev: pd.Series, quantiles=DEFAULT_QUANTILES, window: int = 104,
                            tail: str = "gpd", step: int = 1) -> pd.DataFrame:
    """Causal time series of the resampled quantile boundary POINT estimates (no bootstrap, for speed).
    Column per quantile ('q0.95'); value at date t uses only dev[:t]'s trailing `window`."""
    s = dev.dropna().astype(float)
    n = len(s)
    cols = {f"q{q}": np.full(n, np.nan) for q in quantiles}
    mp = max(window // 2, 26)
    for i in range(mp, n, max(1, step)):
        x = s.values[max(0, i - window + 1): i + 1]
        for q in quantiles:
            cols[f"q{q}"][i] = _quantile_point(x, q, tail)
    df = pd.DataFrame(cols, index=s.index)
    if step > 1:
        df = df.ffill()
    return df


def coverage_test(series: pd.Series, bands, nominal=None) -> dict:  # noqa: ARG001
    """Calibration: fraction of `series` beyond each band boundary vs the nominal exceedance rate.

    `bands` = a rolling_resampled_bands DataFrame (columns 'q<level>', aligned by date) or a static
    {str(q): {'point': ...}} dict. For q>=0.5 breach = series > boundary (nominal = 1-q); for q<0.5
    breach = series < boundary (nominal = q). Returns {str(q): {breach_rate, nominal, n, well_calibrated}}
    where well_calibrated = breach_rate within a 2x band of nominal."""
    s = series.dropna().astype(float)
    res: dict = {}
    if isinstance(bands, pd.DataFrame):
        for col in bands.columns:
            q = float(col[1:]) if col.startswith("q") else float(col)
            b = bands[col].reindex(s.index)
            al = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
            if al.empty:
                res[str(q)] = {"breach_rate": float("nan"), "nominal": (1 - q if q >= 0.5 else q),
                               "n": 0, "well_calibrated": False}
                continue
            breach = (al["s"] > al["b"]) if q >= 0.5 else (al["s"] < al["b"])
            nom = (1 - q) if q >= 0.5 else q
            br = float(breach.mean())
            res[str(q)] = {"breach_rate": br, "nominal": nom, "n": int(len(al)),
                           "well_calibrated": bool(nom / 2 <= br <= nom * 2)}
    else:  # static dict of point boundaries
        for k, v in bands.items():
            if not isinstance(v, dict) or "point" not in v:
                continue
            q = float(k)
            pt = v["point"]
            if not np.isfinite(pt):
                continue
            breach = (s > pt) if q >= 0.5 else (s < pt)
            nom = (1 - q) if q >= 0.5 else q
            br = float(breach.mean())
            res[str(q)] = {"breach_rate": br, "nominal": nom, "n": int(len(s)),
                           "well_calibrated": bool(nom / 2 <= br <= nom * 2)}
    return res


if __name__ == "__main__":  # smoke test
    rng = np.random.default_rng(0)
    # A t-distributed (fat-tailed) deviation with a mid-sample level break -> CUSUM should breach.
    dev = pd.Series(np.concatenate([rng.standard_t(4, 200), rng.standard_t(4, 200) + 3.0]),
                    index=pd.date_range("2019-01-06", periods=400, freq="W"))
    cu = cusum_recursive_resid(dev)
    print(f"CUSUM: n={cu['n']} first_breach={cu['first_breach']} (expect ~mid-sample after the +3 shift)")
    rb = resampled_bands(dev, window=200, tail="gpd", n_boot=400)
    for q in (0.01, 0.05, 0.95, 0.99):
        b = rb[str(q)]
        print(f"  q{q}: point={b['point']:+.2f}  CI=[{b['lo']:+.2f},{b['hi']:+.2f}]")
    roll = rolling_resampled_bands(dev, window=104)
    cov = coverage_test(dev, roll)
    print("coverage (breach vs nominal):", {q: (round(v['breach_rate'],3), v['nominal']) for q, v in cov.items()})
    ok = cu["first_breach"] is not None and np.isfinite(rb["0.95"]["point"]) and len(roll) == 400
    print("regime.py OK" if ok else "regime.py CHECK")
