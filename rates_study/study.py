"""Positioning <-> yields relationship study (rigorous, reproducible).

Answers "does crowded positioning actually precede higher yields, historically?" with:
  * lead-lag correlation (does positioning lead or coincide with yield moves?)
  * conditional forward outcomes by crowding regime, with block-bootstrap CIs and n
  * sub-period robustness
  * Granger causality (positioning -> yield changes) and Newey-West forward-return regressions
  * the long/short asymmetry, made explicit

Everything uses the publication-lagged (causal) composite so nothing peeks ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

K_EXTREME = 1.5
HORIZONS = (4, 8, 13, 26)
SUBPERIODS = (("2016-2019", "2016-01-01", "2019-12-31"),
              ("2020-2022", "2020-01-01", "2022-12-31"),
              ("2023-2026", "2023-01-01", "2026-12-31"))
_RNG = np.random.default_rng(20260723)   # fixed seed -> reproducible CIs


def _aligned(composite: pd.Series, y10_w: pd.Series) -> pd.DataFrame:
    return pd.concat([composite.rename("z"), y10_w.rename("y")], axis=1).dropna()


def lead_lag_correlation(composite: pd.Series, y10_w: pd.Series,
                         max_h: int = 26) -> pd.DataFrame:
    """corr(composite(t), y(t+h) - y(t)) for h in [-max_h, max_h].
    h>0 = forward yield change (positioning leading); h<0 = past change (lagging)."""
    d = _aligned(composite, y10_w)
    rows = []
    for h in range(-max_h, max_h + 1):
        if h == 0:
            continue
        dy = d["y"].shift(-h) - d["y"]
        c = d["z"].corr(dy)
        rows.append({"h_weeks": h, "corr": float(c) if pd.notna(c) else np.nan})
    return pd.DataFrame(rows)


def _block_boot_ci(x: np.ndarray, stat, block: int, n_boot: int = 2000, alpha: float = 0.05):
    """Moving-block bootstrap CI for a statistic of an autocorrelated series."""
    x = np.asarray(x, float)
    n = len(x)
    if n < max(block, 5):
        return (np.nan, np.nan)
    nblocks = int(np.ceil(n / block))
    starts_max = n - block
    ests = np.empty(n_boot)
    for b in range(n_boot):
        starts = _RNG.integers(0, starts_max + 1, size=nblocks)
        samp = np.concatenate([x[s:s + block] for s in starts])[:n]
        ests[b] = stat(samp)
    lo, hi = np.nanpercentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def conditional_outcomes(composite: pd.Series, y10_w: pd.Series,
                         horizons=HORIZONS, k: float = K_EXTREME,
                         with_ci: bool = True) -> dict:
    """For each horizon and crowding regime: P(direction), mean forward Δy (bp), n, and
    block-bootstrap 95% CIs. Long regime -> P(selloff)=P(Δy>0); short regime -> P(rally)=P(Δy<0)."""
    d = _aligned(composite, y10_w)
    out = {}
    for H in horizons:
        fwd = (d["y"].shift(-H) - d["y"]).dropna() * 100.0   # bp
        z = d["z"].reindex(fwd.index)
        lo, sh, mid = fwd[z > k], fwd[z < -k], fwd[z.abs() <= k]
        rec = {
            "crowded_long": {
                "n": int(lo.size),
                "p_selloff": float((lo > 0).mean()) if lo.size else None,
                "mean_bp": float(lo.mean()) if lo.size else None},
            "crowded_short": {
                "n": int(sh.size),
                "p_rally": float((sh < 0).mean()) if sh.size else None,
                "mean_bp": float(sh.mean()) if sh.size else None},
            "mid_mean_bp": float(mid.mean()) if mid.size else None,
            "uncond_p_up": float((fwd > 0).mean()) if fwd.size else None,
            "uncond_mean_bp": float(fwd.mean()) if fwd.size else None,
        }
        if with_ci:
            blk = max(H, 4)
            if lo.size:
                rec["crowded_long"]["p_selloff_ci"] = _block_boot_ci(
                    (lo > 0).values.astype(float), np.mean, blk)
                rec["crowded_long"]["mean_bp_ci"] = _block_boot_ci(lo.values, np.mean, blk)
            if sh.size:
                rec["crowded_short"]["p_rally_ci"] = _block_boot_ci(
                    (sh < 0).values.astype(float), np.mean, blk)
                rec["crowded_short"]["mean_bp_ci"] = _block_boot_ci(sh.values, np.mean, blk)
        out[str(H)] = rec
    return out


def subperiod_outcomes(composite: pd.Series, y10_w: pd.Series,
                       horizons=(8, 13), k: float = K_EXTREME) -> dict:
    """Conditional crowded-long P(selloff) + mean bp per sub-period (robustness)."""
    out = {}
    for name, a, b in SUBPERIODS:
        c = composite.loc[a:b]
        if c.dropna().empty:
            continue
        out[name] = conditional_outcomes(c, y10_w.loc[a:b], horizons=horizons,
                                         k=k, with_ci=False)
    return out


def granger_and_regression(composite: pd.Series, y10_w: pd.Series,
                           horizons=HORIZONS, k: float = K_EXTREME) -> dict:
    """Granger causality (composite -> weekly Δy) + Newey-West forward-return regressions
    Δy(t->t+H) ~ composite(t). HAC lags = H (overlap-robust)."""
    d = _aligned(composite, y10_w)
    dy1 = (d["y"].diff()).dropna()                    # weekly Δy
    z = d["z"].reindex(dy1.index)
    gframe = pd.concat([dy1.rename("dy"), z.rename("z")], axis=1).dropna()
    granger = {}
    try:
        # test whether z Granger-causes dy (statsmodels tests col0 caused by col1)
        gc = grangercausalitytests(gframe[["dy", "z"]], maxlag=4, verbose=False)
        granger = {str(l): float(gc[l][0]["ssr_ftest"][1]) for l in gc}  # p-values by lag
    except Exception as e:  # pragma: no cover
        granger = {"error": str(e)}

    reg = {}
    for H in horizons:
        fwd = (d["y"].shift(-H) - d["y"]) * 100.0
        df = pd.concat([fwd.rename("fwd"), d["z"].rename("z")], axis=1).dropna()
        if len(df) < 30:
            continue
        X = sm.add_constant(df["z"])
        m = sm.OLS(df["fwd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": H})
        reg[str(H)] = {"beta_bp_per_z": float(m.params["z"]),
                       "t_stat": float(m.tvalues["z"]),
                       "p_value": float(m.pvalues["z"]),
                       "r2": float(m.rsquared), "n": int(m.nobs)}
    return {"granger_p_by_lag": granger, "forward_regression": reg}


def extreme_episodes(composite: pd.Series, k: float = K_EXTREME) -> list:
    """Contiguous spans where |composite| > k (crowding episodes), for the history table."""
    c = composite.dropna()
    state = (c > k).astype(int) - (c < -k).astype(int)
    eps, cur = [], None
    for t, s in state.items():
        if s != 0 and (cur is None or cur["sign"] != s):
            if cur:
                eps.append(cur)
            cur = {"sign": int(s), "start": t, "end": t, "peak": float(c[t])}
        elif s != 0:
            cur["end"] = t
            cur["peak"] = max(cur["peak"], float(c[t])) if s > 0 else min(cur["peak"], float(c[t]))
        elif s == 0 and cur:
            eps.append(cur); cur = None
    if cur:
        eps.append(cur)
    return [{"side": "long" if e["sign"] > 0 else "short",
             "start": str(e["start"].date()), "end": str(e["end"].date()),
             "weeks": int((e["end"] - e["start"]).days / 7) + 1,
             "peak_z": round(e["peak"], 2)} for e in eps]


def run_study(sd) -> dict:
    """Full study dict from a StudyData bundle (uses the causal/publication composite)."""
    comp, y = sd.composite_pub, sd.y10_w_pub
    return {
        "sample": {"start": str(comp.index.min().date()), "end": str(comp.index.max().date()),
                   "n_weeks": int(comp.dropna().size), "k_extreme": K_EXTREME},
        "lead_lag": lead_lag_correlation(comp, y).to_dict(orient="list"),
        "conditional": conditional_outcomes(comp, y),
        "subperiods": subperiod_outcomes(comp, y),
        "stats": granger_and_regression(comp, y),
        "episodes": extreme_episodes(sd.composite),   # display uses report-dated
        "latest": {"composite_z": float(sd.composite.iloc[-1]),
                   "as_of": str(sd.composite.index.max().date())},
    }
