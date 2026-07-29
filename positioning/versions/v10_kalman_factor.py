"""V10 Regime-adaptive Kalman dynamic-factor deviation — Kalman INSTEAD of PCA.

Replaces V0's `StandardScaler + PCA -> PC1 -> re-3y-z` with a Kalman DYNAMIC FACTOR over the same
7-KPI panel: the KPIs load on a latent factor evolving as an AR(1) state, and the Kalman-FILTERED
(causal) factor is the "far more informed average" — a dynamic PC1 with a proper covariance. Outputs:
  factor        filtered latent crowding state (sign-fixed + = crowded long) — a dynamic PC1;
  deviation     signed multivariate NIS (eᵀS⁻¹e) = deviation from the informed average, the headline;
  slope/accel   factor trend and its change (deepens V5).
Plus a BREAK-OF-STRUCTURE filter on the innovations (CUSUM recursive-residuals + V8 BOCPD) and, on a
break, RETRAIN (re-standardize the deviation to the new-regime location/scale) and/or RESAMPLE the
new-regime distribution (positioning.versions.regime) for robust quantile deviation bands with CIs.

Everything causal: the FILTERED (never smoothed) factor, trailing regime windows, expanding-safe
standardization. Reuses the V0 feature panel (positioning.core.score.run_score, read-only),
statsmodels DynamicFactor, v8_changepoint._bocpd, and regime.py.

Sign convention matches base: POSITIVE = crowded net LONG duration. Unbounded (bounded=False).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import params as P
from . import base
from . import regime
from . import v8_changepoint
from .base import VersionInputs, VersionResult

# BOCPD P(recent break) is trivially ~1 during the initial run-in (a run length cannot exceed t),
# the same guard V8 documents — exclude the first BURN_IN weeks from break DATING.
BURN_IN = 8


def get_feature_panel(inp: VersionInputs, panel: str = "feature_panel") -> pd.DataFrame:
    """The KPI observation matrix for the Kalman factor.

    panel='feature_panel' -> the exact 7 rolling-3y-z KPIs V0 feeds to PCA
    (positioning.core.score.run_score(...).feature_panel — read-only reuse of core; guarded, with a
    small cat_dv01-derived fallback if the import/build fails).
    panel='raw_level' -> per-category net-DV01 LEVELS + curve tilt, each rolling-z(52) so the columns
    are comparable (longer, near-integrated source).
    """
    if panel == "raw_level":
        cols: dict[str, pd.Series] = {}
        for c in inp.cat_dv01.columns:
            cols[f"{c}_level"] = base.rolling_z(inp.cat_dv01[c].astype(float), 52)
        cols["curve"] = base.rolling_z(inp.buyside_front_long().astype(float), 52)
        return pd.DataFrame(cols)

    # 'feature_panel' (default) — the V0 PCA input, reused read-only.
    try:
        from positioning.core import score as _score
        fp = _score.run_score(inp.fut_raw, inp.cash_raw, inp.yields_raw).feature_panel
        return fp.astype(float)
    except Exception:                                            # pragma: no cover - defensive
        cols = {}
        for c in inp.cat_dv01.columns:
            lvl = inp.cat_dv01[c].astype(float)
            cols[f"{c}_level_z"] = base.rolling_z(lvl, 156)
            cols[f"{c}_flow_z"] = base.rolling_z(lvl.diff(), 52)
        cols["curve_z"] = base.rolling_z(inp.buyside_front_long().astype(float), 156)
        return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
def _fit_dfm(X: pd.DataFrame, p: dict):
    """Fit the DynamicFactor and FILTER the full panel with the fitted params (always causal).

    auto: MLE on the train slice if it has >=30 rows, else full-sample MLE (the free cache's
    <=train_end slice is only ~4 rows). fixed OR MLE non-convergence/exception -> a fixed-variance
    parameter vector (start_params with modest idiosyncratic variances) filtered over the full panel,
    mirroring v5_kalman's fixed fallback. Returns (mod, res, info)."""
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    k, fo = int(p["k_factors"]), int(p["factor_order"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = DynamicFactor(X, k_factors=k, factor_order=fo, error_order=0)
        train = X.loc[:p["train_end"]]
        mode = p["fit_mode"]

        if mode == "auto":
            # (a) train-slice MLE when it is large enough (production leakage guard)
            if len(train) >= 30:
                try:
                    mtr = DynamicFactor(train, k_factors=k, factor_order=fo, error_order=0)
                    rtr = mtr.fit(disp=False, maxiter=300)
                    if bool(rtr.mle_retvals.get("converged", False)):
                        res = mod.filter(rtr.params)
                        return mod, res, {"dfm_converged": True, "fit_scope": "train",
                                          "fit_mode_used": "auto"}
                except Exception:
                    pass
            # (b) full-sample MLE
            try:
                rf = mod.fit(disp=False, maxiter=300)
                if bool(rf.mle_retvals.get("converged", False)):
                    return mod, rf, {"dfm_converged": True, "fit_scope": "full",
                                     "fit_mode_used": "auto"}
            except Exception:
                pass

        # (c) fixed-variance vector — explicit fit_mode='fixed' or a fallback on non-convergence
        vec = np.array(mod.start_params, dtype=float)
        for i, nm in enumerate(mod.param_names):
            if nm.startswith("sigma2."):
                vec[i] = 0.5                                     # modest idiosyncratic variance
        res = mod.filter(vec)
        return mod, res, {"dfm_converged": False, "fit_scope": "full", "fit_mode_used": "fixed"}


def _nis_and_projection(res, n_endog: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-t multivariate NIS = eᵀ S⁻¹ e and the loading-weighted innovation Λ·e (crowding
    direction). S is regularized with a small ridge then pseudo-inverted (singular-robust)."""
    e = np.asarray(res.filter_results.forecasts_error)          # (k_obs, T)
    S = np.asarray(res.filter_results.forecasts_error_cov)      # (k_obs, k_obs, T)
    design = np.asarray(res.filter_results.design)              # (k_obs, k_states, 1)
    load = design[:, 0, 0]                                      # loadings on the (first) factor
    T = e.shape[1]
    ridge = 1e-8 * np.eye(n_endog)
    nis = np.full(T, np.nan)
    proj = np.full(T, np.nan)
    for t in range(T):
        et = e[:, t]
        if not np.all(np.isfinite(et)):
            continue
        St = S[:, :, t]
        try:
            Sinv = np.linalg.pinv(St + ridge)
            nis[t] = float(et @ Sinv @ et)
        except Exception:
            nis[t] = np.nan
        proj[t] = float(load @ et)
    return nis, proj


def _latest(s: pd.Series):
    s = s.dropna()
    return None if s.empty else float(s.iloc[-1])


# --------------------------------------------------------------------------- #
def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v10", params)
    idx0 = inp.report_dates
    note_bits: list[str] = []

    try:
        X = get_feature_panel(inp, p["panel"]).dropna()
        if len(X) < 30:
            raise ValueError(f"panel too short ({len(X)} rows)")

        # 1-2) fit the dynamic factor and FILTER the full history (causal states)
        mod, res, fit_info = _fit_dfm(X, p)

        # 3) filtered factor -> sign-fixed (+ = crowded long) -> standardized crowding state.
        # Crowded-long positive per the framework convention (base.py: POSITIVE = crowded net long =
        # agrees with the live composite V0). The equal-weight mean of the *level_z columns is the
        # intended proxy, but on this sample lev_money & asset_mgr level_z anti-correlate (the factor
        # loads + on lev_money, - on asset_mgr), so that mean is a weak, sign-ambiguous anchor; we
        # therefore sign to agree with V0 when available (matching the framework), else the mean-level.
        fac = pd.Series(np.asarray(res.factors.filtered[0]).ravel(), index=X.index, name="factor")
        level_cols = [c for c in X.columns if c.endswith("level_z") or c.endswith("_level")]
        if not level_cols:
            level_cols = list(X.columns)
        mean_level = X[level_cols].mean(axis=1)
        v0_sig = inp.prod_composite.reindex(fac.index)
        c_v0 = fac.corr(v0_sig)
        c_lvl = fac.corr(mean_level)
        if np.isfinite(c_v0) and abs(c_v0) > 1e-9:
            sign = 1.0 if c_v0 > 0 else -1.0
        elif np.isfinite(c_lvl):
            sign = 1.0 if c_lvl > 0 else -1.0
        else:
            sign = 1.0
        fac = sign * fac
        fstd = float(fac.std(ddof=0)) or 1.0
        fac = ((fac - fac.mean()) / fstd).rename("factor")       # ~unit crowding state

        # 4) deviation = signed multivariate NIS (deviation from the informed average)
        nis_arr, proj_arr = _nis_and_projection(res, X.shape[1])
        nis = pd.Series(nis_arr, index=X.index, name="nis")
        proj = pd.Series(proj_arr, index=X.index)
        dev_sign = np.sign(sign * proj).replace(0.0, 1.0)        # point along the crowding direction
        dev_raw = (dev_sign * np.sqrt(nis.clip(lower=0.0))).rename("deviation")
        dstd = float(dev_raw.std(ddof=0)) or 1.0
        dev = (dev_raw / dstd).rename("deviation")               # ~unit σ signed deviation

        # 5) slope / acceleration (deepens V5) — EWMA-smoothed factor trend
        slope = fac.diff().ewm(halflife=8).mean().rename("slope")
        acceleration = slope.diff().rename("acceleration")

        # 6) break-of-structure filter on the standardized innovation series
        z_innov = ((dev - dev.mean()) / (float(dev.std(ddof=0)) or 1.0)).rename("z_innov")
        det = p["break_detector"]
        cusum: dict = {}
        break_prob = pd.Series(np.nan, index=z_innov.index, name="break_prob")
        if det in ("cusum", "both"):
            cusum = regime.cusum_recursive_resid(z_innov, alpha=0.05)
        if det in ("bocpd", "both"):
            bp = v8_changepoint._bocpd(z_innov.to_numpy(), float(p["bocpd_hazard"]), obs="t")
            break_prob = pd.Series(bp, index=z_innov.index, name="break_prob").clip(0.0, 1.0)

        # earliest break: min of the CUSUM first-breach and the first week BOCPD>0.5 (past the run-in)
        cand: list[pd.Timestamp] = []
        cfb = cusum.get("first_breach") if isinstance(cusum, dict) else None
        if cfb:
            cand.append(pd.Timestamp(cfb))
        bp_post = break_prob.iloc[BURN_IN:]
        bp_hi = bp_post[bp_post > 0.5]
        if len(bp_hi):
            cand.append(pd.Timestamp(bp_hi.index.min()))
        first_break = str(min(cand).date()) if cand else None
        n_breaks = int(((bp_post > 0.5).astype(int).diff() > 0).sum())
        if cfb:
            n_breaks = max(n_breaks, 1)

        # 7a) RETRAIN response — re-standardize the deviation to the NEW regime's location/scale
        retrained_at = None
        if p["break_response"] in ("retrain", "both") and first_break is not None:
            fb = pd.Timestamp(first_break)
            post = dev.loc[fb:]
            if len(post) >= 10:
                win = post.iloc[-int(p["regime_window"]):] if len(post) > int(p["regime_window"]) \
                    else post
                loc = float(win.median())
                sc = float(win.std(ddof=0)) or 1.0
                dev = dev.copy()
                dev.loc[fb:] = (dev.loc[fb:] - loc) / sc
                retrained_at = first_break
                note_bits.append(f"Deviation re-standardized to the post-{first_break} regime.")

        # 7b) RESAMPLE response (HEADLINE) — robust regime-resampled quantile bands + coverage
        bands = regime.resampled_bands(dev, window=int(p["regime_window"]), tail="gpd")
        rolling_bands = regime.rolling_resampled_bands(dev, window=int(p["regime_window"]))
        coverage = regime.coverage_test(dev, rolling_bands)

        # 8) select the headline composite
        outputs = {"deviation": dev, "factor": fac, "slope": slope, "acceleration": acceleration}
        composite = outputs[p["output"]].rename("v10")
        curve = fac.rename("v10_factor")

        v0 = inp.prod_composite.reindex(fac.index)
        corr_v0 = float(fac.corr(v0)) if v0.notna().sum() > 3 else float("nan")

        meta = {
            "panel": p["panel"],
            "dfm_converged": fit_info["dfm_converged"],
            "fit_mode_used": fit_info["fit_mode_used"],
            "fit_scope": fit_info["fit_scope"],
            "k_factors": int(p["k_factors"]),
            "factor_order": int(p["factor_order"]),
            "sign": float(sign),
            "corr_with_v0": corr_v0,
            "factor_latest": _latest(fac),
            "deviation_latest": _latest(dev),
            "nis": nis,
            "nis_latest": _latest(nis),
            "deviation_std": float(dev.std(ddof=0)),
            "break_detector": det,
            "break_response": p["break_response"],
            "first_break": first_break,
            "n_breaks": n_breaks,
            "break_prob": break_prob,
            "cusum": cusum,
            "resampled_bands": bands,
            "rolling_bands": rolling_bands,
            "coverage": coverage,
            "retrained_at": retrained_at,
            "n_obs": int(len(X)),
            "note": ("Kalman DynamicFactor over the KPI panel (Kalman instead of PCA): filtered factor "
                     "(sign-fixed +=crowded long) + signed multivariate-NIS deviation + "
                     f"{det} break filter + regime-resampled robust bands. " + " ".join(note_bits)
                     ).strip(),
        }
        return VersionResult(
            name="V10 Regime-adaptive Kalman dynamic-factor",
            description="DynamicFactor (Kalman) over the 7-KPI panel: filtered crowding factor + "
                        "signed multivariate-NIS deviation + CUSUM/BOCPD break filter + "
                        "regime-resampled robust deviation bands.",
            composite=composite,
            bounded=False,
            curve=curve,
            meta=meta,
        )
    except Exception as ex:                                      # never raise — degrade to NaN
        nan_s = pd.Series(np.nan, index=idx0, name="v10")
        return VersionResult(
            name="V10 Regime-adaptive Kalman dynamic-factor",
            description="DynamicFactor (Kalman) over the KPI panel — degraded to NaN.",
            composite=nan_s, bounded=False, curve=nan_s,
            meta={"dfm_converged": False, "fit_mode_used": None, "fit_scope": None,
                  "first_break": None, "break_prob": pd.Series(dtype=float), "cusum": {},
                  "resampled_bands": {}, "rolling_bands": pd.DataFrame(), "coverage": {},
                  "retrained_at": None,
                  "note": f"V10 build failed ({type(ex).__name__}: {ex}); composite=NaN.",
                  "params": p},
        )


if __name__ == "__main__":
    from .base import load_inputs
    inp = load_inputs()
    for out in ("deviation", "factor", "slope", "acceleration"):
        r = build(inp, {"output": out})
        c = r.composite.dropna()
        print(f'{out:12s} n={len(c):3d} latest={(c.iloc[-1] if len(c) else float("nan")):+.4f} '
              f'dfm_conv={r.meta.get("dfm_converged")} corrV0={r.meta.get("corr_with_v0")}')
    r = build(inp, {})   # default deviation
    print("first_break:", r.meta.get("first_break"), "| break_prob latest:",
          (r.meta["break_prob"].dropna().iloc[-1] if hasattr(r.meta.get("break_prob"), "dropna") else None))
    print("resampled_bands:", {q: round(v["point"], 2) for q, v in r.meta["resampled_bands"].items()
                               if isinstance(v, dict)})
    print("coverage:", {q: (round(v["breach_rate"], 3), v["nominal"]) for q, v in r.meta["coverage"].items()})
    print("deviation_std (~1?):", round(r.meta.get("deviation_std", float("nan")), 3))
    print("raw_level:", build(inp, {"panel": "raw_level"}).composite.dropna().shape)
    print("fixed fit:", build(inp, {"fit_mode": "fixed"}).meta.get("dfm_converged"))
