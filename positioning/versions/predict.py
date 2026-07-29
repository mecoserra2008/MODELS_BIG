"""Predictive-read helpers for the app's "does this signal predict yields?" panel.

Two reads of a version's composite vs forward US10Y changes, each returning coefficient
VALUES and a plain-English INTERPRETATION:

  forward_regression  OLS of forward Δ10Y (bp) on the standardized composite, per horizon,
                      Newey-West errors  ->  beta in bp per +1σ, t, p, R², n, interpretation.
  logistic_read       calibrated logistic P(selloff) coefficients (composite + regime
                      controls) as log-odds  ->  coef, interpretation, OOS metrics.

Honesty: positioning is a weak predictor (CIs/OOS AUC near chance) — the interpretation
strings say so. Everything is publication-lagged (no look-ahead).

NOTE: this is the frozen-signature STUB written by the orchestrator so the frontend can
import it immediately; the backend agent fills in the real logic. Stub returns empty/NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import base

HORIZONS = (4, 8, 13)


def _to_z(composite: pd.Series, bounded: bool) -> pd.Series:
    """Standardize the composite to ~unit scale for a comparable 'per +1σ' beta.
    Bounded [0,100] -> (x-50)/25; z-family -> as-is (already ~unit)."""
    s = composite.dropna().astype(float)
    if bounded:
        return (s - 50.0) / 25.0
    return s


def forward_regression(composite: pd.Series, y10_w: pd.Series,
                       horizons=HORIZONS, bounded: bool = False) -> dict:
    """OLS forward Δ10Y (bp) ~ standardized composite, per horizon (HAC/Newey-West).

    Returns {str(H): {beta_bp_per_sigma, t, p, r2, n, interpretation}}.

    The composite is standardized to a truly per-+1σ scale: `_to_z` maps a bounded
    [0,100] composite to (x-50)/25, and leaves a z-family composite as-is. z-family
    series are only *roughly* unit-variance, so we additionally divide by the series'
    own full-sample std whenever it deviates from 1 (|std-1| > 0.1) — otherwise the
    beta would be "per +1 raw unit" rather than "per +1σ". The scale factor applied is
    recorded under each horizon's `sigma_scale`.
    """
    import statsmodels.api as sm

    zseries = _to_z(composite, bounded).dropna()
    # Make the regressor genuinely "per +1σ": rescale to unit std if it isn't already.
    sigma_scale = 1.0
    if not bounded and len(zseries) > 2:
        sd = float(zseries.std(ddof=0))
        if sd and abs(sd - 1.0) > 0.1:
            zseries = zseries / sd
            sigma_scale = sd

    # Publication-lag, then as-of align onto the weekly y10 grid (no look-ahead).
    z_pub = base.pub_lag(zseries.dropna())
    z_on_y = (z_pub.reindex(y10_w.index.union(z_pub.index))
              .ffill().reindex(y10_w.index))

    out: dict = {}
    for H in horizons:
        H = int(H)
        fwd = (y10_w.shift(-H) - y10_w) * 100.0            # forward Δ10Y in bp
        d = pd.concat([fwd.rename("fwd"), z_on_y.rename("z")], axis=1).dropna()
        if len(d) < 10 or d["z"].std(ddof=0) == 0:
            out[str(H)] = {"beta_bp_per_sigma": float("nan"), "t": float("nan"),
                           "p": float("nan"), "r2": float("nan"), "n": int(len(d)),
                           "sigma_scale": sigma_scale,
                           "interpretation": "insufficient overlapping data"}
            continue
        y = d["fwd"]
        x = sm.add_constant(d["z"])
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": H})
        beta = float(model.params["z"])
        t = float(model.tvalues["z"])
        p = float(model.pvalues["z"])
        r2 = float(model.rsquared)
        n = int(model.nobs)
        interp = (
            f"+1σ crowding → {beta:+.1f}bp over {H}w (t={t:.2f}, R²={r2:.1%}); "
            f"{'significant' if p < 0.05 else 'not significant'}; sign "
            f"{'consistent with contrarian selloff' if beta > 0 else 'inconsistent (crowding→lower yields)'}."
        )
        out[str(H)] = {"beta_bp_per_sigma": beta, "t": t, "p": p, "r2": r2, "n": n,
                       "sigma_scale": sigma_scale, "interpretation": interp}
    return out


def logistic_read(composite: pd.Series, y10_w: pd.Series, horizon: int = 4) -> dict:
    """Calibrated-logistic P(selloff) coefficients + interpretation + OOS metrics.

    Returns {coef: {feat: val}, interpretation: {feat: str}, metrics: {auc, n, base_rate}}.

    Mirrors positioning.core.predictive: builds the same six regime-aware features on the
    AS-KNOWN (publication) weekly timeline, labels a forward selloff, and reads the mean
    logistic coefficients + honest walk-forward OOS metrics off `predictive.walk_forward`
    (which fits the L2 CalibratedClassifierCV(LogisticRegression(C=0.5),'sigmoid',cv=3)).

    NOTE: the frozen signature only passes the composite and DGS10, so the 2s10s `slope`
    regime control (which needs DGS2) is unavailable here and enters as a constant 0 —
    it therefore carries ~no information and its coefficient is regularized toward 0.
    """
    from positioning.core import predictive
    from positioning.core.normalize import rolling_zscore

    horizon = int(horizon)
    # Publication-lag composite and yields onto the same as-known weekly grid.
    comp_pub = base.pub_lag(composite.dropna())
    y_pub = base.pub_lag(y10_w.dropna())

    feats = pd.DataFrame({
        "composite": comp_pub,
        "composite_chg": comp_pub.diff(),
        "extreme": comp_pub.abs(),
        "y_level_z": rolling_zscore(y_pub, 156),
        "slope": 0.0,                     # DGS2 unavailable in this signature -> constant
        "realized_vol": y_pub.diff().rolling(26, min_periods=13).std(),
    })
    label = predictive.make_label(y_pub, horizon)
    pred = predictive.walk_forward(feats, label, embargo=horizon)

    coef = {k: float(v) for k, v in pred.coef.items()}
    interpretation = {
        feat: (f"coef {v:+.2f} log-odds: higher {feat} → "
               f"{'more' if v > 0 else 'less'} likely selloff")
        for feat, v in coef.items()
    }
    m = pred.metrics
    metrics = {"auc": float(m.get("auc", float("nan"))),
               "n": int(m.get("n_oos", 0)),
               "base_rate": float(m.get("base_rate", float("nan")))}
    metrics["note"] = (f"OOS AUC≈{metrics['auc']:.2f} (near the 0.5 coin-flip / typical ~0.55); "
                       f"positioning is a weak, barely-better-than-chance predictor of forward "
                       f"selloffs on the available sample.")
    return {"coef": coef, "interpretation": interpretation, "metrics": metrics}
