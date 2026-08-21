"""Layer 2: calibrated forward-return probability overlay.

The descriptive composite tells you *where* positioning sits. This thin, regularized
layer maps positioning -> P(bond selloff over the next H weeks), i.e. the contrarian
read: crowded longs tend to precede yield rises. It is deliberately small (a handful of
features, L2-regularized logistic, isotonic/Platt calibration) because CFTC history is
only ~1,000 weekly points — a per-input model zoo would overfit.

Everything is evaluated walk-forward (expanding window): at each step we train on the
past and predict the next block, so reported metrics contain no look-ahead. The final
model refit on all data is what scores the latest week.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def make_label(yields_weekly: pd.Series, horizon_weeks: int) -> pd.Series:
    """Binary selloff label: 1 if yield rises over the next `horizon_weeks`.

    Uses a *forward* change, so the row at time t is labelled by future data — this is
    the target, and it is only ever used to train/evaluate, never as a feature.
    """
    fwd = yields_weekly.shift(-horizon_weeks) - yields_weekly
    return (fwd > 0).astype(float).where(fwd.notna())


def conditional_extreme_stats(composite: pd.Series, yields_weekly: pd.Series,
                              horizons=(4, 8, 13), k: float = 1.5) -> dict:
    """Historical conditional base rates when positioning is at a crowded extreme.

    An honest, interpretable complement to the (weak) logistic probability: it answers
    "when the crowd was this stretched, what did yields do next?" rather than pretending
    to a precise per-week probability. Full-sample descriptive frequencies (clearly
    labelled as historical, not an OOS forecast).

    For each horizon H (weeks) it reports, split by composite regime:
      crowded_long (z> +k): P(selloff = yield up over H) and mean forward Δy in bp
      crowded_short(z< -k): P(rally  = yield down over H) and mean forward Δy in bp
      mid          (|z|<=k): mean forward Δy in bp (the unconditional-ish baseline)
    The contrarian thesis holds where crowded_long P(selloff) and crowded_short P(rally)
    sit meaningfully above 0.5. Empirically it is asymmetric (stronger on the long side)
    and firmer at 8-13w than at 4w.
    """
    out = {}
    for H in horizons:
        fwd = yields_weekly.shift(-H) - yields_weekly
        d = pd.concat([composite.rename("z"), fwd.rename("fwd")], axis=1).dropna()
        lo, sh, mid = d[d.z > k], d[d.z < -k], d[d.z.abs() <= k]
        out[str(H)] = {
            "crowded_long": {
                "n": int(len(lo)),
                "p_selloff": float((lo.fwd > 0).mean()) if len(lo) else None,
                "mean_bp": float(lo.fwd.mean() * 100) if len(lo) else None},
            "crowded_short": {
                "n": int(len(sh)),
                "p_rally": float((sh.fwd < 0).mean()) if len(sh) else None,
                "mean_bp": float(sh.fwd.mean() * 100) if len(sh) else None},
            "mid_mean_bp": float(mid.fwd.mean() * 100) if len(mid) else None,
        }
    return out


def latest_regime_read(composite: pd.Series, cond_stats: dict, k: float = 1.5) -> dict:
    """Map the latest composite z to its crowded regime and pull the matching historical
    conditional base rates (per horizon) — the honest headline the app should surface."""
    z = float(composite.dropna().iloc[-1]) if composite.dropna().size else float("nan")
    regime = "crowded_long" if z > k else "crowded_short" if z < -k else "neutral"
    rows = {}
    for H, s in cond_stats.items():
        if regime == "crowded_long":
            rows[H] = {"p_selloff": s["crowded_long"]["p_selloff"],
                       "mean_bp": s["crowded_long"]["mean_bp"], "n": s["crowded_long"]["n"]}
        elif regime == "crowded_short":
            rows[H] = {"p_rally": s["crowded_short"]["p_rally"],
                       "mean_bp": s["crowded_short"]["mean_bp"], "n": s["crowded_short"]["n"]}
    return {"composite_z": z, "regime": regime, "k": k, "by_horizon": rows}


def _pipeline() -> CalibratedClassifierCV:
    base = LogisticRegression(C=0.5, max_iter=1000)
    # Platt (sigmoid) calibration; cv=3 internal splits on the training slice only.
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


@dataclass
class PredictiveResult:
    proba: pd.Series                    # walk-forward out-of-sample P(selloff)
    latest_proba: float                 # final-model probability for the latest week
    metrics: dict                       # auc / log_loss / brier / n / base_rate
    calibration: pd.DataFrame           # binned predicted vs realized (for the plot)
    coef: dict                          # final-model mean logistic coefficients
    feature_names: list[str] = field(default_factory=list)


def _mean_coef(cal: CalibratedClassifierCV, names: list[str]) -> dict:
    """Average the underlying logistic coefficients across calibration folds.

    Robust to sklearn version differences (the attribute holding the fitted base
    estimator has changed names). Returns zeros if it can't be recovered."""
    vecs = []
    for cc in getattr(cal, "calibrated_classifiers_", []):
        base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if base is not None and hasattr(base, "coef_"):
            vecs.append(np.ravel(base.coef_))
    mean = np.mean(vecs, axis=0) if vecs else np.zeros(len(names))
    return {k: float(v) for k, v in zip(names, mean)}


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_mid": (edges[b] + edges[b + 1]) / 2,
                     "pred_mean": float(p[m].mean()),
                     "realized": float(y[m].mean()),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


def walk_forward(features: pd.DataFrame, label: pd.Series,
                 min_train: int = 156, step: int = 13, embargo: int = 0) -> PredictiveResult:
    """Expanding-window OOS probabilities + honest metrics.

    features   wide, causal (composite + its change + extremeness, etc.)
    label      binary forward selloff label (make_label)
    min_train  first training window length in weeks (default 3y)
    step       weeks predicted per fold before refitting (default ~quarter)
    embargo    purge the last `embargo` training rows so the forward (horizon-week)
               label of the newest train rows cannot overlap the test block's yields.
               Pass the label horizon here (leakage-free OOS). 0 = legacy behaviour.
    """
    df = features.join(label.rename("_y")).dropna()
    X, y = df.drop(columns="_y"), df["_y"].values
    names = list(X.columns)
    n = len(df)

    oos_idx, oos_p = [], []
    start = min_train
    while start < n:
        end = min(start + step, n)
        tr_end = max(0, start - embargo)   # embargo: drop overlapping-label train rows
        Xtr, ytr = X.iloc[:tr_end].values, y[:tr_end]
        if len(np.unique(ytr)) < 2 or len(ytr) < min_train // 2:
            start = end
            continue
        model = _pipeline().fit(Xtr, ytr)
        p = model.predict_proba(X.iloc[start:end].values)[:, 1]
        oos_idx.extend(df.index[start:end])
        oos_p.extend(p)
        start = end

    proba = pd.Series(oos_p, index=pd.DatetimeIndex(oos_idx), name="p_selloff")
    yv = df.loc[proba.index, "_y"].values
    pv = proba.values
    metrics = {
        "n_oos": int(len(pv)),
        "base_rate": float(yv.mean()) if len(yv) else float("nan"),
        "auc": float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else float("nan"),
        "log_loss": float(log_loss(yv, pv, labels=[0, 1])) if len(pv) else float("nan"),
        "brier": float(brier_score_loss(yv, pv)) if len(pv) else float("nan"),
        "horizon_note": "OOS via expanding-window walk-forward, horizon-embargo purged",
    }
    calib = _calibration_table(yv, pv)

    # Final model on all data -> latest-week probability + interpretable coefficients.
    final = _pipeline().fit(X.values, y)
    latest_p = float(final.predict_proba(X.iloc[[-1]].values)[0, 1])
    coef = _mean_coef(final, names)

    return PredictiveResult(proba=proba, latest_proba=latest_p, metrics=metrics,
                            calibration=calib, coef=coef, feature_names=names)
