"""Orchestrator: standardized panels -> PositioningScore.

Owns the point-in-time policy so leakage rules live in one place:

* Futures features are built on the weekly CFTC report (Tuesday) sequence — the ordering
  is all the causal rolling z-scores need.
* For anything that touches the future (the predictive label), the composite is stamped
  with its *publication* date (report + COT lag), and the yield label is read on that
  same as-known timeline. So features (known at publication) and label (yields strictly
  after publication) never overlap.

Both the free and Bloomberg tracks call run_score() with the same three frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    COT_PUBLICATION_LAG_BDAYS, FORWARD_HORIZON_WEEKS, LABEL_YIELD, SCORING_CATEGORIES,
    SIGNAL_K_ENTRY, SIGNAL_K_EXIT, TRAIN_END, Z_LOOKBACK_LONG,
)
from . import aggregate, dv01, predictive, schema, signal
from .normalize import rolling_zscore


@dataclass
class PositioningScore:
    composite: pd.Series            # crowding z-index, report-dated (for display)
    percentile: pd.Series           # trailing percentile of composite
    equal_weight: pd.Series         # transparent equal-weight fallback
    category_duration: pd.DataFrame # per-category net DV01 over time
    feature_panel: pd.DataFrame     # standardized features fed to PCA
    factors: aggregate.FactorResult
    predictive: predictive.PredictiveResult
    stance: pd.Series               # discrete contrarian {+1,0,-1}
    fut_dv01: pd.DataFrame          # futures panel with net_dv01 (for heatmap)
    latest: dict                    # latest-week snapshot (headline outputs)


def _weekly_yields(yields: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of (ffill) the daily yields frame onto an arbitrary weekly index."""
    y = yields.sort_index().reindex(
        yields.index.union(index)).ffill().reindex(index)
    return y


def _predictive_features(comp_pub: pd.Series, ywk_pub: pd.DataFrame) -> pd.DataFrame:
    """Layer-2 features on the as-known (publication) weekly timeline.

    Positioning signals + regime context (all causal, no look-ahead):
      composite / composite_chg / extreme  -> the crowding read and its momentum
      y_level_z                            -> where 10y yields sit (156w z)
      slope                                -> 2s10s curve level (regime)
      realized_vol                         -> 26w std of weekly 10y yield changes

    Regime context matters because positioning tends to predict returns more strongly
    at yield/vol extremes than on average.
    """
    y10 = ywk_pub[LABEL_YIELD]
    feats = {
        "composite": comp_pub,
        "composite_chg": comp_pub.diff(),
        "extreme": comp_pub.abs(),
        "y_level_z": rolling_zscore(y10, Z_LOOKBACK_LONG),
        "slope": (y10 - ywk_pub["DGS2"]) if "DGS2" in ywk_pub.columns else y10 * 0.0,
        "realized_vol": y10.diff().rolling(26, min_periods=13).std(),
    }
    return pd.DataFrame(feats)


def run_score(futures_raw: pd.DataFrame, cash_raw: pd.DataFrame,
              yields_raw: pd.DataFrame, *,
              train_end: str = TRAIN_END,
              forward_horizon: int = FORWARD_HORIZON_WEEKS,
              k_entry: float = SIGNAL_K_ENTRY,
              k_exit: float = SIGNAL_K_EXIT) -> PositioningScore:
    """Build the full positioning score.

    The methodology knobs (train split, predictive horizon, signal bands) are optional
    overrides defaulting to config, so a UI can recompute reactively without touching
    globals. Deeper lookbacks live in config (fixed for a stable factor definition).
    """
    fut = schema.validate_futures_panel(futures_raw)
    cash = schema.validate_cash_panel(cash_raw) if cash_raw is not None and len(cash_raw) \
        else pd.DataFrame(columns=schema.CASH_COLUMNS)
    yields = schema.validate_yields(yields_raw)

    # --- weekly report grid (Tuesday snapshots) ---
    report_dates = pd.DatetimeIndex(sorted(fut["date"].unique()))
    ywk = _weekly_yields(yields, report_dates)

    # --- Step 1: DV01 conversion ---
    fut_dv01 = dv01.add_net_dv01(fut, ywk)

    # --- Steps 2-4: features + PCA composite ---
    feat = aggregate.build_feature_panel(fut_dv01, cash)
    factors = aggregate.pca_factors(feat, train_end=train_end)
    composite = factors.composite
    catdur = aggregate.category_duration(fut_dv01)

    # --- Step 7: discrete contrarian stance ---
    stance = signal.contrarian_signal(composite, k_entry, k_exit)

    # --- Step 6: predictive overlay on the AS-KNOWN (publication) timeline ---
    pub_offset = pd.tseries.offsets.BDay(COT_PUBLICATION_LAG_BDAYS)
    pub_index = composite.index + pub_offset
    comp_pub = pd.Series(composite.values, index=pub_index, name="composite")
    ywk_pub = _weekly_yields(yields, pub_index)
    ywk_pub.index = pub_index
    pred_feats = _predictive_features(comp_pub, ywk_pub)
    y_pub = ywk_pub[LABEL_YIELD]
    label = predictive.make_label(y_pub, forward_horizon)
    pred = predictive.walk_forward(pred_feats, label)

    latest = _latest_snapshot(composite, factors, catdur, feat, stance, pred, fut_dv01)
    return PositioningScore(
        composite=composite, percentile=factors.percentile,
        equal_weight=factors.equal_weight, category_duration=catdur,
        feature_panel=feat, factors=factors, predictive=pred, stance=stance,
        fut_dv01=fut_dv01, latest=latest,
    )


def _latest_snapshot(composite, factors, catdur, feat, stance, pred, fut_dv01) -> dict:
    t = composite.dropna().index.max()
    def g(s):
        v = s.get(t, np.nan) if hasattr(s, "get") else np.nan
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    cat_latest = {}
    for cat in SCORING_CATEGORIES:
        col = f"{cat}_level_z"
        if col in feat.columns:
            cat_latest[cat] = g(feat[col])

    st = stance.get(t, 0)
    stance_txt = {1: "long duration (fade crowded shorts)",
                  0: "neutral",
                  -1: "short duration (fade crowded longs)"}[int(st)]
    return {
        "as_of": str(pd.Timestamp(t).date()),
        "composite_z": g(composite),
        "percentile": g(factors.percentile),
        "equal_weight_z": g(factors.equal_weight),
        "category_level_z": cat_latest,
        "curve_z": g(feat["curve_z"]) if "curve_z" in feat else None,
        "concentration_z": g(feat["conc_z"]) if "conc_z" in feat else None,
        "cash_z": g(feat["cash_z"]) if "cash_z" in feat else None,
        "stance": int(st),
        "stance_text": stance_txt,
        "p_selloff_next_%dw" % 4: pred.latest_proba,
        "pca_explained_var": factors.explained_var,
    }
