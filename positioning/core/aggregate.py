"""Hierarchical aggregation and PCA factor extraction.

The weighting scheme, top to bottom:

1. DV01 within tenor  (done upstream in dv01.add_net_dv01): net contracts -> net DV01,
   so tenors are additive in risk units.
2. Sum DV01 across tenors within a trader category -> per-category duration positioning.
3. Causal z-score / percentile each derived series (normalize.py).
4. PCA across the standardized feature panel -> PC1 (level positioning factor) and
   PC2 (curve positioning factor). The composite crowding index is the sign-aligned,
   re-standardized PC1. An equal-weight average of the level z-scores is retained as a
   transparent fallback (and as the sign anchor for PC1).

PCA is fit on the training slice only and projected forward — no look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..config import (
    FRONT_END, LONG_END, SCORING_CATEGORIES, Z_LOOKBACK_LONG, Z_LOOKBACK_SHORT,
)
from .normalize import rolling_percentile, rolling_zscore


# --------------------------------------------------------------------------- #
# Step 2: per-category and curve duration positioning (weekly, wide)
# --------------------------------------------------------------------------- #
def category_duration(fut_dv01: pd.DataFrame) -> pd.DataFrame:
    """Sum net_dv01 across instruments within each category -> wide [date x category]."""
    g = (fut_dv01.groupby(["date", "category"])["net_dv01"].sum()
         .unstack("category").sort_index())
    g.index = pd.to_datetime(g.index)
    return g


def curve_duration(fut_dv01: pd.DataFrame, category: str) -> pd.Series:
    """Front-end minus long-end net DV01 for one category (curve positioning)."""
    sub = fut_dv01[fut_dv01["category"] == category]
    piv = sub.pivot_table(index="date", columns="instrument",
                          values="net_dv01", aggfunc="sum")
    front = piv.reindex(columns=FRONT_END).sum(axis=1)
    long = piv.reindex(columns=LONG_END).sum(axis=1)
    s = (front - long)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def curve_crowding(fut_dv01: pd.DataFrame) -> pd.DataFrame:
    """Per-category curve-crowding z-scores + their equal-weight aggregate.

    Each scoring category's curve_duration (front-end minus long-end net DV01) is
    causally z-scored over the long lookback. Positive = crowded front-end-long vs
    long-end-short (steepener positioning); negative = flattener crowding.
    Returns wide [date x (categories..., 'aggregate', 'aggregate_restd')]:
    'aggregate' = mean of the category z's; 'aggregate_restd' = that mean re-z-scored
    — the headline read. The two categories sit on opposite sides of the curve trade
    (z correlation ~ -0.8), so the raw mean is compressed (std ~0.45, essentially
    never past +/-1.5); re-standardizing the average is the house convention for
    averaged z's (cf. equal_weight / composite above) and makes the +/-k regime
    thresholds meaningful."""
    cats = set(fut_dv01["category"])
    cols = {cat: rolling_zscore(curve_duration(fut_dv01, cat), Z_LOOKBACK_LONG)
            for cat in SCORING_CATEGORIES if cat in cats}
    out = pd.DataFrame(cols).sort_index()
    out["aggregate"] = out.mean(axis=1)
    out["aggregate_restd"] = rolling_zscore(out["aggregate"], Z_LOOKBACK_LONG)
    return out


def concentration_tilt(fut_dv01: pd.DataFrame) -> pd.Series:
    """Market net top-4 concentration tilt = top4_long% - top4_short%, OI-weighted
    across instruments. Concentration is a market-level field (same for all
    categories), so we take one row per (date, instrument)."""
    one = fut_dv01.drop_duplicates(["date", "instrument"])
    tilt = one["conc_net_top4_long"] - one["conc_net_top4_short"]
    w = one["open_interest"].fillna(0.0)
    tmp = pd.DataFrame({"date": one["date"], "tilt": tilt, "w": w})
    num = tmp.assign(tw=tmp["tilt"] * tmp["w"]).groupby("date")["tw"].sum()
    den = tmp.groupby("date")["w"].sum().replace(0.0, np.nan)
    s = (num / den)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def cash_duration(cash: pd.DataFrame) -> pd.Series:
    """Primary-dealer cash Treasury duration positioning: sum over tenor buckets of
    net_value * tenor (par-years, a monotone DV01 proxy). Weekly series."""
    tmp = cash.assign(dur=cash["net_value"] * cash["tenor_years"])
    s = tmp.groupby("date")["dur"].sum()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


# --------------------------------------------------------------------------- #
# Step 3: feature panel (causal z-scores / percentiles), wide weekly
# --------------------------------------------------------------------------- #
def build_feature_panel(fut_dv01: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
    """Assemble the standardized feature matrix fed to PCA.

    Features (all causal):
      {cat}_level_z  net DV01 level, long lookback           (lev_money, asset_mgr)
      {cat}_flow_z   weekly change in net DV01, short lookback
      curve_z        lev_money front-minus-long DV01, long lookback
      conc_z         market net top-4 concentration tilt, long lookback
      cash_z         primary-dealer cash duration, long lookback
    """
    catdur = category_duration(fut_dv01)
    feats: dict[str, pd.Series] = {}
    for cat in SCORING_CATEGORIES:
        if cat not in catdur.columns:
            continue
        level = catdur[cat]
        feats[f"{cat}_level_z"] = rolling_zscore(level, Z_LOOKBACK_LONG)
        feats[f"{cat}_flow_z"] = rolling_zscore(level.diff(), Z_LOOKBACK_SHORT)

    feats["curve_z"] = rolling_zscore(curve_duration(fut_dv01, "lev_money"), Z_LOOKBACK_LONG)
    feats["conc_z"] = rolling_zscore(concentration_tilt(fut_dv01), Z_LOOKBACK_LONG)
    if not cash.empty:
        # Cash is on the NY Fed (Wednesday) grid; as-of align it to the futures
        # (Tuesday report) grid so every feature shares one weekly index.
        target = catdur.index
        cd = cash_duration(cash)
        cd = cd.reindex(cd.index.union(target)).ffill().reindex(target)
        feats["cash_z"] = rolling_zscore(cd, Z_LOOKBACK_LONG)

    panel = pd.DataFrame(feats).sort_index()
    return panel


# --------------------------------------------------------------------------- #
# Step 4: PCA factors + composite index
# --------------------------------------------------------------------------- #
@dataclass
class FactorResult:
    scores: pd.DataFrame          # PC1, PC2 over time (projected full history)
    composite: pd.Series          # sign-aligned, re-standardized PC1 (the crowding index)
    equal_weight: pd.Series       # transparent fallback = mean of *_level_z
    percentile: pd.Series         # trailing percentile of composite (extremeness)
    loadings: pd.DataFrame        # feature loadings for PC1/PC2
    explained_var: list[float]    # explained variance ratio per PC
    feature_names: list[str]


def pca_factors(features: pd.DataFrame, train_end: str,
                n_components: int = 2) -> FactorResult:
    """Fit PCA on the training slice, project the full history.

    Returns a FactorResult. PC1 is sign-aligned so it correlates positively with the
    equal-weight level average (positive composite = crowded net long duration).
    """
    feat = features.dropna(how="all", axis=1)
    level_cols = [c for c in feat.columns if c.endswith("_level_z")]
    equal_weight = feat[level_cols].mean(axis=1) if level_cols else pd.Series(dtype=float)

    train = feat.loc[:train_end].dropna()
    n_comp = int(min(n_components, train.shape[1], max(train.shape[0] - 1, 1)))
    if train.empty or n_comp < 1:
        raise ValueError("insufficient training data for PCA")

    scaler = StandardScaler().fit(train.values)
    pca = PCA(n_components=n_comp).fit(scaler.transform(train.values))

    full = feat.dropna()
    Z = pca.transform(scaler.transform(full.values))
    cols = [f"PC{i+1}" for i in range(n_comp)]
    scores = pd.DataFrame(Z, index=full.index, columns=cols)

    # sign-align PC1 to the equal-weight level average
    pc1 = scores["PC1"]
    ew = equal_weight.reindex(pc1.index)
    if ew.notna().sum() > 5 and np.corrcoef(pc1.fillna(0), ew.fillna(0))[0, 1] < 0:
        scores = scores * -1.0
        pca.components_ = pca.components_ * -1.0
        pc1 = scores["PC1"]

    composite = rolling_zscore(pc1, Z_LOOKBACK_LONG)
    percentile = rolling_percentile(composite, Z_LOOKBACK_LONG)

    loadings = pd.DataFrame(pca.components_.T, index=full.columns, columns=cols)
    return FactorResult(
        scores=scores,
        composite=composite,
        equal_weight=rolling_zscore(equal_weight, Z_LOOKBACK_LONG),
        percentile=percentile,
        loadings=loadings,
        explained_var=[float(v) for v in pca.explained_variance_ratio_],
        feature_names=list(full.columns),
    )
