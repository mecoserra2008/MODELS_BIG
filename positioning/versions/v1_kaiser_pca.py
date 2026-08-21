"""V1 — Kaiser-PCA composite (single standardization, eigenvalue>1 retention).

This version is the *de-triple-normalized* PCA composite. The production composite
(`positioning.core.aggregate.pca_factors`) normalizes THREE times before you ever read a
number off it:

    1. rolling-3y z-score every feature series        (causal, but a big lag)
    2. StandardScaler fit on the train slice          (static, leakage-guarded)
    3. rolling-3y z-score of PC1                       (a second causal lag)

Stacking a rolling-3y z on the features AND a rolling-3y z on PC1 double-counts the
same slow, causal normalization — it is the single biggest source of the production
composite's sluggishness (it only crosses "crowded" long after the stock of positioning
has actually turned). This version keeps EXACTLY the production leakage guard (step 2:
fit the scaler + PCA on the pre-`TRAIN_END` slice, project the full history) and removes
the two rolling z's:

    * features enter as RAW net-DV01 LEVELS (no rolling-3y z)            <- removes step 1
    * PC1 is read straight off the projection, already ~unit variance    <- removes step 3
    * retention is by the Kaiser rule (eigenvalue > 1) instead of a hard n_components=2

So the ONLY methodological differences vs production are (a) single static
standardization instead of rolling-3y-z on the features, (b) Kaiser eigenvalue>1
retention instead of a fixed 2 components, and (c) no rolling re-z of PC1.

Features used (documented, exactly three — concentration is not available on the
`VersionInputs` interface, so it is intentionally skipped):

    lev_money   leveraged-funds net DV01 LEVEL ($mm/bp)   from inp.cat_dv01
    asset_mgr   asset-manager net DV01 LEVEL ($mm/bp)      from inp.cat_dv01
    front_long  buy-side front(TU+FV)-minus-long(US+UB)    from inp.buyside_front_long()

Sign convention (base.py): composite POSITIVE = crowded net LONG duration. PC1 is
sign-aligned to a POSITIVE correlation with `inp.buyside_level()` (lev+AM net DV01
level). If a 2nd PC survives Kaiser and is a curve factor (its dominant loading is the
front_long feature), it is exposed as `curve`, sign-aligned so + = steepener (more
front-end long vs long-end short).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from positioning.config import TRAIN_END
from positioning.versions import params as P
from positioning.versions.base import (
    VersionInputs, VersionResult, load_inputs, responsiveness,
)

# Buy-side scoring categories whose net-DV01 LEVELS are the level features.
LEVEL_CATEGORIES = ("lev_money", "asset_mgr")
CURVE_FEATURE = "front_long"
KAISER = 1.0  # eigenvalue > 1 retention threshold


# --------------------------------------------------------------------------- #
# Feature matrix (RAW levels — no rolling-z; that is the production lag removed)
# --------------------------------------------------------------------------- #
def _feature_matrix(inp: VersionInputs, categories, include_curve: bool,
                    curve_series: pd.Series) -> pd.DataFrame:
    """Assemble the RAW-level feature panel fed to PCA (no rolling z-score).

    `categories` selects which per-category net-DV01 LEVELS enter the matrix; when
    `include_curve` is set, the buy-side front-minus-long curve feature is appended.
    """
    cols: dict[str, pd.Series] = {}
    for cat in categories:
        if cat in inp.cat_dv01.columns:
            cols[cat] = inp.cat_dv01[cat]
    if include_curve:
        cols[CURVE_FEATURE] = curve_series
    feat = pd.DataFrame(cols).sort_index()
    feat.index = pd.to_datetime(feat.index)
    return feat


def _sign_align(series: pd.Series, anchor: pd.Series) -> float:
    """Return +1/-1 to flip `series` to a positive correlation with `anchor`."""
    a = anchor.reindex(series.index)
    mask = series.notna() & a.notna()
    if mask.sum() <= 5:
        return 1.0
    c = np.corrcoef(series[mask].values, a[mask].values)[0, 1]
    return -1.0 if np.isfinite(c) and c < 0 else 1.0


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v1", params)
    categories = list(p.get("categories") or LEVEL_CATEGORIES)
    include_curve = bool(p.get("include_curve", True))
    train_end = p.get("train_end", TRAIN_END)
    kaiser = float(p.get("kaiser_cutoff", KAISER))
    scope = p.get("standardize_scope", "train")

    feat = _feature_matrix(inp, categories, include_curve,
                           inp.buyside_front_long()).dropna(how="all", axis=1)
    features_used = list(feat.columns)

    full = feat.dropna()
    train = feat.loc[:train_end].dropna()
    n_feat = feat.shape[1]
    n_comp = int(min(n_feat, max(train.shape[0] - 1, 1)))
    if train.empty or n_comp < 1:
        raise ValueError("insufficient training data for Kaiser-PCA")

    # --- single static standardization + PCA (leakage guard) ---
    # standardize_scope: 'train' fits the scaler on the pre-train_end slice (production
    # leakage guard, the default); 'full' fits it on all available history. PCA is always
    # fit on the (standardized) training slice.
    scaler_fit = train.values if scope == "train" else full.values
    scaler = StandardScaler().fit(scaler_fit)
    pca = PCA(n_components=n_comp).fit(scaler.transform(train.values))

    Z = pca.transform(scaler.transform(full.values))
    cols = [f"PC{i + 1}" for i in range(n_comp)]
    scores = pd.DataFrame(Z, index=full.index, columns=cols)
    loadings = pd.DataFrame(pca.components_.T, index=features_used, columns=cols)

    eigenvalues = [float(v) for v in pca.explained_variance_]          # ~correlation eigenvalues
    explained_variance = [float(v) for v in pca.explained_variance_ratio_]
    n_pcs_kaiser = int(sum(1 for e in eigenvalues if e > kaiser))

    # --- composite = PC1, sign-aligned to POSITIVE corr with buy-side net-DV01 level ---
    buyside_level = inp.buyside_level()
    s1 = _sign_align(scores["PC1"], buyside_level)
    scores = scores * s1
    loadings = loadings * s1
    pc1 = scores["PC1"].rename("composite")  # already ~unit variance; NO rolling re-z

    # --- optional curve factor: PC2 must survive Kaiser AND be dominated by front_long ---
    curve = None
    curve_note = "no 2nd PC exposed"
    if n_comp >= 2 and eigenvalues[1] > kaiser:
        pc2_load = loadings["PC2"].abs()
        if CURVE_FEATURE in pc2_load.index and pc2_load.idxmax() == CURVE_FEATURE:
            pc2 = scores["PC2"]
            s2 = _sign_align(pc2, inp.buyside_front_long())  # + = steepener (front>long)
            curve = (pc2 * s2).rename("curve")
            curve_note = "PC2 survived Kaiser and loads dominantly on front_long -> exposed as curve (+ = steepener)"
        else:
            curve_note = "PC2 survived Kaiser but is not a curve factor (front_long not dominant); not exposed"
    elif n_comp >= 2:
        curve_note = f"PC2 eigenvalue {eigenvalues[1]:.3f} <= {kaiser:g} (Kaiser): only PC1 retained"

    # --- documented sub-variant (meta ONLY): PCA on the RAW (unscaled) covariance ---
    # No scaler -> the largest-variance series dominates PC1. This is exactly what the
    # single standardization above fixes, so it is reported but never used as composite.
    raw_pca = PCA(n_components=n_comp).fit(train.values)
    raw_Z = raw_pca.transform(full.values)
    raw_pc1 = pd.Series(raw_Z[:, 0], index=full.index)
    raw_pc1 = raw_pc1 * _sign_align(raw_pc1, buyside_level)
    raw_eigs = [float(v) for v in raw_pca.explained_variance_]
    raw_n_kaiser = int(sum(1 for e in raw_eigs if e > kaiser))
    dominant = train.var().idxmax()
    raw_cov_pc1_latest = float(raw_pc1.dropna().iloc[-1]) if raw_pc1.notna().any() else None

    note = (
        "Single static standardization (StandardScaler fit on TRAIN=%s) + PCA fit on the "
        "same train slice, projected to full history. Kaiser eigenvalue>1 retention. "
        "composite = raw PC1 (unit-variance, NO rolling re-z). Removes production's rolling-3y-z "
        "on features AND rolling-3y-z on PC1 (the triple-normalization lag). %s. raw_cov "
        "sub-variant: PCA on the UNSCALED covariance of the same levels is dominated by the "
        "largest-variance series (%s, raw var >> others), which is why standardization matters; "
        "reported in meta only, never the composite."
        % (train_end, curve_note, dominant)
    )

    meta = {
        "n_pcs_kaiser": n_pcs_kaiser,
        "eigenvalues": eigenvalues,
        "explained_variance": explained_variance,
        "features_used": features_used,
        "raw_cov_pc1_latest": raw_cov_pc1_latest,
        "note": note,
        # extras (transparency)
        "train_end": train_end,
        "categories": categories,
        "include_curve": include_curve,
        "kaiser_cutoff": kaiser,
        "standardize_scope": scope,
        "loadings": {c: loadings[c].round(4).to_dict() for c in cols},
        "raw_cov_eigenvalues": raw_eigs,
        "raw_cov_n_pcs_kaiser": raw_n_kaiser,
        "raw_cov_dominant_feature": dominant,
        "curve_exposed": curve is not None,
        "n_train_rows": int(train.shape[0]),
    }

    return VersionResult(
        name="V1 Kaiser-PCA (single-std, eigenvalue>1)",
        description=(
            "PCA composite with a SINGLE static standardization (StandardScaler fit on the "
            "pre-TRAIN_END slice, projected forward) over RAW per-category net-DV01 levels "
            "(lev_money, asset_mgr) plus the buy-side front-minus-long curve feature. Kaiser "
            "(eigenvalue>1) component retention; composite = sign-aligned PC1 with NO rolling "
            "re-z. Removes the production composite's triple normalization (rolling-3y-z on "
            "features + rolling-3y-z on PC1)."
        ),
        composite=pc1,
        bounded=False,
        curve=curve,
        meta=meta,
    )


if __name__ == "__main__":
    inp = load_inputs()
    res = build(inp)
    comp = res.composite.dropna()
    print("name:", res.name)
    print("features_used:", res.meta["features_used"])
    print("n_pcs_kaiser:", res.meta["n_pcs_kaiser"])
    print("eigenvalues:", [round(e, 4) for e in res.meta["eigenvalues"]])
    print("explained_variance:", [round(e, 4) for e in res.meta["explained_variance"]])
    print("curve_exposed:", res.meta["curve_exposed"])
    print("raw_cov_pc1_latest:", res.meta["raw_cov_pc1_latest"],
          "(dominant feature:", res.meta["raw_cov_dominant_feature"] + ")")
    print("latest composite:", None if comp.empty else round(float(comp.iloc[-1]), 4),
          "as of", None if comp.empty else str(comp.index.max().date()))
    print("responsiveness:", responsiveness(res.composite, res.bounded))
