"""V7 Breadth / orthogonalized composite (multi-signal breadth).

More INDEPENDENT bets per unit time (Grinold IR=TC·IC·√BR). Gram–Schmidt orthogonalize
level -> flow⊥level -> accel⊥flow, add a cross-sectional breadth-diffusion index (signed share of
the category×tenor cells with |z|>thr) and a dispersion index (std across cell-z) as SEPARATE
weekly signals, combined by robust-z / rank average; optional detoned-PCA (strip PC1 -> PC2) or ICA.
Reports BR_eff = N/[1+(N−1)ρ] so PC1-cousins aren't double-counted.

Reuses base helpers, transforms.apply_norm and core.normalize.zscore_frame. Built from the RAW
per-cell / aggregate net-DV01 panel (2018→2026). POSITIVE = crowded net long.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from positioning.config import SCORING_CATEGORIES, TRAIN_END
from positioning.core.normalize import zscore_frame

from . import base
from . import params as P
from . import transforms
from .base import VersionInputs, VersionResult

_ALL_CATS = ("lev_money", "asset_mgr", "dealer")


def _last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _residual(y: pd.Series, regressors: list[pd.Series]) -> pd.Series:
    """OLS residual of y on [1, *regressors] (aligned dropna), reindexed to y's index."""
    parts = {"y": y}
    for i, r in enumerate(regressors):
        parts[f"x{i}"] = r
    df = pd.DataFrame(parts).dropna()
    if len(df) < len(regressors) + 3:
        return y.copy()
    A = np.column_stack([np.ones(len(df))] + [df[f"x{i}"].to_numpy() for i in range(len(regressors))])
    coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - A @ coef
    return pd.Series(resid, index=df.index).reindex(y.index)


def _trailing_rank(s: pd.Series, window: int) -> pd.Series:
    """Causal trailing percentile rank in (0,1] of the last point within its trailing window."""
    mp = max(window // 2, 20)
    return s.rolling(window, min_periods=mp).apply(
        lambda w: (w.argsort().argsort()[-1] + 1) / len(w), raw=True)


def _detoned_pc2(X: pd.DataFrame, z_window: int, level: pd.Series,
                 train_end: str = TRAIN_END) -> tuple[pd.Series, dict]:
    """Strip PC1 (the aggregate-duration factor); score = rolling-z of the PC2 projection.
    Scaler+PCA fit on the pre-train_end slice, projected over the full history (no look-ahead)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xd = X.dropna()
    diag: dict = {}
    if Xd.shape[1] < 2 or Xd.loc[:train_end].shape[0] < 10:
        return pd.Series(index=X.index, dtype=float), diag
    train = Xd.loc[:train_end]
    scaler = StandardScaler().fit(train.values)
    pca = PCA(n_components=Xd.shape[1]).fit(scaler.transform(train.values))
    scores = pca.transform(scaler.transform(Xd.values))
    pc2 = pd.Series(scores[:, 1], index=Xd.index).reindex(X.index)
    # Orient so + correlates with the aggregate level (crowded long).
    aln = pd.concat([pc2.rename("p"), level.rename("l")], axis=1).dropna()
    if len(aln) > 2 and aln["p"].corr(aln["l"]) < 0:
        pc2 = -pc2
    diag = {"pca_evr": [round(float(v), 3) for v in pca.explained_variance_ratio_[:3]]}
    return base.rolling_z(pc2, z_window), diag


def _ica_component(cellz: pd.DataFrame, z_window: int, level: pd.Series) -> tuple[pd.Series, dict]:
    """FastICA on the (whitened) cell-z panel; pick the source with max |corr| to level, z it."""
    from sklearn.decomposition import FastICA

    C = cellz.dropna()
    diag: dict = {}
    if C.shape[0] < 30 or C.shape[1] < 2:
        return pd.Series(index=cellz.index, dtype=float), diag
    k = int(min(5, C.shape[1]))
    try:
        ica = FastICA(n_components=k, whiten="unit-variance", random_state=0, max_iter=1000)
        S = ica.fit_transform(C.values)
    except Exception as exc:                         # convergence / numerical — degrade to level
        return base.rolling_z(level, z_window), {"ica_error": str(exc)}
    Sdf = pd.DataFrame(S, index=C.index)
    lev = level.reindex(C.index)
    corrs = {c: abs(Sdf[c].corr(lev)) for c in Sdf.columns}
    best = max(corrs, key=lambda c: (corrs[c] if corrs[c] == corrs[c] else -1.0))
    comp = Sdf[best].reindex(cellz.index)
    aln = pd.concat([comp.rename("c"), level.rename("l")], axis=1).dropna()
    if len(aln) > 2 and aln["c"].corr(aln["l"]) < 0:  # orient + with level
        comp = -comp
    diag = {"ica_k": k, "ica_pick": int(best), "ica_absorr": round(float(corrs[best]), 3)}
    return base.rolling_z(comp, z_window), diag


def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v7", params)
    z_w = int(p["z_window"])
    flow_w = int(p["flow_window"])
    accel_w = int(p["accel_window"])

    # 1. Cell panel X + aggregate buy-side level (upstream-normalized).
    cell_cats = _ALL_CATS if p.get("panel") == "all" else tuple(SCORING_CATEGORIES)
    cols = [c for c in inp.contract_dv01.columns if c[0] in cell_cats]
    X = inp.contract_dv01[cols].astype(float)
    sig = P.resolve_signal(inp, p).astype(float)
    sig = transforms.apply_norm(sig, p["norm_mode"]).rename("signal")

    # 2. Time-series components (each a z-series).
    level = base.rolling_z(sig, z_w).rename("level")
    flow = base.rolling_z(sig.diff(flow_w), z_w).rename("flow")
    accel = base.rolling_z(sig.diff(flow_w).diff(accel_w), z_w).rename("accel")

    if p.get("orthogonalize", True):
        # Gram–Schmidt: flow⊥level, accel⊥{level,flow_res}; replace with re-z-scored residuals.
        flow_res = _residual(flow, [level])
        accel_res = _residual(accel, [level, flow_res])
        flow = base.rolling_z(flow_res, z_w).rename("flow")
        accel = base.rolling_z(accel_res, z_w).rename("accel")

    # 3. Cross-sectional components from the cell-z panel (a fresh weekly reading).
    cellz = zscore_frame(X, z_w)
    thr = float(p["breadth_thr"])
    breadth_raw = (np.sign(cellz) * (cellz.abs() > thr)).mean(axis=1)   # signed diffusion in [−1,1]
    breadth = base.rolling_z(breadth_raw, z_w).rename("breadth")
    dispersion_raw = cellz.std(axis=1)                                  # cross-sectional disagreement
    dispersion = base.rolling_z(dispersion_raw, z_w).rename("dispersion")

    comp_map = {"level": level, "flow": flow, "accel": accel,
                "breadth": breadth, "dispersion": dispersion}
    selected = [c for c in p["components"] if c in comp_map]
    if not selected:
        selected = ["level"]
    comp_df = pd.DataFrame({c: comp_map[c] for c in selected})

    # 4. Combine.
    combine = p.get("combine", "equal")
    extra: dict = {}
    if combine == "rank":
        ranks = pd.DataFrame({c: _trailing_rank(comp_df[c], z_w) for c in selected})
        combined = (ranks.mean(axis=1) - 0.5).rename("v7")          # centered (+ = above-median crowd)
    elif combine == "detoned_pca":
        combined, extra = _detoned_pc2(X, z_w, level, p.get("train_end", TRAIN_END))
        combined = combined.rename("v7")
    elif combine == "ica":
        combined, extra = _ica_component(cellz, z_w, level)
        combined = combined.rename("v7")
    else:  # equal (robust average of the selected z-series)
        combined = comp_df.mean(axis=1).rename("v7")

    # 5. Effective breadth BR_eff = N / [1 + (N−1)ρ], ρ = mean off-diagonal cell-z correlation.
    N = int(cellz.shape[1])
    corr = cellz.corr()
    if N > 1 and corr.notna().values.any():
        off = corr.values[~np.eye(N, dtype=bool)]
        rho = float(np.nanmean(off))
    else:
        rho = float("nan")
    denom = 1.0 + (N - 1) * rho if (N > 1 and np.isfinite(rho)) else np.nan
    # Guard the Grinold singularity at ρ -> −1/(N−1) (denom -> 0): report inf rather than a wild sign.
    if isinstance(denom, float) and np.isfinite(denom) and denom > 1e-6:
        br_eff = float(N / denom)
    elif isinstance(denom, float) and np.isfinite(denom) and denom <= 1e-6:
        br_eff = float("inf")
    else:
        br_eff = float("nan")

    # Honest, data-conditional note: for the buy-side panel the cells are OFFSETTING (asset
    # managers long duration vs leveraged funds short), so mean ρ is ~0/negative and BR_eff is not
    # ≪ N — the naive "positions co-move so real breadth ≪ N" prior does not hold on this panel.
    if np.isfinite(rho) and rho > 0.05:
        breadth_msg = (f"cells co-move (ρ={rho:+.2f}) so real breadth BR_eff≈{br_eff:.1f} ≪ N={N} "
                       f"— nominal cell count overstates independence")
    else:
        breadth_msg = (f"cells are ~independent/offsetting (ρ={rho:+.2f}: asset-mgr long vs "
                       f"lev-money short) so BR_eff≈{br_eff:.1f} ≳ N={N} — an honest finding that "
                       f"contradicts the 'positions co-move' prior")
    note = (
        f"combine={combine}, orthogonalize={p.get('orthogonalize', True)}: {len(selected)} "
        f"components; {breadth_msg}. POSITIVE = crowded net long. dispersion is a "
        f"disagreement/regime read, not directional."
    )
    meta = {
        "combine": combine,
        "orthogonalize": bool(p.get("orthogonalize", True)),
        "panel": p.get("panel"),
        "norm_mode": p["norm_mode"],
        "components": selected,
        "level_latest": _last(level),
        "flow_latest": _last(flow),
        "accel_latest": _last(accel),
        "breadth_latest": _last(breadth),
        "dispersion_latest": _last(dispersion),
        "combined_latest": _last(combined),
        "BR_eff": br_eff,
        "N": N,
        "rho": rho,
        "n_finite": int(combined.notna().sum()),
        "note": note,
        **extra,
    }

    return VersionResult(
        name="V7 Breadth / orthogonalized composite",
        description="Gram–Schmidt-orthogonalized level/flow/accel + cross-sectional breadth-diffusion "
                    "and dispersion, combined by equal/rank/detoned-PCA/ICA; reports BR_eff so "
                    "co-moving cells aren't double-counted. POSITIVE = crowded net long.",
        composite=combined,
        bounded=False,
        meta=meta,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from .base import load_inputs

    inp = load_inputs()
    cases = [
        ("defaults", None),
        ("rank", {"combine": "rank"}),
        ("detoned_pca", {"combine": "detoned_pca"}),
        ("ica", {"combine": "ica"}),
        ("no_orth", {"orthogonalize": False}),
    ]
    for label, ov in cases:
        res = build(inp, ov)
        comp = res.composite
        print(f"[{label:12s}] n_finite={int(comp.notna().sum()):4d}  latest={_last(comp):+.4f}  "
              f"BR_eff={res.meta['BR_eff']:.2f}  N={res.meta['N']}  rho={res.meta['rho']:.3f}")
