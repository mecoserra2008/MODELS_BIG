"""Track A driver: build the free/public Fixed Income positioning score.

Wires the free adapters (CFTC TFF + NY Fed cash + FRED yields) into the source-agnostic
scoring core, then writes results/positioning_score.json and fig_*.png diagnostics.

    python -m positioning.run_positioning [--force]

The Bloomberg/BQuant track (bquant/positioning_bbg.ipynb) reuses positioning.core with a
bql adapter, so only the three fetch_* calls below change.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    CATEGORY_LABELS, CONTRACTS, RESULTS_DIR, SIGNAL_K_ENTRY, TRAIN_END,
    Z_LOOKBACK_LONG,
)
from .adapters import cftc_free, nyfed_free, yields_free
from .core import score as score_mod
from .core.normalize import rolling_zscore

INSTRUMENT_ORDER = [c.key for c in CONTRACTS]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_composite(res, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    comp = res.composite.dropna()
    ax.plot(comp.index, comp.values, color="#1f4e79", lw=1.3, label="Composite crowding z")
    ax.axhline(SIGNAL_K_ENTRY, color="#c00000", ls="--", lw=0.8)
    ax.axhline(-SIGNAL_K_ENTRY, color="#c00000", ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    st = res.stance.reindex(comp.index).fillna(0)
    ax.fill_between(comp.index, comp.values, 0, where=st.values < 0, color="#c00000",
                    alpha=0.12, label="short-duration stance")
    ax.fill_between(comp.index, comp.values, 0, where=st.values > 0, color="#2e7d32",
                    alpha=0.12, label="long-duration stance")
    ax.set_title("US Treasury futures positioning — composite crowding index (PC1)")
    ax.set_ylabel("z-score"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _fig_heatmap(res, path):
    # latest net-DV01 z by instrument x category (level-normalized per series)
    piv = res.fut_dv01.pivot_table(index="date", columns=["category", "instrument"],
                                   values="net_dv01", aggfunc="sum").sort_index()
    z = piv.apply(lambda c: rolling_zscore(c, Z_LOOKBACK_LONG))
    latest = z.dropna(how="all").iloc[-1]
    cats = [c for c in CATEGORY_LABELS if c in piv.columns.get_level_values(0)]
    M = np.full((len(cats), len(INSTRUMENT_ORDER)), np.nan)
    for i, cat in enumerate(cats):
        for j, ins in enumerate(INSTRUMENT_ORDER):
            if (cat, ins) in latest.index:
                M[i, j] = latest[(cat, ins)]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(INSTRUMENT_ORDER))); ax.set_xticklabels(INSTRUMENT_ORDER)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([CATEGORY_LABELS[c] for c in cats])
    for i in range(len(cats)):
        for j in range(len(INSTRUMENT_ORDER)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"Net-DV01 positioning z by trader × tenor (as of {res.latest['as_of']})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="z")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _fig_loadings(res, path):
    L = res.factors.loadings
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(L.shape[1])); ax.set_xticklabels(L.columns)
    ax.set_yticks(range(L.shape[0])); ax.set_yticklabels(L.index, fontsize=8)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ev = ", ".join(f"PC{i+1}={v:.0%}" for i, v in enumerate(res.factors.explained_var))
    ax.set_title(f"PCA loadings (var explained: {ev})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _fig_forward_prob(res, path):
    p = res.predictive.proba
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(p.index, p.values, color="#7030a0", lw=1.0, label="OOS P(selloff, next 4w)")
    ax.axhline(res.predictive.metrics["base_rate"], color="grey", ls="--", lw=0.8,
               label=f"base rate {res.predictive.metrics['base_rate']:.2f}")
    ax.set_ylim(0, 1); ax.set_title("Walk-forward P(selloff) from positioning")
    m = res.predictive.metrics
    ax.text(0.01, 0.02, f"AUC={m['auc']:.2f}  logloss={m['log_loss']:.2f}  "
            f"brier={m['brier']:.2f}  n={m['n_oos']}", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _fig_calibration(res, path):
    c = res.predictive.calibration
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    if not c.empty:
        ax.plot(c["pred_mean"], c["realized"], "o-", color="#1f4e79")
        for _, r in c.iterrows():
            ax.annotate(int(r["n"]), (r["pred_mean"], r["realized"]), fontsize=7)
    ax.set_xlabel("predicted P(selloff)"); ax.set_ylabel("realized frequency")
    ax.set_title("Calibration (walk-forward)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
def build_results(res) -> dict:
    catdur_latest = res.category_duration.dropna(how="all").iloc[-1]
    net_dv01_latest = (res.fut_dv01[res.fut_dv01["date"] == res.fut_dv01["date"].max()]
                       .groupby("instrument")["net_dv01"].sum())
    return {
        "config": {
            "instruments": INSTRUMENT_ORDER,
            "scoring_categories": list(CATEGORY_LABELS),
            "train_end": TRAIN_END,
            "z_lookback_long_weeks": Z_LOOKBACK_LONG,
            "forward_horizon_weeks": 4,
        },
        "latest": res.latest,
        "pca": {
            "explained_variance_ratio": res.factors.explained_var,
            "loadings": res.factors.loadings.round(4).to_dict(),
            "feature_names": res.factors.feature_names,
        },
        "predictive": {
            "metrics": res.predictive.metrics,
            "coefficients": {k: round(v, 4) for k, v in res.predictive.coef.items()},
            "latest_p_selloff": res.predictive.latest_proba,
        },
        "latest_category_net_dv01_z": {k: (None if pd.isna(v) else float(v))
                                       for k, v in catdur_latest.items()},
        "latest_instrument_net_dv01_usd_per_bp": {k: float(v)
                                                  for k, v in net_dv01_latest.items()},
    }


def main(force: bool = False):
    print("Fetching CFTC TFF ...")
    fut = cftc_free.fetch_cftc_tff(force_download=force)
    print(f"  {len(fut)} rows, {fut['date'].min().date()} -> {fut['date'].max().date()}")
    print("Fetching NY Fed primary-dealer cash ...")
    cash = nyfed_free.fetch_nyfed_cash(force_download=force)
    print(f"  {len(cash)} rows" + ("" if len(cash) else " (unavailable — futures-only)"))
    print("Fetching FRED yields ...")
    yields = yields_free.fetch_yields(force_download=force)
    print(f"  {yields.shape[0]} days x {yields.shape[1]} tenors")

    print("Scoring ...")
    res = score_mod.run_score(fut, cash, yields)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _fig_composite(res, RESULTS_DIR / "fig_positioning_composite.png")
    _fig_heatmap(res, RESULTS_DIR / "fig_positioning_heatmap.png")
    _fig_loadings(res, RESULTS_DIR / "fig_pca_loadings.png")
    _fig_forward_prob(res, RESULTS_DIR / "fig_forward_prob.png")
    _fig_calibration(res, RESULTS_DIR / "fig_calibration.png")

    results = build_results(res)
    out = RESULTS_DIR / "positioning_score.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")
    print(json.dumps(res.latest, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="force re-download of all data")
    args = ap.parse_args()
    main(force=args.force)
