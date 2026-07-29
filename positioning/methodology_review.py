"""Methodology review: why the composite is slow + faster-normalization alternatives.

Rebuilds the positioning inputs from positioning.core (cache-first) and applies alternative
NORMALIZATIONS to the SAME underlying net-DV01 signal, so the only thing that varies is the
transform. Quantifies responsiveness (how early each flags the current crowding episode) and
the "3y-window self-catch-up" that suppresses extremes. Production default is untouched.

    python -m positioning.methodology_review     # writes results_positioning_method/*.png + *.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from positioning.adapters import cftc_free, nyfed_free, yields_free
from positioning.config import (FRONT_END, LONG_END, SCORING_CATEGORIES,
                                Z_LOOKBACK_LONG, TRAIN_END)
from positioning.core import aggregate, dv01, schema
from positioning.core import score as score_mod
from positioning.core.normalize import rolling_percentile, rolling_zscore

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results_positioning_method"
OUT.mkdir(parents=True, exist_ok=True)

ORANGE = "#FF8200"; GREY = "#3C3C3C"; RED = "#C00000"; BLUE = "#1F4E79"; GREEN = "#2E7D32"
PURPLE = "#7030A0"; TEAL = "#0F9D9D"
W_LONG = Z_LOOKBACK_LONG      # 156
W_SHORT = 52


# --------------------------------------------------------------------------- #
# Alternative normalizations (all causal, trailing)
# --------------------------------------------------------------------------- #
def ewma_z(s: pd.Series, halflife: int = 26) -> pd.Series:
    m = s.ewm(halflife=halflife, min_periods=halflife).mean()
    v = s.ewm(halflife=halflife, min_periods=halflife).var(bias=False)
    return (s - m) / np.sqrt(v)


def cot_index(s: pd.Series, window: int) -> pd.Series:
    """Classic COT Index min-max stochastic, 0-100 (bounded, saturating)."""
    lo = s.rolling(window, min_periods=window // 2).min()
    hi = s.rolling(window, min_periods=window // 2).max()
    return 100.0 * (s - lo) / (hi - lo).replace(0.0, np.nan)


def build_inputs(force: bool = False):
    """Returns (fut_dv01 panel, production composite series) — the latter is the EXACT
    live output via run_score, so the reference line matches positioning_score.json."""
    fut_raw = cftc_free.fetch_cftc_tff(force_download=force)
    cash_raw = nyfed_free.fetch_nyfed_cash(force_download=force)
    yields_raw = yields_free.fetch_yields(force_download=force)
    res = score_mod.run_score(fut_raw, cash_raw, yields_raw)   # real composite (incl. cash/curve/conc)
    fut = schema.validate_futures_panel(fut_raw)
    yields = schema.validate_yields(yields_raw)
    rd = pd.DatetimeIndex(sorted(fut["date"].unique()))
    ywk = yields.reindex(yields.index.union(rd)).ffill().reindex(rd)
    fd = dv01.add_net_dv01(fut, ywk)
    return fd, res.composite.rename("composite_prod")


def duration_signal(fd: pd.DataFrame) -> pd.Series:
    """The raw underlying: aggregate buy-side (lev+AM) net DV01 level ($mm/bp).
    This is the stock the composite ultimately standardizes."""
    cat = aggregate.category_duration(fd)
    cols = [c for c in SCORING_CATEGORIES if c in cat.columns]
    return (cat[cols].sum(axis=1) / 1e6).rename("buyside_net_dv01")


def open_interest(fd: pd.DataFrame) -> pd.Series:
    one = fd.drop_duplicates(["date", "instrument"])
    oi = one.groupby("date")["open_interest"].sum()
    oi.index = pd.to_datetime(oi.index)
    return oi.sort_index()


# --------------------------------------------------------------------------- #
def responsiveness(series: dict, bounded: dict, current_episode_start="2024-06-01") -> pd.DataFrame:
    """For each normalization: latest value, and lead time to first flag 'crowded' during the
    run-up into the current episode. z-metrics flag at |z|>1.5; bounded (0-100) flag at >80."""
    rows = []
    for name, s in series.items():
        s = s.dropna()
        thr = 80.0 if bounded.get(name) else 1.5
        seg = s.loc[current_episode_start:]
        first = seg[seg > thr].index.min() if (seg > thr).any() else None
        rows.append({"metric": name, "latest": float(s.iloc[-1]),
                     "threshold": thr,
                     "first_flag": None if first is None else str(first.date()),
                     "weeks_since_flag": None if first is None else int((s.index.max() - first).days / 7),
                     "pct_time_extreme": float((s.abs() > thr).mean()) if not bounded.get(name)
                     else float((s > thr).mean())})
    return pd.DataFrame(rows)


def main(force: bool = False):
    fd, comp_prod = build_inputs(force)
    sig = duration_signal(fd)                 # raw buy-side net DV01 ($mm/bp)
    oi = open_interest(fd).reindex(sig.index).ffill()
    comp_prod = comp_prod.reindex(sig.index)

    # ---- composite-level normalizations of the SAME buy-side signal ----
    variants = {
        "prod composite (2x 3y-z + PCA)": comp_prod,
        "3y z (single pass)": rolling_zscore(sig, W_LONG),
        "1y z (single pass)": rolling_zscore(sig, W_SHORT),
        "EWMA z (hl=26w)": ewma_z(sig, 26),
        "%OI then 1y z": rolling_zscore((sig / (oi / 1e6)).replace([np.inf, -np.inf], np.nan), W_SHORT),
        "flow (13w chg) 1y z": rolling_zscore(sig.diff(13), W_SHORT),
    }
    bounded_variants = {
        "COT-index 3y (0-100)": cot_index(sig, W_LONG),
        "COT-index 1y (0-100)": cot_index(sig, W_SHORT),
        "percentile 1y (0-100)": rolling_percentile(sig, W_SHORT) * 100,
    }
    bounded_flag = {**{k: False for k in variants}, **{k: True for k in bounded_variants}}

    resp = responsiveness({**variants, **bounded_variants}, bounded_flag)

    # ---- Figure 1: the 3y self-catch-up (why z is suppressed) ----
    fig, ax = plt.subplots(figsize=(11, 4.4))
    m = sig.rolling(W_LONG, min_periods=W_LONG // 2).mean()
    sd = sig.rolling(W_LONG, min_periods=W_LONG // 2).std(ddof=0)
    ax.plot(sig.index, sig.values, color=BLUE, lw=1.2, label="buy-side net DV01 ($mm/bp)")
    ax.plot(m.index, m.values, color=ORANGE, lw=1.4, label="trailing 3y mean (the moving goalpost)")
    ax.fill_between(sd.index, (m - 2 * sd).values, (m + 2 * sd).values, color=ORANGE, alpha=0.12,
                    label="3y mean ±2σ")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_title("Why the composite is slow: the 3y mean & band drift up INTO the crowd, "
                 "capping the z-score")
    ax.set_ylabel("$mm / bp"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_selfcatchup.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 2: z-style variants overlaid ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colz = {"prod composite (2x 3y-z + PCA)": GREY, "3y z (single pass)": BLUE,
            "1y z (single pass)": ORANGE, "EWMA z (hl=26w)": RED,
            "%OI then 1y z": GREEN, "flow (13w chg) 1y z": TEAL}
    for name, s in variants.items():
        ax.plot(s.index, s.values, lw=1.3 if "prod" in name else 1.0,
                color=colz[name], alpha=0.9, label=name)
    for k in (1.5, -1.5):
        ax.axhline(k, color=GREY, ls="--", lw=0.7)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_title("Same buy-side signal, different normalization — z-family (crowd threshold ±1.5)")
    ax.set_ylabel("z"); ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_variants_z.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 3: bounded 0-100 variants ----
    fig, ax = plt.subplots(figsize=(11, 4.0))
    for name, s, c in (("COT-index 3y (0-100)", bounded_variants["COT-index 3y (0-100)"], BLUE),
                       ("COT-index 1y (0-100)", bounded_variants["COT-index 1y (0-100)"], ORANGE),
                       ("percentile 1y (0-100)", bounded_variants["percentile 1y (0-100)"], PURPLE)):
        ax.plot(s.index, s.values, lw=1.1, color=c, label=name)
    ax.axhline(80, color=RED, ls="--", lw=0.7, label="crowded (>80)")
    ax.axhline(20, color=GREEN, ls="--", lw=0.7)
    ax.set_ylim(0, 100)
    ax.set_title("Bounded normalizations saturate fast — COT-index & percentile (crowd >80)")
    ax.set_ylabel("0–100"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_variants_bounded.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Curve variants (front TU+FV minus long US+UB) ----
    curve = {}
    for cat in SCORING_CATEGORIES:
        cd = aggregate.curve_duration(fd, cat) / 1e6
        curve[cat] = cd
    curve_agg = pd.DataFrame(curve).mean(axis=1)      # raw front-long ($mm/bp), book tilt
    curve_tbl = {
        "raw front-long ($mm/bp)": curve_agg,
        "prod 3y-z (re-z)": rolling_zscore(rolling_zscore(pd.DataFrame(curve), W_LONG).mean(axis=1), W_LONG),
        "1y z": rolling_zscore(curve_agg, W_SHORT),
        "EWMA z (hl=26w)": ewma_z(curve_agg, 26),
        "COT-index 1y": cot_index(curve_agg, W_SHORT),
        "percentile 1y": rolling_percentile(curve_agg, W_SHORT) * 100,
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(curve_agg.index, curve_agg.values, color=GREY, lw=1.2, label="raw front-long ($mm/bp) = BOOK tilt")
    axes[0].axhline(0, color=GREY, lw=0.6); axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title("Curve positioning — BOOK tilt (raw) vs MOMENTUM (normalized)"); axes[0].set_ylabel("$mm/bp")
    for name, c in (("prod 3y-z (re-z)", GREY), ("1y z", ORANGE), ("EWMA z (hl=26w)", RED)):
        s = curve_tbl[name]; axes[1].plot(s.index, s.values, color=c, lw=1.1, label=name)
    for k in (1.5, -1.5):
        axes[1].axhline(k, color=GREY, ls="--", lw=0.7)
    axes[1].set_ylabel("z"); axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_curve_variants.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    out = {
        "as_of": str(sig.dropna().index.max().date()),
        "buyside_net_dv01_latest_mm_bp": float(sig.dropna().iloc[-1]),
        "window_weeks": {"prod_long": W_LONG, "short": W_SHORT},
        "responsiveness": resp.to_dict(orient="records"),
        "curve_latest": {
            "raw_front_minus_long_mm_bp": float(curve_agg.dropna().iloc[-1]),
            "prod_z": float(curve_tbl["prod 3y-z (re-z)"].dropna().iloc[-1]),
            "z_1y": float(curve_tbl["1y z"].dropna().iloc[-1]),
            "cot_index_1y": float(curve_tbl["COT-index 1y"].dropna().iloc[-1]),
        },
    }
    (OUT / "methodology_review.json").write_text(json.dumps(out, indent=2))
    print("RESPONSIVENESS (lead time to flag current crowding):")
    print(resp.to_string(index=False))
    print(f"\nWrote figures + methodology_review.json to {OUT}")
    return out


if __name__ == "__main__":
    main()
