"""
10s5s Curve RV Model — Kalman + Bollinger signals with DV01 translation.

Estimates a daily mean-reversion model for the 10Y-5Y slope on several
sovereign curves (US, EU-AAA/Bund, Italy, France, Spain, UK, Canada), using:

  * a time-varying (alpha_t, beta_t) Kalman hedge ratio between the legs,
  * a local-linear-trend Kalman for the spread's dynamic fair value mu_t,
  * rolling Bollinger / z-score bands for steepener / flattener signals,

and translates the signals into DV01-neutral leg sizing plus a structural
net-long duration-allocation table.

Fit once on history, then re-run forward on live data:

    python run_10s5s.py            # full estimation + backtest + artifacts
    python run_10s5s.py --force    # ignore caches, re-download all data
    python run_10s5s.py --live     # reuse frozen params, refresh today's signal

Artifacts land in results_10s5s/ (figures, JSON, model_params.json, signals_live.csv).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # benign statsmodels MLE / pandas notices

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:  # seaborn optional for styling only
    pass

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

# FRED key for US Treasuries lives in alternative_models/.env; surface it before
# data_retrieval fetches (its own load_dotenv only sees the repo-root .env).
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / "alternative_models" / ".env")
except Exception:
    pass

from data_retrieval import (  # noqa: E402
    fetch_us_yields,
    fetch_eu_yields,
    fetch_egb_yields,
    fetch_uk_yields,
    fetch_ca_yields,
    align_yield_data,
)
import curve_signals as cs  # noqa: E402
import dv01 as dv  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RESULTS = _ROOT / "results_10s5s"
RESULTS.mkdir(exist_ok=True)

START = os.environ.get("CURVE_START", "2010-01-01")
END = os.environ.get("CURVE_END", "2026-07-01")
WINDOW = int(os.environ.get("CURVE_WINDOW", "60"))       # rolling band window (days)
# Candidate windows for the --sweep comparison (~3m/4.5m/6m/9m/1y/2y).
WINDOWS = [int(w) for w in os.environ.get(
    "CURVE_WINDOWS", "60,90,120,180,252,504").split(",") if w.strip()]
K_ENTRY = float(os.environ.get("CURVE_K_ENTRY", "2.0"))  # z entry threshold
K_EXIT = float(os.environ.get("CURVE_K_EXIT", "0.5"))    # z exit threshold
TVP_DELTA = float(os.environ.get("CURVE_TVP_DELTA", "1e-5"))
TCOST_BP = float(os.environ.get("CURVE_TCOST_BP", "0.5"))  # per position change
TRAIN_END = os.environ.get("CURVE_TRAIN_END", "2022-12-31")
STRUCTURAL_DV01 = float(os.environ.get("CURVE_STRUCT_DV01", "100000"))  # net long target

# Colour vocabulary (repo convention)
C_GREEN, C_RED, C_BLUE, C_ORANGE, C_GREY = (
    "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#bdbdbd",
)

# Markets: prefix -> label. Legs are <prefix>_5Y / <prefix>_10Y after alignment.
MARKETS = {
    "US": "US Treasuries",
    "EU": "Bund (EU AAA)",
    "IT": "Italy BTP",
    "FR": "France OAT",
    "ES": "Spain Bonos",
    "UK": "UK Gilts",
    "CA": "Canada",
}


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def load_all_yields(force: bool) -> pd.DataFrame:
    print("=" * 60)
    print("1. DATA — fetching / loading yield curves")
    print("=" * 60)
    frames = []

    def _try(name, fn):
        try:
            df = fn()
            print(f"  {name}: {df.shape[0]} rows, cols={list(df.columns)}")
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: {name} fetch failed ({e}) — skipped")

    _try("US", lambda: fetch_us_yields(START, END, force_download=force))
    _try("EU", lambda: fetch_eu_yields(START, END, force_download=force))
    _try("EGB", lambda: fetch_egb_yields(START, END, force_download=force))
    _try("UK", lambda: fetch_uk_yields(START, END, maturities=[5, 10],
                                       force_download=force))
    _try("CA", lambda: fetch_ca_yields(START, END, force_download=force))

    if not frames:
        raise RuntimeError("No yield data could be loaded from any source.")
    combined = align_yield_data(*frames, ffill_limit=5)
    print(f"  aligned panel: {combined.shape[0]} rows x {combined.shape[1]} cols")
    return combined


# ---------------------------------------------------------------------------
# 2-3. SPREADS + SIGNALS per market
# ---------------------------------------------------------------------------
def signals_for_window(spread: pd.Series, window: int) -> tuple:
    """Window-dependent block: (bands, z vs rolling mean, positions).

    Shared by build_market and the window sweep so the signal logic lives in
    exactly one place. The two Kalman filters do NOT depend on the window and
    are computed separately once per market.
    """
    bands = cs.bollinger_bands(spread, window=window, k=K_ENTRY)
    z = cs.zscore(spread, window=window)
    signal = cs.generate_signals(z, k_entry=K_ENTRY, k_exit=K_EXIT)
    return bands, z, signal


def build_market(prefix: str, panel: pd.DataFrame,
                 frozen: dict | None = None) -> dict | None:
    """Compute spread, Kalman fair value, hedge ratio, z-score and signals."""
    c5, c10 = f"{prefix}_5Y", f"{prefix}_10Y"
    if c5 not in panel.columns or c10 not in panel.columns:
        print(f"  {prefix}: legs missing ({c5}/{c10}) — dropped")
        return None
    df = panel[[c5, c10]].dropna()
    if len(df) < max(120, WINDOW + 30):
        print(f"  {prefix}: too few points ({len(df)}) — dropped")
        return None

    y5, y10 = df[c5], df[c10]
    spread = (y10 - y5).rename("spread")            # percentage points

    # Kalman #1 — time-varying hedge ratio between legs
    tvp = cs.kalman_tvp_regression(y10, y5, delta=TVP_DELTA)

    # Kalman #2 — local-linear-trend fair value of the spread
    lt_params = (frozen or {}).get("lt_params") if frozen else None
    lt = cs.kalman_local_linear_trend(spread, fixed_params=lt_params)

    # Signals: z-score of the spread vs its rolling mean (the Bollinger mid).
    # NOTE: the *filtered* Kalman level chases the observation, so a z-score
    # against it is degenerate (~0) and never fires — the rolling-mean center is
    # the tradable one and is exactly the "rolling Bollinger band" requested.
    # The Kalman fair value (lt.level) is kept as an adaptive overlay and the
    # TVP beta as the DV01 hedge-ratio cross-check.
    bands, z, signal = signals_for_window(spread, WINDOW)
    half_life = cs.ou_half_life((spread - bands["mid"]).dropna())

    return {
        "prefix": prefix,
        "spread": spread,
        "y5": y5, "y10": y10,
        "beta": tvp.beta,
        "level": lt.level,
        "z": z,
        "bands": bands,
        "signal": signal,
        "half_life": half_life,
        "lt_params": lt.params,
    }


# ---------------------------------------------------------------------------
# 4. BACKTEST — DV01-neutral spread PnL (self-contained, causal)
# ---------------------------------------------------------------------------
def backtest(m: dict) -> dict:
    """One-day-lagged position PnL in bp of spread; metrics per market."""
    spread_bp = m["spread"] * 100.0                  # pp -> bp
    pos = m["signal"].reindex(spread_bp.index).ffill().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)               # enter at close, earn next day
    dspread = spread_bp.diff()
    gross = pos_lag * dspread
    turnover = pos.diff().abs().fillna(0.0)
    costs = turnover * TCOST_BP
    pnl = (gross - costs).dropna()

    if pnl.std(ddof=0) > 0:
        sharpe = float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = float("nan")
    active = pnl[pos_lag.reindex(pnl.index) != 0]
    hit = float((active > 0).mean()) if len(active) else float("nan")
    equity = pnl.cumsum()
    dd = float((equity - equity.cummax()).min())

    # average holding period = mean run length of nonzero positions
    runs, cur = [], 0
    for p in pos.to_numpy():
        if p != 0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    avg_hold = float(np.mean(runs)) if runs else 0.0

    return {
        "pnl": pnl, "equity": equity,
        "sharpe": sharpe, "hit_rate": hit,
        "max_drawdown_bp": dd, "avg_hold_days": avg_hold,
        "n_trades": int((turnover > 0).sum()),
    }


def _count_entries(signal: pd.Series) -> int:
    """Number of new positions opened (flat -> nonzero transitions)."""
    s = signal.fillna(0.0).to_numpy()
    prev = np.concatenate([[0.0], s[:-1]])
    return int(np.sum((prev == 0.0) & (s != 0.0)))


# ---------------------------------------------------------------------------
# WINDOW SWEEP — compare rolling-window sizes at fixed k-thresholds
# ---------------------------------------------------------------------------
def sweep_windows(markets_base: dict, windows: list[int]) -> dict:
    """
    For each candidate window x market, recompute only the window-dependent
    signal block (Kalmans are window-independent and already in markets_base)
    and score it. Returns {window: {"per_market": {...}, "average": {...}}}.

    Metric of interest for the user: ``trades_per_year`` (how often it fires)
    and ``pct_in_position`` (how much of the time it holds a view), alongside
    Sharpe / hit-rate / avg-hold as quality context.
    """
    out = {}
    for w in windows:
        per_market = {}
        for pfx, base in markets_base.items():
            spread = base["spread"]
            years = max((spread.index[-1] - spread.index[0]).days / 365.25, 1e-9)
            bands, z, signal = signals_for_window(spread, w)
            bt = backtest({"spread": spread, "signal": signal})
            entries = _count_entries(signal)
            per_market[pfx] = {
                "trades_per_year": round(entries / years, 2),
                "pct_in_position": round(float((signal != 0).mean()) * 100, 1),
                "avg_hold_days": round(bt["avg_hold_days"], 1),
                "sharpe": round(bt["sharpe"], 3),
                "hit_rate": round(bt["hit_rate"], 3),
                "max_drawdown_bp": round(bt["max_drawdown_bp"], 1),
                "n_entries": entries,
            }
        keys = ["trades_per_year", "pct_in_position", "avg_hold_days",
                "sharpe", "hit_rate", "max_drawdown_bp"]
        avg = {k: round(float(np.nanmean([per_market[p][k] for p in per_market])), 3)
               for k in keys}
        out[w] = {"per_market": per_market, "average": avg}
    return out


def print_sweep_table(sweep: dict):
    print("=" * 72)
    print("WINDOW COMPARISON — mean across markets (k_entry=%.1f, k_exit=%.1f)"
          % (K_ENTRY, K_EXIT))
    print("=" * 72)
    hdr = f"{'window':>7} {'trades/yr':>10} {'%in-pos':>8} {'avgHold':>8} " \
          f"{'Sharpe':>7} {'hit':>6} {'maxDD':>8}"
    print(hdr)
    print("-" * len(hdr))
    for w, d in sweep.items():
        a = d["average"]
        print(f"{w:>7} {a['trades_per_year']:>10.2f} {a['pct_in_position']:>8.1f} "
              f"{a['avg_hold_days']:>8.1f} {a['sharpe']:>7.3f} {a['hit_rate']:>6.3f} "
              f"{a['max_drawdown_bp']:>8.1f}")
    # per-market trades/yr detail
    mkts = list(next(iter(sweep.values()))["per_market"].keys())
    print("\ntrades/yr by market:")
    print(f"{'window':>7} " + " ".join(f"{m:>8}" for m in mkts))
    for w, d in sweep.items():
        pm = d["per_market"]
        print(f"{w:>7} " + " ".join(f"{pm[m]['trades_per_year']:>8.2f}" for m in mkts))


def plot_window_comparison(sweep: dict):
    windows = list(sweep.keys())
    mkts = list(next(iter(sweep.values()))["per_market"].keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _series(metric, mkt=None):
        if mkt is None:
            return [sweep[w]["average"][metric] for w in windows]
        return [sweep[w]["per_market"][mkt][metric] for w in windows]

    panels = [
        ("trades_per_year", "Signals opened per year", C_BLUE),
        ("pct_in_position", "% of time in a position", C_ORANGE),
        ("sharpe", "Backtest Sharpe", C_GREEN),
    ]
    for ax, (metric, title, colour) in zip(axes, panels):
        for m in mkts:
            ax.plot(windows, _series(metric, m), color=C_GREY, lw=0.9,
                    alpha=0.6, marker="o", ms=3)
        ax.plot(windows, _series(metric), color=colour, lw=2.4, marker="o",
                label="mean across markets")
        ax.set_title(title)
        ax.set_xlabel("rolling window (trading days)")
        ax.legend(fontsize=8)
        if metric == "sharpe":
            ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("10s5s rolling-window comparison (faint = per market, bold = mean)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_window_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 6. FIGURES
# ---------------------------------------------------------------------------
def plot_spread(m: dict, label: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    sp = m["spread"] * 100.0
    lv = m["level"] * 100.0
    band = m["bands"] * 100.0
    ax.fill_between(band.index, band["lower"], band["upper"],
                    color=C_GREY, alpha=0.25, label=f"±{K_ENTRY}σ band")
    ax.plot(band.index, band["mid"], color=C_ORANGE, lw=1.4,
            label="rolling-mean fair value")
    ax.plot(lv.index, lv, color="#7f3fbf", lw=0.8, ls=":", alpha=0.8,
            label="Kalman level (filtered)")
    ax.plot(sp.index, sp, color=C_BLUE, lw=1.0, label="10s5s spread (bp)",
            zorder=5)
    pos = m["signal"].reindex(sp.index).fillna(0.0)
    ax.fill_between(sp.index, sp.min(), sp.max(), where=pos > 0,
                    color=C_GREEN, alpha=0.08)
    ax.fill_between(sp.index, sp.min(), sp.max(), where=pos < 0,
                    color=C_RED, alpha=0.08)
    ax.set_title(f"{label} — 10s5s spread, Kalman fair value & signals")
    ax.set_ylabel("bp"); ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / f"fig_{m['prefix']}_spread_kalman.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_zscore(m: dict, label: str):
    fig, ax = plt.subplots(figsize=(13, 4))
    z = m["z"]
    ax.plot(z.index, z, color=C_BLUE, lw=0.9)
    for lvl, c in [(K_ENTRY, C_RED), (-K_ENTRY, C_GREEN),
                   (K_EXIT, C_GREY), (-K_EXIT, C_GREY)]:
        ax.axhline(lvl, color=c, ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{label} — z-score vs Kalman fair value")
    ax.set_ylabel("z"); plt.tight_layout()
    plt.savefig(RESULTS / f"fig_{m['prefix']}_zscore.png",
                dpi=200, bbox_inches="tight")
    plt.close()


def plot_beta(markets: dict):
    fig, ax = plt.subplots(figsize=(13, 5))
    for m in markets.values():
        ax.plot(m["beta"].index, m["beta"], lw=1.0, label=m["prefix"])
    ax.set_title("Time-varying hedge ratio β_t (Δy10 / Δy5) by market")
    ax.set_ylabel("β_t"); ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_beta_t.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_equity(bts: dict):
    fig, ax = plt.subplots(figsize=(13, 5))
    total = None
    for pfx, bt in bts.items():
        ax.plot(bt["equity"].index, bt["equity"], lw=1.0, label=pfx)
        total = bt["equity"] if total is None else total.add(bt["equity"], fill_value=0)
    if total is not None:
        ax.plot(total.index, total, color="k", lw=2.0, label="COMBINED")
    ax.set_title("Backtest cumulative PnL (bp of spread, DV01-neutral)")
    ax.set_ylabel("cumulative bp"); ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_backtest_equity.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_signal_heatmap(markets: dict):
    rows = [(m["prefix"], float(m["z"].dropna().iloc[-1]) if m["z"].dropna().size else np.nan)
            for m in markets.values()]
    labels = [r[0] for r in rows]
    vals = np.array([[r[1] for r in rows]])
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 2.2))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks([0]); ax.set_yticklabels(["z today"])
    for j, v in enumerate(vals[0]):
        ax.text(j, 0, f"{v:.1f}" if v == v else "—", ha="center", va="center",
                fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="z-score")
    ax.set_title("Current 10s5s z-score across markets")
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_signal_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="10s5s curve RV model")
    ap.add_argument("--force", action="store_true", help="re-download all data")
    ap.add_argument("--live", action="store_true",
                    help="reuse frozen params, refresh today's signal only")
    ap.add_argument("--sweep", action="store_true",
                    help="compare rolling windows (WINDOWS) and exit")
    args = ap.parse_args()

    panel = load_all_yields(force=args.force)

    # --sweep: build each market's spread once, compare candidate windows, exit.
    if args.sweep:
        markets_base = {}
        for pfx in MARKETS:
            c5, c10 = f"{pfx}_5Y", f"{pfx}_10Y"
            if c5 not in panel.columns or c10 not in panel.columns:
                continue
            df = panel[[c5, c10]].dropna()
            if len(df) < max(120, max(WINDOWS) + 30):
                print(f"  {pfx}: too few points for max window — skipped")
                continue
            markets_base[pfx] = {"spread": (df[c10] - df[c5]).rename("spread")}
        if not markets_base:
            raise RuntimeError("No market has enough history for the window sweep.")
        print(f"Sweeping windows {WINDOWS} over markets {list(markets_base)}")
        sweep = sweep_windows(markets_base, WINDOWS)
        print_sweep_table(sweep)
        plot_window_comparison(sweep)
        (RESULTS / "window_comparison.json").write_text(
            json.dumps(sweep, indent=2, default=str))
        print(f"\nWrote {RESULTS}/window_comparison.json and fig_window_comparison.png")
        print("Pick a global window, then set CURVE_WINDOW=<w> and re-run the pipeline.")
        return

    frozen_all = None
    if args.live:
        pfile = RESULTS / "model_params.json"
        if pfile.exists():
            frozen_all = json.loads(pfile.read_text())
            print(f"LIVE MODE — loaded frozen params for {list(frozen_all)}")
        else:
            print("LIVE MODE requested but model_params.json missing — full fit")

    print("=" * 60)
    print("2-3. SPREADS + SIGNALS")
    print("=" * 60)
    markets, params_out = {}, {}
    for pfx in MARKETS:
        frozen = (frozen_all or {}).get(pfx)
        m = build_market(pfx, panel, frozen=frozen)
        if m is None:
            continue
        markets[pfx] = m
        params_out[pfx] = {
            "lt_params": {k: float(v) for k, v in m["lt_params"].items()},
            "window": WINDOW, "k_entry": K_ENTRY, "k_exit": K_EXIT,
            "tvp_delta": TVP_DELTA,
        }
        z_last = m["z"].dropna()
        sig_last = int(m["signal"].dropna().iloc[-1]) if m["signal"].dropna().size else 0
        print(f"  {pfx}: half-life={m['half_life']:.1f}d  "
              f"z={z_last.iloc[-1]:.2f}  β={m['beta'].iloc[-1]:.2f}  signal={sig_last:+d}"
              if z_last.size else f"  {pfx}: insufficient z history")

    if not markets:
        raise RuntimeError("No market produced a valid signal series.")

    print("=" * 60)
    print("4. BACKTEST")
    print("=" * 60)
    bts = {}
    for pfx, m in markets.items():
        bt = backtest(m)
        bts[pfx] = bt
        print(f"  {pfx}: Sharpe={bt['sharpe']:.2f}  hit={bt['hit_rate']:.2f}  "
              f"maxDD={bt['max_drawdown_bp']:.1f}bp  avgHold={bt['avg_hold_days']:.0f}d  "
              f"trades={bt['n_trades']}")

    print("=" * 60)
    print("5. DV01 TRANSLATION")
    print("=" * 60)
    market_signals = {}
    for pfx, m in markets.items():
        sig = int(m["signal"].dropna().iloc[-1]) if m["signal"].dropna().size else 0
        z = float(m["z"].dropna().iloc[-1]) if m["z"].dropna().size else np.nan
        market_signals[pfx] = {
            "signal": sig, "z": z,
            "y5": float(m["y5"].iloc[-1]), "y10": float(m["y10"].iloc[-1]),
        }
    alloc = dv.duration_allocation(STRUCTURAL_DV01, market_signals)
    print(alloc.to_string())

    # per-market DV01-neutral leg sizing at the current signal
    leg_sizing = {}
    for pfx, info in market_signals.items():
        direction = info["signal"] if info["signal"] != 0 else 1
        w = dv.dv01_neutral_weights(info["y5"], info["y10"], direction=direction)
        leg_sizing[pfx] = {
            "notional_5y": round(w.notional_5y, 0),
            "notional_10y": round(w.notional_10y, 0),
            "dv01_neutral_hedge_ratio": round(w.hedge_ratio, 3),
            "empirical_beta": round(float(markets[pfx]["beta"].iloc[-1]), 3),
        }

    print("=" * 60)
    print("6. ARTIFACTS")
    print("=" * 60)
    for pfx, m in markets.items():
        plot_spread(m, MARKETS[pfx]); plot_zscore(m, MARKETS[pfx])
    plot_beta(markets); plot_equity(bts); plot_signal_heatmap(markets)

    # JSON results
    results = {"config": {"window": WINDOW, "k_entry": K_ENTRY, "k_exit": K_EXIT,
                          "train_end": TRAIN_END, "structural_dv01": STRUCTURAL_DV01},
               "markets": {}}
    for pfx, m in markets.items():
        bt = bts[pfx]
        results["markets"][pfx] = {
            "label": MARKETS[pfx],
            "half_life_days": m["half_life"],
            "z_today": market_signals[pfx]["z"],
            "signal_today": market_signals[pfx]["signal"],
            "beta_today": float(m["beta"].iloc[-1]),
            "sharpe": bt["sharpe"], "hit_rate": bt["hit_rate"],
            "max_drawdown_bp": bt["max_drawdown_bp"],
            "avg_hold_days": bt["avg_hold_days"], "n_trades": bt["n_trades"],
            "leg_sizing": leg_sizing[pfx],
        }
    results["duration_allocation"] = alloc.reset_index().to_dict(orient="records")
    (RESULTS / "curve_10s5s_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    # frozen params for live mode
    (RESULTS / "model_params.json").write_text(
        json.dumps(params_out, indent=2, default=str))

    # live signal CSV (the refresh-on-live-data hook)
    live_rows = []
    for pfx, m in markets.items():
        ls = leg_sizing[pfx]
        arow = alloc.loc[pfx] if pfx in alloc.index else None
        live_rows.append({
            "market": pfx, "label": MARKETS[pfx],
            "asof": str(m["spread"].index[-1].date()),
            "spread_bp": round(float(m["spread"].iloc[-1]) * 100, 1),
            # fair value = rolling-mean band centre (what z is measured against)
            "fair_value_bp": round(float(m["bands"]["mid"].iloc[-1]) * 100, 1),
            "z": round(market_signals[pfx]["z"], 2),
            "signal": market_signals[pfx]["signal"],
            "action": {1: "STEEPENER", -1: "FLATTENER", 0: "FLAT"}[market_signals[pfx]["signal"]],
            "notional_5y": ls["notional_5y"], "notional_10y": ls["notional_10y"],
            "dv01_5y": None if arow is None else arow["dv01_5y"],
            "dv01_10y": None if arow is None else arow["dv01_10y"],
        })
    pd.DataFrame(live_rows).to_csv(RESULTS / "signals_live.csv", index=False)

    print(f"\nDone. Artifacts written to {RESULTS}/")
    print(f"  markets: {list(markets)}")


if __name__ == "__main__":
    main()
