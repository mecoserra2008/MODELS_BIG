"""Data layer for the app: cached fetch + score, plus derived per-asset board frames.

`load_score` is cached on the methodology params so sidebar changes recompute cleanly.
The per-asset helpers turn the core outputs into the tile rows the Bull/Bear board needs:
each row carries a *positioning* read (buy-side crowding) and a contrarian *signal*.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from positioning.adapters import cftc_free, nyfed_free, yields_free  # noqa: E402
from positioning.config import (  # noqa: E402
    CONTRACT_BY_KEY, CONTRACTS, FRONT_END, LONG_END, Z_LOOKBACK_LONG,
)
from positioning.core import score as score_mod  # noqa: E402
from positioning.core.normalize import rolling_zscore  # noqa: E402

BELLY = ("TY", "TN")
INSTRUMENT_ORDER = [c.key for c in CONTRACTS]
BUYSIDE = ("lev_money", "asset_mgr")


# --------------------------------------------------------------------------- #
# Cached load
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _fetch_raw(start: str, force: bool):
    fut = cftc_free.fetch_cftc_tff(start=start, force_download=force)
    cash = nyfed_free.fetch_nyfed_cash(start=start, force_download=force)
    yields = yields_free.fetch_yields(start=start, force_download=force)
    return fut, cash, yields


@st.cache_data(show_spinner=False)
def load_score(start: str, train_end: str, horizon: int, k_entry: float,
               k_exit: float, force: bool = False):
    """Fetch (cached) and score (cached on params). Returns a PositioningScore."""
    fut, cash, yields = _fetch_raw(start, force)
    return score_mod.run_score(
        fut, cash, yields, train_end=train_end, forward_horizon=horizon,
        k_entry=k_entry, k_exit=k_exit,
    )


def freshness(res) -> str:
    last = pd.to_datetime(res.composite.dropna().index.max())
    latest_report = pd.to_datetime(res.fut_dv01["date"].max())
    return f"CFTC report {latest_report.date()} · score {last.date()}"


@st.cache_data(show_spinner=False, ttl=900)
def load_sentiment(use_gdelt: bool = True, use_rss: bool = True):
    """News + FinBERT sentiment (1D/1W/1M). Cached 15 min (news is slow-moving)."""
    from positioning.sentiment import aggregate
    return aggregate.compute(use_gdelt=use_gdelt, use_rss=use_rss)


@st.cache_data(show_spinner=False, ttl=3600)
def load_extra_indicators():
    """The four standalone extra positioning indicators. Cached 1h."""
    from positioning.adapters import extra_free
    return extra_free.all_indicators()


# --------------------------------------------------------------------------- #
# Per-asset board rows
# --------------------------------------------------------------------------- #
@dataclass
class AssetRow:
    name: str
    sub: str
    z: float                 # buy-side net-DV01 crowding z (latest)
    positioning: str         # "long" | "short" | "light"
    signal: str              # "bull" | "bear" | "neutral"
    spark: pd.Series         # recent z history


def _buyside_dv01_by_instrument(res) -> pd.DataFrame:
    """Wide [date x instrument] buy-side (lev+asset mgr) net DV01."""
    d = res.fut_dv01[res.fut_dv01["category"].isin(BUYSIDE)]
    return (d.groupby(["date", "instrument"])["net_dv01"].sum()
            .unstack("instrument").sort_index())


def _classify(z: float, pos_thr: float, sig_thr: float) -> tuple[str, str]:
    positioning = "long" if z > pos_thr else "short" if z < -pos_thr else "light"
    # contrarian: crowded long -> fade -> bearish duration
    signal = "bear" if z > sig_thr else "bull" if z < -sig_thr else "neutral"
    return positioning, signal


def _row(name: str, sub: str, series: pd.Series, pos_thr: float,
         sig_thr: float) -> AssetRow:
    z = rolling_zscore(series, Z_LOOKBACK_LONG)
    latest = float(z.dropna().iloc[-1]) if z.notna().any() else float("nan")
    positioning, signal = _classify(latest, pos_thr, sig_thr)
    return AssetRow(name, sub, latest, positioning, signal, z.dropna().tail(52))


def board_rows(res, pos_thr: float = 0.5, sig_thr: float = 1.0) -> dict[str, list[AssetRow]]:
    """Three tiers of tiles: contracts, curve buckets, segments."""
    bs = _buyside_dv01_by_instrument(res)

    contracts = [
        _row(k, f"{CONTRACT_BY_KEY[k].label} · {CONTRACT_BY_KEY[k].tenor_years:g}y",
             bs[k], pos_thr, sig_thr)
        for k in INSTRUMENT_ORDER if k in bs.columns
    ]

    def bucket(keys):
        cols = [k for k in keys if k in bs.columns]
        return bs[cols].sum(axis=1)

    buckets = [
        _row("Front-end", "TU · FV", bucket(FRONT_END), pos_thr, sig_thr),
        _row("Belly", "TY · TN", bucket(BELLY), pos_thr, sig_thr),
        _row("Long-end", "US · UB", bucket(LONG_END), pos_thr, sig_thr),
    ]

    # segments: by trader type + cash (dealer inventory)
    def cat_dv01(cat):
        d = res.fut_dv01[res.fut_dv01["category"] == cat]
        return d.groupby("date")["net_dv01"].sum().sort_index()

    segments = [
        _row("Leveraged Funds", "fast money", cat_dv01("lev_money"), pos_thr, sig_thr),
        _row("Asset Managers", "real money", cat_dv01("asset_mgr"), pos_thr, sig_thr),
    ]
    if "cash_z" in res.feature_panel.columns:
        cashz = res.feature_panel["cash_z"].dropna()
        latest = float(cashz.iloc[-1]) if len(cashz) else float("nan")
        positioning, signal = _classify(latest, pos_thr, sig_thr)
        segments.append(AssetRow("Primary Dealers", "cash Treasuries", latest,
                                 positioning, signal, cashz.tail(52)))

    return {"contracts": contracts, "buckets": buckets, "segments": segments}


def board_table(rows: dict[str, list[AssetRow]]) -> pd.DataFrame:
    """Flat table view (CVD relief + desk sortability)."""
    pos_txt = {"long": "Crowded Long", "short": "Crowded Short", "light": "Light"}
    sig_txt = {"bull": "▲ Bullish", "bear": "▼ Bearish", "neutral": "● Neutral"}
    recs = []
    for tier, items in rows.items():
        for r in items:
            recs.append({"Tier": tier, "Asset": r.name, "Detail": r.sub,
                         "Crowding z": round(r.z, 2),
                         "Positioning": pos_txt[r.positioning],
                         "Signal": sig_txt[r.signal]})
    return pd.DataFrame(recs)
