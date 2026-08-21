"""Fixed shared interface for positioning-composite versions.

Every version module implements `build(inp: VersionInputs) -> VersionResult` using ONLY the
fields on `VersionInputs` and the normalizer helpers here, so versions are interchangeable and
comparable. Data is reconstructed from positioning.core (cache-first); the live production
composite is exposed as `prod_composite` for reference. Nothing here mutates positioning.core.

Sign convention for every `composite`: POSITIVE = crowded net LONG duration (matches the live
composite). z-family versions are unbounded (roughly N(0,1)); bounded versions live in [0,100]
with `bounded=True` and use 80/20 crowd thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from positioning.adapters import cftc_free, nyfed_free, yields_free
from positioning.config import (COT_PUBLICATION_LAG_BDAYS, FRONT_END, LONG_END,
                                SCORING_CATEGORIES)
from positioning.core import aggregate, dv01, schema
from positioning.core import score as score_mod
from positioning.core.normalize import rolling_percentile, rolling_zscore

W_LONG = 156     # 3y (production window)
W_SHORT = 52     # 1y (Street default)
CURRENT_EPISODE_START = "2024-06-01"


# --------------------------------------------------------------------------- #
# Inputs bundle (built once, shared by every version)
# --------------------------------------------------------------------------- #
@dataclass
class VersionInputs:
    cat_dv01: pd.DataFrame       # [date x category] net DV01 ($mm/bp), buy-side categories
    contract_dv01: pd.DataFrame  # [date x (category,instrument)] net DV01 ($mm/bp)
    oi: pd.Series                # total open interest by date (contracts)
    front_long: pd.DataFrame     # [date x category] front(TU+FV) - long(US+UB) net DV01 ($mm/bp)
    slope: pd.Series             # weekly 2s10s (DGS10-DGS2), report-dated (%)
    prod_composite: pd.Series    # live production composite (reference)
    report_dates: pd.DatetimeIndex
    y10_w: pd.Series             # weekly DGS10 (%) on the report grid (for the conditional panel)
    fut_raw: object = None       # raw CFTC frame (so V0 can re-run run_score with edited params)
    cash_raw: object = None      # raw NY Fed cash frame
    yields_raw: object = None    # raw FRED yields frame

    def buyside_level(self) -> pd.Series:
        """Aggregate buy-side (lev+AM) net DV01 level ($mm/bp) — the core crowding stock."""
        cols = [c for c in SCORING_CATEGORIES if c in self.cat_dv01.columns]
        return self.cat_dv01[cols].sum(axis=1).rename("buyside_net_dv01")

    def buyside_front_long(self) -> pd.Series:
        """Aggregate buy-side front-minus-long DV01 ($mm/bp) — the raw curve book tilt."""
        cols = [c for c in SCORING_CATEGORIES if c in self.front_long.columns]
        return self.front_long[cols].mean(axis=1).rename("front_minus_long")


@dataclass
class VersionResult:
    name: str
    description: str
    composite: pd.Series          # POSITIVE = crowded net long duration
    bounded: bool = False         # True => composite in [0,100] (crowd>80)
    curve: pd.Series | None = None  # optional curve-positioning series (+ = steepener)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Shared normalizer helpers (versions should reuse these)
# --------------------------------------------------------------------------- #
def rolling_z(s: pd.Series, window: int) -> pd.Series:
    """Trailing rolling z-score (re-exports the core implementation)."""
    return rolling_zscore(s, window)


def ewma_z(s: pd.Series, halflife: int = 26) -> pd.Series:
    """Exponentially-weighted z: recency-weighted mean/vol, causal."""
    m = s.ewm(halflife=halflife, min_periods=halflife).mean()
    v = s.ewm(halflife=halflife, min_periods=halflife).var(bias=False)
    return (s - m) / np.sqrt(v.replace(0.0, np.nan))


def cot_index(s: pd.Series, window: int) -> pd.Series:
    """Classic COT Index min-max stochastic in [0,100] (bounded; saturates)."""
    lo = s.rolling(window, min_periods=window // 2).min()
    hi = s.rolling(window, min_periods=window // 2).max()
    return 100.0 * (s - lo) / (hi - lo).replace(0.0, np.nan)


def percentile01(s: pd.Series, window: int) -> pd.Series:
    """Trailing percentile in [0,100]."""
    return rolling_percentile(s, window) * 100.0


def pct_of_oi(level_mm_bp: pd.Series, oi: pd.Series) -> pd.Series:
    """Normalize a $mm/bp level by total open interest (market-size invariant)."""
    o = oi.reindex(level_mm_bp.index).ffill()
    return (level_mm_bp / (o / 1e6)).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_inputs(force: bool = False, start: str | None = None) -> VersionInputs:
    kw = {"force_download": force}
    if start is not None:
        kw["start"] = start
    fut_raw = cftc_free.fetch_cftc_tff(**kw)
    cash_raw = nyfed_free.fetch_nyfed_cash(**kw)
    yields_raw = yields_free.fetch_yields(force_download=force)

    res = score_mod.run_score(fut_raw, cash_raw, yields_raw)   # production reference
    fut = schema.validate_futures_panel(fut_raw)
    yields = schema.validate_yields(yields_raw)
    rd = pd.DatetimeIndex(sorted(fut["date"].unique()))
    ywk = yields.reindex(yields.index.union(rd)).ffill().reindex(rd)
    fd = dv01.add_net_dv01(fut, ywk)

    cat = aggregate.category_duration(fd) / 1e6
    contract = fd.pivot_table(index="date", columns=["category", "instrument"],
                              values="net_dv01", aggfunc="sum").sort_index() / 1e6
    contract.index = pd.to_datetime(contract.index)

    one = fd.drop_duplicates(["date", "instrument"])
    oi = one.groupby("date")["open_interest"].sum()
    oi.index = pd.to_datetime(oi.index); oi = oi.sort_index()

    fl = {c: (aggregate.curve_duration(fd, c) / 1e6) for c in SCORING_CATEGORIES}
    front_long = pd.DataFrame(fl)

    slope = (ywk["DGS10"] - ywk["DGS2"]).rename("slope_2s10s") if "DGS2" in ywk.columns \
        else pd.Series(index=rd, dtype=float)
    y10_w = ywk["DGS10"].rename("y10_w")

    return VersionInputs(cat_dv01=cat, contract_dv01=contract, oi=oi, front_long=front_long,
                         slope=slope, prod_composite=res.composite.rename("prod"),
                         report_dates=rd, y10_w=y10_w,
                         fut_raw=fut_raw, cash_raw=cash_raw, yields_raw=yields_raw)


def pub_lag(s: pd.Series) -> pd.Series:
    """Publication-lag a weekly report-dated series (report + COT lag), mirroring score.py,
    so predictive/conditional reads never peek ahead."""
    return pd.Series(s.values,
                     index=s.index + pd.tseries.offsets.BDay(COT_PUBLICATION_LAG_BDAYS),
                     name=s.name)


# --------------------------------------------------------------------------- #
# Responsiveness metric (shared, for the comparison)
# --------------------------------------------------------------------------- #
def responsiveness(series: pd.Series, bounded: bool,
                   episode_start: str = CURRENT_EPISODE_START) -> dict:
    """Latest value + when the series first flagged 'crowded' during the current build-up.
    z-family flags at |z|>1.5; bounded (0-100) flags at >80."""
    s = series.dropna()
    if s.empty:
        return {"latest": None, "threshold": None, "first_flag": None, "weeks_since_flag": None,
                "pct_time_extreme": None}
    thr = 80.0 if bounded else 1.5
    seg = s.loc[episode_start:]
    hits = (seg > thr) if bounded else (seg.abs() > thr)
    first = hits[hits].index.min() if hits.any() else None
    pct = float((s > thr).mean()) if bounded else float((s.abs() > thr).mean())
    return {"latest": float(s.iloc[-1]), "threshold": thr,
            "first_flag": None if first is None else str(first.date()),
            "weeks_since_flag": None if first is None else int((s.index.max() - first).days / 7),
            "pct_time_extreme": pct}
