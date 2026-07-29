"""Data layer for the positioning<->yields study and the US10Y forecaster.

All series are free/open and reuse the positioning subsystem's adapters + scoring core:
  * weekly CFTC positioning composite  -> positioning.core.score.run_score(...).composite
  * daily US Treasury yields (FRED)     -> positioning.adapters.yields_free.fetch_yields()

We expose both a REPORT-dated composite (Tuesday CFTC snapshot, for display) and a
PUBLICATION-lagged causal composite (report + COT lag), so any predictive use is leakage-free
- exactly the timeline the positioning core already uses (see positioning/core/score.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from positioning.adapters import cftc_free, nyfed_free, yields_free
from positioning.config import COT_PUBLICATION_LAG_BDAYS, LABEL_YIELD
from positioning.core import score as score_mod


def _weekly_asof(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """As-of (ffill) a daily series onto an arbitrary weekly index (mirrors score._weekly_yields)."""
    return (daily.sort_index()
            .reindex(daily.index.union(index)).ffill().reindex(index))


@dataclass
class StudyData:
    composite: pd.Series          # weekly, report-dated (Tuesday) crowding z-index
    composite_pub: pd.Series      # weekly, publication-lagged (causal) composite
    y10_w: pd.Series              # weekly DGS10 (%), as-of on the report grid
    y10_w_pub: pd.Series          # weekly DGS10 (%), as-of on the publication grid
    slope_w: pd.Series            # weekly 2s10s (DGS10 - DGS2), report grid
    dgs10_daily: pd.Series        # daily DGS10 (%) for the ARIMA
    dgs2_daily: pd.Series         # daily DGS2 (%)
    score: object                 # the full PositioningScore (for .latest, .percentile, ...)


def load(force: bool = False) -> StudyData:
    """Build every series the study/forecaster needs, off the free adapters (cache-first)."""
    fut = cftc_free.fetch_cftc_tff(force_download=force)
    cash = nyfed_free.fetch_nyfed_cash(force_download=force)
    yields = yields_free.fetch_yields(force_download=force)          # wide, daily, %

    res = score_mod.run_score(fut, cash, yields)
    composite = res.composite.dropna()

    dgs10 = yields[LABEL_YIELD].dropna()                             # daily %
    dgs2 = yields["DGS2"].dropna() if "DGS2" in yields.columns else pd.Series(dtype=float)

    # weekly, report-dated grid
    wk = composite.index
    y10_w = _weekly_asof(dgs10, wk)
    slope_w = (y10_w - _weekly_asof(dgs2, wk)) if len(dgs2) else y10_w * 0.0

    # weekly, publication-lagged (causal) grid
    pub_index = composite.index + pd.tseries.offsets.BDay(COT_PUBLICATION_LAG_BDAYS)
    composite_pub = pd.Series(composite.values, index=pub_index, name="composite_pub")
    y10_w_pub = _weekly_asof(dgs10, pub_index)

    return StudyData(
        composite=composite, composite_pub=composite_pub,
        y10_w=y10_w, y10_w_pub=y10_w_pub, slope_w=slope_w,
        dgs10_daily=dgs10, dgs2_daily=dgs2, score=res,
    )
