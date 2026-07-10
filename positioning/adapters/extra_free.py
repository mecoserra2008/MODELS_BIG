"""Extra positioning indicators (standalone, keyless).

Each returns an ``Indicator`` (latest value + z + short history + a read), rendered beside
the composite. They are intentionally standalone now and shaped so any can later become a
PCA feature (the "composite-later" path). Every fetch degrades gracefully to an
``available=False`` placeholder so the app section always renders.

Four indicators:
  retail        CFTC non-reportable (small-trader) net — classic contra-indicator.
  dealer_repo   NY Fed primary-dealer Treasury repo financing — leverage / short-base proxy.
  etf_demand    TLT (20y+ Treasury ETF) price-momentum — long-duration demand proxy.
  rate_vol      Realized vol of FRED 10y yield — bond-vol regime (MOVE proxy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from ..config import CFTC_CODES, DATA_DIR
from ..core.normalize import rolling_zscore

_UA = {"User-Agent": "Mozilla/5.0 (positioning-tool)"}


@dataclass
class Indicator:
    key: str
    name: str
    note: str
    available: bool = False
    value: float = float("nan")     # latest raw value
    z: float = float("nan")         # latest z-score
    read: str = "n/a"               # "bull" | "bear" | "neutral" | regime text
    series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _contra(z: float) -> str:
    """Contrarian read: crowded long -> bearish, crowded short -> bullish."""
    if not np.isfinite(z):
        return "neutral"
    return "bear" if z > 1.0 else "bull" if z < -1.0 else "neutral"


# --------------------------------------------------------------------------- #
# 1. Retail / non-reportable (CFTC)
# --------------------------------------------------------------------------- #
def retail() -> Indicator:
    note = ("CFTC non-reportable (small-trader) net position across the Treasury complex, "
            "z-scored. Retail is a contra-indicator — crowded retail longs → fade → bearish.")
    ind = Indicator("retail", "Retail (non-reportable)", note)
    try:
        codes = ",".join(f"'{c}'" for c in CFTC_CODES)
        params = {
            "$select": ("report_date_as_yyyy_mm_dd,nonrept_positions_long_all,"
                        "nonrept_positions_short_all"),
            "$where": f"cftc_contract_market_code in ({codes})",
            "$order": "report_date_as_yyyy_mm_dd", "$limit": 200000,
        }
        r = requests.get("https://publicreporting.cftc.gov/resource/gpe5-46if.json",
                         params=params, timeout=45)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        for c in ("nonrept_positions_long_all", "nonrept_positions_short_all"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        net = (df.assign(net=df["nonrept_positions_long_all"]
                         - df["nonrept_positions_short_all"])
               .groupby("date")["net"].sum().sort_index())
        z = rolling_zscore(net, 156).dropna()
        ind.available = True
        ind.value = float(net.iloc[-1])
        ind.z = float(z.iloc[-1])
        ind.read = _contra(ind.z)
        ind.series = z.tail(104)
    except Exception:
        pass
    return ind


# --------------------------------------------------------------------------- #
# 2. Dealer repo financing (NY Fed) — leverage / short-base proxy
# --------------------------------------------------------------------------- #
def dealer_repo() -> Indicator:
    note = ("NY Fed primary-dealer Treasury repo financing (securities-in via reverse repo). "
            "A leverage / short-base proxy — elevated financing signals a stretched dealer "
            "book. Note: dealer *futures* positioning is already in the composite (CFTC "
            "Dealer category); this adds the financing angle.")
    ind = Indicator("dealer_repo", "Dealer repo financing", note)
    try:
        url = "https://markets.newyorkfed.org/api/pd/get/PDSIRRA-GCFUTSET.json"
        r = requests.get(url, headers=_UA, timeout=30)
        r.raise_for_status()
        ts = r.json()["pd"]["timeseries"]
        s = pd.Series({pd.to_datetime(x["asofdate"]): pd.to_numeric(x["value"], errors="coerce")
                       for x in ts}).sort_index().dropna()
        z = rolling_zscore(s, 156).dropna()
        ind.available = True
        ind.value = float(s.iloc[-1])
        ind.z = float(z.iloc[-1])
        ind.read = ("elevated leverage" if ind.z > 1 else
                    "light leverage" if ind.z < -1 else "normal")
        ind.series = z.tail(104)
    except Exception:
        pass
    return ind


# --------------------------------------------------------------------------- #
# 3. Treasury-ETF demand (TLT momentum, Yahoo chart API)
# --------------------------------------------------------------------------- #
def etf_demand() -> Indicator:
    note = ("iShares 20y+ Treasury ETF (TLT) price momentum (20d return, z-scored) as a "
            "long-duration demand proxy. Contrarian: crowded ETF longs → fade → bearish. "
            "Best-effort (keyless Yahoo chart API); true creation/redemption flows need a vendor.")
    ind = Indicator("etf_demand", "Long-duration ETF demand (TLT)", note)
    try:
        url = ("https://query1.finance.yahoo.com/v8/finance/chart/TLT"
               "?range=2y&interval=1d")
        r = requests.get(url, headers=_UA, timeout=20)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        closes = res["indicators"]["quote"][0]["close"]
        ts = res["timestamp"]
        px = pd.Series(closes, index=pd.to_datetime(ts, unit="s")).dropna()
        mom = px.pct_change(20)
        z = rolling_zscore(mom, 252).dropna()
        ind.available = True
        ind.value = float(mom.iloc[-1] * 100)
        ind.z = float(z.iloc[-1])
        ind.read = _contra(ind.z)
        ind.series = z.tail(126)
    except Exception:
        pass
    return ind


# --------------------------------------------------------------------------- #
# 4. Rate-vol regime (FRED DGS10 realized vol) — MOVE proxy
# --------------------------------------------------------------------------- #
def rate_vol(yields: pd.DataFrame | None = None) -> Indicator:
    note = ("Realized volatility of the 10y Treasury yield (21d, annualized), z-scored — a "
            "keyless MOVE proxy for the bond-vol regime. High vol = stressed / risk-off "
            "conditions that amplify how positioning matters.")
    ind = Indicator("rate_vol", "Rate-vol regime (MOVE proxy)", note)
    try:
        if yields is None:
            from . import yields_free
            yields = yields_free.fetch_yields()
        y10 = yields["DGS10"].dropna()
        rv = y10.diff().rolling(21).std() * np.sqrt(252)
        z = rolling_zscore(rv, 252).dropna()
        ind.available = True
        ind.value = float(rv.iloc[-1] * 100)  # bp/day annualized, in bp
        ind.z = float(z.iloc[-1])
        ind.read = ("stressed" if ind.z > 1 else "calm" if ind.z < -1 else "normal")
        ind.series = z.tail(126)
    except Exception:
        pass
    return ind


def all_indicators(yields: pd.DataFrame | None = None) -> list[Indicator]:
    return [retail(), dealer_repo(), etf_demand(), rate_vol(yields)]
