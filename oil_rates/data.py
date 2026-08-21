"""Data layer: FRED daily series -> aligned oil/rates/conditioner frames.

Reuses the robust FRED key resolver from the positioning package (works from any CWD;
key lives in alternative_models/.env). All series are free FRED dailies.

WTI note: DCOILWTICO went NEGATIVE in April 2020 (-$37.63). Log-returns are undefined
there, so return computation drops price<=0 days and flags them - those days are
themselves prime examples of the "datapoints that skew the distribution".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from positioning.adapters.yields_free import _fred_key

PKG_DIR = Path(__file__).resolve().parent
CACHE = PKG_DIR / "data"
CACHE.mkdir(parents=True, exist_ok=True)
_CSV = CACHE / "fred_oil_rates.csv"
_EQ_CSV = CACHE / "equity.csv"

_STOOQ_SPX = "https://stooq.com/q/d/l/?s=^spx&i=d"
_EQ_SOURCES = {"spx_stooq": "stooq ^spx (full history)",
               "spx_fred": "FRED SP500 (~10y history only; stooq unavailable)"}

SERIES = ["DCOILWTICO", "DCOILBRENTEU", "DGS10", "DGS2", "DFII10", "T10YIE",
          "T5YIFR", "VIXCLS", "DTWEXBGS", "DFF"]
START = "1986-01-02"          # WTI start; TIPS/BEI legs begin 2003 (handled downstream)
VOL_WIN = 21                  # ~1 month realized-vol window


def fetch(force: bool = False) -> pd.DataFrame:
    """Wide daily FRED frame (columns = FRED codes), cache-first."""
    if _CSV.exists() and not force:
        return pd.read_csv(_CSV, index_col=0, parse_dates=True)
    from fredapi import Fred
    fred = Fred(api_key=_fred_key())
    df = pd.DataFrame({s: fred.get_series(s, observation_start=START) for s in SERIES})
    df = df.sort_index()
    df.to_csv(_CSV)
    return df


def fetch_equity(force: bool = False) -> tuple[pd.Series | None, str]:
    """Long daily S&P 500 close for demand/supply shock classification, cache-first.

    Tries the stooq daily CSV first (full history; mirrors src.data_retrieval's
    stooq pattern), then falls back to FRED "SP500" (licensing caps it at ~10y).
    Returns (series, source_note); (None, note) if both sources fail.
    """
    if _EQ_CSV.exists() and not force:
        df = pd.read_csv(_EQ_CSV, index_col=0, parse_dates=True)
        col = df.columns[0]
        return df[col], _EQ_SOURCES.get(col, col) + " [cached]"
    # 1) stooq daily CSV (Date,Open,High,Low,Close)
    try:
        import io
        import requests
        resp = requests.get(_STOOQ_SPX, timeout=60)
        resp.raise_for_status()
        text = resp.text.strip()
        if text and "No data" not in text and text.lower().startswith("date"):
            raw = pd.read_csv(io.StringIO(text))
            s = pd.Series(pd.to_numeric(raw["Close"], errors="coerce").values,
                          index=pd.to_datetime(raw["Date"]), name="spx_stooq").dropna()
            if len(s):
                s.to_frame().to_csv(_EQ_CSV)
                return s, _EQ_SOURCES["spx_stooq"]
        print("stooq ^spx returned no CSV (blocked/empty) - falling back to FRED SP500")
    except Exception as e:
        print(f"stooq ^spx fetch failed ({e}) - falling back to FRED SP500")
    # 2) FRED SP500 fallback (~10y of daily closes only)
    try:
        from fredapi import Fred
        s = Fred(api_key=_fred_key()).get_series("SP500", observation_start=START)
        s = s.dropna().rename("spx_fred")
        if len(s):
            s.to_frame().to_csv(_EQ_CSV)
            return s, _EQ_SOURCES["spx_fred"]
    except Exception as e:
        print(f"FRED SP500 fetch failed: {e}")
    return None, "no equity series available (stooq and FRED SP500 both failed)"


def _log_returns(price: pd.Series) -> tuple[pd.Series, list]:
    """Log-returns with non-positive prices dropped; returns (ret, dropped_dates)."""
    p = price.dropna()
    bad = p[p <= 0].index
    p = p[p > 0]
    r = np.log(p).diff()
    return r, [str(d.date()) for d in bad]


@dataclass
class OilRatesData:
    raw: pd.DataFrame                 # wide FRED levels
    daily: pd.DataFrame               # aligned daily analysis frame (see build)
    monthly: pd.DataFrame             # month-end aggregation of the same
    dropped_wti_days: list = field(default_factory=list)   # negative-price days
    equity_note: str = ""             # provenance of the spx_ret column


def build(force: bool = False) -> OilRatesData:
    raw = fetch(force=force)
    equity, eq_note = fetch_equity(force=force)

    wti_ret, dropped = _log_returns(raw["DCOILWTICO"])
    brent_ret, _ = _log_returns(raw["DCOILBRENTEU"])
    spx_ret = _log_returns(equity)[0] if equity is not None else None

    d = pd.DataFrame({
        "oil_ret": wti_ret * 100.0,                       # WTI log-return, %
        "brent_ret": brent_ret * 100.0,                   # Brent log-return, %
        "spx_ret": spx_ret * 100.0 if spx_ret is not None else np.nan,  # S&P 500 log-ret, %
        "dy10": raw["DGS10"].diff() * 100.0,              # Δ nominal 10y, bp
        "dreal": raw["DFII10"].diff() * 100.0,            # Δ 10y real (TIPS), bp
        "dbei": raw["T10YIE"].diff() * 100.0,             # Δ 10y breakeven, bp
        "d5y5y": raw["T5YIFR"].diff() * 100.0,            # Δ 5y5y fwd inflation, bp
        # conditioning variables (levels / slow-moving, all observable same-day)
        "vix": raw["VIXCLS"],
        "usd_ret": np.log(raw["DTWEXBGS"]).diff() * 100.0,
        "bei_level": raw["T10YIE"],
        "y10_level": raw["DGS10"],
        "dff": raw["DFF"],
        "wti_level": raw["DCOILWTICO"],
    })
    d["oil_vol"] = d["oil_ret"].rolling(VOL_WIN, min_periods=10).std()
    d["rate_vol"] = d["dy10"].rolling(VOL_WIN, min_periods=10).std()

    # month-end aggregation: returns/changes sum; conditioners take month-end level
    m = pd.DataFrame({
        "oil_ret": d["oil_ret"].resample("ME").sum(min_count=5),
        "brent_ret": d["brent_ret"].resample("ME").sum(min_count=5),
        "dy10": d["dy10"].resample("ME").sum(min_count=5),
        "dreal": d["dreal"].resample("ME").sum(min_count=5),
        "dbei": d["dbei"].resample("ME").sum(min_count=5),
        "vix": d["vix"].resample("ME").last(),
        "bei_level": d["bei_level"].resample("ME").last(),
        "oil_vol": d["oil_vol"].resample("ME").last(),
        "dff": d["dff"].resample("ME").last(),
    })
    return OilRatesData(raw=raw, daily=d, monthly=m, dropped_wti_days=dropped,
                        equity_note=eq_note)
