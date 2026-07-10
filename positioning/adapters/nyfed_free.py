"""Free adapter: NY Fed primary-dealer cash Treasury net positions (FR2004).

Source: markets.newyorkfed.org primary-dealer API (keyless).
Series: PDPOSGSC-* net outright coupon positions per maturity bucket (config.NYFED_CASH_SERIES),
i.e. the *cash/spot* Treasury positioning leg that complements the futures COT data.
Values are $ par (net long - short), weekly (Wednesday as-of).

Produces the standardized CashPanel (see core/schema.py). Cache-first CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import DATA_DIR, DEFAULT_START, NYFED_CASH_SERIES

GET_URL = "https://markets.newyorkfed.org/api/pd/get/{keyid}.json"
_CACHE = DATA_DIR / "nyfed_cash.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (positioning-tool)"}


def _download() -> pd.DataFrame:
    frames = []
    for keyid, tenor in NYFED_CASH_SERIES.items():
        r = requests.get(GET_URL.format(keyid=keyid), headers=_HEADERS, timeout=60)
        r.raise_for_status()
        ts = r.json().get("pd", {}).get("timeseries", [])
        if not ts:
            continue
        df = pd.DataFrame(ts)
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(df["asofdate"]),
            "tenor_years": tenor,
            "net_value": pd.to_numeric(df["value"], errors="coerce"),
        }))
    if not frames:
        return pd.DataFrame(columns=["date", "tenor_years", "net_value"])
    return pd.concat(frames, ignore_index=True)


def fetch_nyfed_cash(start: str = DEFAULT_START, cache_path: Path | None = None,
                     force_download: bool = False) -> pd.DataFrame:
    """Return the standardized CashPanel; cache-first CSV.

    Returns an empty (schema-shaped) frame on any network failure so the composite can
    still be built from futures alone — cash is a secondary input.
    """
    cache = Path(cache_path) if cache_path else _CACHE
    if cache.exists() and not force_download:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    try:
        panel = _download()
    except requests.RequestException:
        return pd.DataFrame(columns=["date", "tenor_years", "net_value"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return panel[panel["date"] >= pd.Timestamp(start)].reset_index(drop=True)
