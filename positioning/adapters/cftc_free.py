"""Free adapter: CFTC Traders in Financial Futures (TFF), futures-only.

Source: CFTC Public Reporting Environment Socrata API (keyless).
Dataset: gpe5-46if  ("Traders in Financial Futures; Futures Only Report").

We query by cftc_contract_market_code (config.CFTC_CODES) rather than the market name,
because CFTC renamed the Treasury contracts in Feb-2022 (e.g. "10-YEAR U.S. TREASURY
NOTES" -> "UST 10Y NOTE"); the code is stable across the rename, so this pulls the full
history back to 2006 in one shot.

Produces the standardized FuturesPanel (see core/schema.py). Cache-first CSV, mirroring
the repo's src/data_retrieval.py convention.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..config import (
    CFTC_CODES, CONCENTRATION_FIELDS, CONTRACT_BY_CODE, DATA_DIR, DEFAULT_START,
    TFF_CATEGORIES,
)

SOCRATA_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
_CACHE = DATA_DIR / "cftc_tff.csv"

_FIELDS = [
    "report_date_as_yyyy_mm_dd", "cftc_contract_market_code", "open_interest_all",
    *[f for pair in TFF_CATEGORIES.values() for f in pair],
    *CONCENTRATION_FIELDS,
]


def _download(start: str) -> pd.DataFrame:
    codes = ",".join(f"'{c}'" for c in CFTC_CODES)
    params = {
        "$select": ",".join(_FIELDS),
        "$where": (f"cftc_contract_market_code in ({codes}) "
                   f"and report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000'"),
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": 200000,
    }
    r = requests.get(SOCRATA_URL, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def _to_panel(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    num = [c for c in df.columns if c != "cftc_contract_market_code"]
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in num:
        if c != "report_date_as_yyyy_mm_dd":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for _, r in df.iterrows():
        spec = CONTRACT_BY_CODE.get(r["cftc_contract_market_code"])
        if spec is None:
            continue
        for cat, (lf, sf) in TFF_CATEGORIES.items():
            rows.append({
                "date": r["report_date_as_yyyy_mm_dd"],
                "instrument": spec.key,
                "category": cat,
                "net_contracts": r.get(lf, float("nan")) - r.get(sf, float("nan")),
                "open_interest": r.get("open_interest_all", float("nan")),
                "conc_net_top4_long": r.get(CONCENTRATION_FIELDS[0], float("nan")),
                "conc_net_top4_short": r.get(CONCENTRATION_FIELDS[1], float("nan")),
            })
    return pd.DataFrame(rows)


def fetch_cftc_tff(start: str = DEFAULT_START, cache_path: Path | None = None,
                   force_download: bool = False) -> pd.DataFrame:
    """Return the standardized FuturesPanel; cache-first CSV."""
    cache = Path(cache_path) if cache_path else _CACHE
    if cache.exists() and not force_download:
        df = pd.read_csv(cache, parse_dates=["date"])
        return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    panel = _to_panel(_download(start))
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache, index=False)
    return panel
