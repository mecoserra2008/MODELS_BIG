"""Free adapter: FRED constant-maturity Treasury yields for the DV01 layer.

Reuses the repo's established FRED access pattern (fredapi + the key in
alternative_models/.env, as run_10s5s.py does). Only the tenors referenced by
config.CONTRACTS are pulled. Returns a wide YieldsFrame (DatetimeIndex, percent).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from ..config import CONTRACTS, DATA_DIR, DEFAULT_START, FRED_ENV_PATH, LABEL_YIELD

_CACHE = DATA_DIR / "fred_yields.csv"
_SERIES = sorted({c.fred_yield for c in CONTRACTS} | {LABEL_YIELD})


def _fred_key() -> str:
    """Resolve the FRED key from env / .env / Streamlit secrets (cloud deploy)."""
    load_dotenv(FRED_ENV_PATH)
    load_dotenv()  # repo-root fallback
    key = os.environ.get("FRED_API_KEY")
    if not key:  # Streamlit Community Cloud: set FRED_API_KEY in the app's Secrets
        try:
            import streamlit as st
            key = st.secrets.get("FRED_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not found (looked in env, "
            f"{FRED_ENV_PATH}, and Streamlit secrets).")
    return key


def _download(start: str) -> pd.DataFrame:
    from fredapi import Fred
    fred = Fred(api_key=_fred_key())
    cols = {}
    for s in _SERIES:
        cols[s] = fred.get_series(s, observation_start=start)
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def fetch_yields(start: str = DEFAULT_START, cache_path: Path | None = None,
                 force_download: bool = False) -> pd.DataFrame:
    """Return wide FRED yields (percent); cache-first CSV."""
    cache = Path(cache_path) if cache_path else _CACHE
    if cache.exists() and not force_download:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df[df.index >= pd.Timestamp(start)]
    df = _download(start)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df
