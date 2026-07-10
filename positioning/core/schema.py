"""Standardized positioning-panel contract.

This is the single interface every data adapter (free CFTC/NYFed, or Bloomberg bql)
must produce. The scoring core consumes only these frames, never raw source data, so
Track A and Track B differ *only* in the adapter that builds them.

Three tidy frames:

FuturesPanel  (long / tidy)
    columns: date, instrument, category, net_contracts, open_interest,
             conc_net_top4_long, conc_net_top4_short
    - one row per (weekly date, instrument in TU/FV/TY/TN/US/UB, TFF category)
    - net_contracts = long - short (signed, in contracts)
    - date is the CFTC report (Tuesday snapshot) date, tz-naive

CashPanel  (long / tidy)
    columns: date, tenor_years, net_value
    - NY Fed primary-dealer net Treasury coupon position per tenor bucket ($ par)

YieldsFrame  (wide)
    index: DatetimeIndex ; columns: FRED yield series (DGS2, DGS5, ...) in percent

Adapters return raw (unlagged) frames on the native report dates. Publication-lag
alignment and weekly resampling happen inside the scoring core (score.py), so the
leakage policy lives in one place.
"""
from __future__ import annotations

import pandas as pd

FUTURES_COLUMNS = [
    "date",
    "instrument",
    "category",
    "net_contracts",
    "open_interest",
    "conc_net_top4_long",
    "conc_net_top4_short",
]

CASH_COLUMNS = ["date", "tenor_years", "net_value"]


class SchemaError(ValueError):
    """Raised when a panel does not satisfy the standardized contract."""


def validate_futures_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate & normalize a FuturesPanel; returns a cleaned copy."""
    missing = set(FUTURES_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"FuturesPanel missing columns: {sorted(missing)}")
    if df.empty:
        raise SchemaError("FuturesPanel is empty")
    out = df[FUTURES_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for c in ("net_contracts", "open_interest",
              "conc_net_top4_long", "conc_net_top4_short"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[["date", "instrument", "category"]].duplicated().any():
        raise SchemaError("FuturesPanel has duplicate (date, instrument, category) rows")
    return out.sort_values(["date", "instrument", "category"]).reset_index(drop=True)


def validate_cash_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate & normalize a CashPanel; returns a cleaned copy."""
    missing = set(CASH_COLUMNS) - set(df.columns)
    if missing:
        raise SchemaError(f"CashPanel missing columns: {sorted(missing)}")
    out = df[CASH_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["tenor_years"] = pd.to_numeric(out["tenor_years"], errors="coerce")
    out["net_value"] = pd.to_numeric(out["net_value"], errors="coerce")
    return out.sort_values(["date", "tenor_years"]).reset_index(drop=True)


def validate_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide yields frame with a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError("YieldsFrame must have a DatetimeIndex")
    if df.empty:
        raise SchemaError("YieldsFrame is empty")
    out = df.copy()
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out.sort_index().apply(pd.to_numeric, errors="coerce")
