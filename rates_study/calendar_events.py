"""Scheduled macro-event calendar for event-aware yield volatility.

Two event families with KNOWN-IN-ADVANCE dates:
  * NFP  - first Friday of each month (deterministic rule; the rare schedule shifts are
           second-order for a vol multiplier).
  * FOMC - decision days fetched from federalreserve.gov calendar pages (cached), with a
           hardcoded 2025-2026 fallback so the LIVE horizon still works offline.

Used by stoch_validation.py (expanding event-day |dY| multiplier, no look-ahead) and by
stochastic.py (how many scheduled event days sit inside the next 30bd).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_CACHE = Path(__file__).resolve().parent / "data"
_CACHE.mkdir(parents=True, exist_ok=True)
_FOMC_CSV = _CACHE / "fomc_dates.csv"

# Fallback: known/announced FOMC decision (day-2) dates. Enough for the live 30bd horizon.
_FOMC_FALLBACK = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}


def nfp_dates(start: str, end: str) -> pd.DatetimeIndex:
    """First Friday of each month in [start, end]."""
    months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
    out = []
    for m in months:
        d = m + pd.Timedelta(days=(4 - m.weekday()) % 7)   # first Friday
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            out.append(d)
    return pd.DatetimeIndex(out)


def _fetch_fomc() -> list[str]:
    """Best-effort scrape of FOMC meeting dates (2011+). Returns ISO date strings."""
    import requests
    dates: set[str] = set()
    urls = (["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"] +
            [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
             for y in range(2011, 2021)])
    for url in urls:
        try:
            html = requests.get(url, timeout=15).text
        except Exception:
            continue
        # patterns like "January 28-29" / "March 18" near a year context
        for ym in re.finditer(r"(20\d{2})", url + " 2024"):
            pass
        # generic: Month dd-dd or Month dd, with the page year(s) found in the html
        years = set(int(y) for y in re.findall(r"20[12]\d", html))
        for m in re.finditer(r"([A-Z][a-z]+)\s+(\d{1,2})(?:-(\d{1,2}))?", html):
            mon = m.group(1)
            if mon not in _MONTHS:
                continue
            day = int(m.group(3) or m.group(2))            # decision = 2nd day if a range
            for y in years:
                try:
                    d = pd.Timestamp(year=y, month=_MONTHS[mon], day=day)
                except ValueError:
                    continue
                if 2011 <= y <= 2027:
                    dates.add(str(d.date()))
    return sorted(dates)


def fomc_dates(start: str, end: str, allow_fetch: bool = True) -> pd.DatetimeIndex:
    """FOMC decision dates in [start, end]: cache -> fetch -> fallback list.
    The scrape is best-effort; when it looks implausible (too many dates/year) we keep
    only the fallback + cache."""
    dates: list[str] = []
    if _FOMC_CSV.exists():
        dates = pd.read_csv(_FOMC_CSV)["date"].tolist()
    elif allow_fetch:
        try:
            fetched = _fetch_fomc()
            per_year = pd.Series([d[:4] for d in fetched]).value_counts()
            if len(fetched) and per_year.max() <= 12:      # plausibility gate
                dates = fetched
                pd.DataFrame({"date": dates}).to_csv(_FOMC_CSV, index=False)
        except Exception:
            dates = []
    if not dates:
        dates = _FOMC_FALLBACK
    idx = pd.DatetimeIndex(sorted(set(dates)))
    return idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]


def event_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Union of NFP + FOMC dates (normalized)."""
    return nfp_dates(start, end).union(fomc_dates(start, end)).normalize()
