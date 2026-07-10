"""Causal normalization: rolling z-score and percentile.

Every raw positioning series is put on a common, unit-free scale before aggregation.
All windows are trailing (causal) so a value at time t uses only data up to t — this is
what makes the composite point-in-time and comparable across instruments/categories.
"""
from __future__ import annotations

import pandas as pd


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing z-score: (x_t - rolling_mean) / rolling_std, causal."""
    mp = min_periods if min_periods is not None else max(window // 2, 10)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std(ddof=0)
    z = (s - mean) / std.replace(0.0, pd.NA)
    return z.astype(float)


def rolling_percentile(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing percentile rank in [0, 1] of x_t within its trailing window."""
    mp = min_periods if min_periods is not None else max(window // 2, 10)

    def _rank(x) -> float:
        # fraction of the window strictly below the last value (the current point)
        last = x[-1]
        return (x < last).mean()

    return s.rolling(window, min_periods=mp).apply(_rank, raw=True)


def zscore_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Column-wise causal z-score of a wide frame."""
    return df.apply(lambda c: rolling_zscore(c, window))
