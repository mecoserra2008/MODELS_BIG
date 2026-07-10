"""Aggregate scored headlines into 1D / 1W / 1M sentiment readings + a feed.

Signed per-headline scores in [-1, 1] are averaged over trailing windows and rescaled to
a -100..+100 barometer (positive = bullish FI / risk-on tone). Also builds a daily
sentiment time series and the live feed rows the UI renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import finbert, news_fetch

WINDOWS = {"1D": 1, "1W": 7, "1M": 30}


@dataclass
class WindowScore:
    label: str
    score: float          # -100..+100 barometer
    n: int                # headline count in window
    n_bull: int
    n_bear: int


@dataclass
class SentimentResult:
    windows: dict[str, WindowScore]
    series: pd.Series          # daily mean barometer over ~1M
    feed: pd.DataFrame         # recent scored headlines (ts, source, headline, url, score, label)
    backend: str
    as_of: pd.Timestamp
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)


def _label(score: float) -> str:
    if score >= 15:
        return "bull"
    if score <= -15:
        return "bear"
    return "neutral"


def compute(use_gdelt: bool = True, use_rss: bool = True) -> SentimentResult:
    news = news_fetch.fetch_news(use_gdelt=use_gdelt, use_rss=use_rss)
    now = pd.Timestamp.now(tz="UTC")
    if news.empty:
        empty = {k: WindowScore(k, 0.0, 0, 0, 0) for k in WINDOWS}
        return SentimentResult(empty, pd.Series(dtype=float), news, finbert.backend(), now)

    news = news.sort_values("ts").reset_index(drop=True)
    scores = finbert.score_headlines(news["headline"].tolist())
    news = news.assign(score=scores, bar=scores * 100.0)
    news["label"] = news["bar"].map(_label)

    windows = {}
    for k, days in WINDOWS.items():
        cut = now - pd.Timedelta(days=days)
        w = news[news["ts"] >= cut]
        bar = float(w["bar"].mean()) if len(w) else 0.0
        windows[k] = WindowScore(k, bar, int(len(w)),
                                 int((w["label"] == "bull").sum()),
                                 int((w["label"] == "bear").sum()))

    # daily mean barometer over the last ~30d for the sparkline
    recent = news[news["ts"] >= now - pd.Timedelta(days=30)].copy()
    series = (recent.set_index("ts")["bar"].resample("1D").mean().dropna()
              if len(recent) else pd.Series(dtype=float))

    feed = (news.sort_values("ts", ascending=False)
            .head(40)[["ts", "source", "headline", "url", "bar", "label"]]
            .rename(columns={"bar": "score"}).reset_index(drop=True))

    return SentimentResult(windows, series, feed, finbert.backend(), now, scored=news)
