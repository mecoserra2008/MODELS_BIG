"""Fixed-income news fetch: GDELT DOC 2.0 + curated RSS, with a rolling cache.

Two keyless sources, unified into one tidy frame ``[ts, source, headline, url]``:

* **GDELT DOC 2.0** (``artlist`` mode) — global news with per-article timestamps up to ~1
  month back, so the 1W / 1M sentiment windows are populated immediately (no waiting to
  accumulate). Keyless.
* **Curated RSS** — Fed / US Treasury / ECB press releases + FI-focused market feeds
  (ForexLive, FXStreet, MarketWatch bonds). Freshest headlines for the 1D window and the
  live feed. Hand-rolled ``xml.etree`` parsing, mirroring
  ``alternative_models/ecb_analysis/news_reader.py`` (no feedparser dependency).

Everything is appended to a **rolling cache** (``data/raw/news_cache.csv``) deduped by URL,
so history grows across runs even though each source only exposes a short window.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from ..config import DATA_DIR

_CACHE = DATA_DIR / "news_cache.csv"
_UA = {"User-Agent": "Mozilla/5.0 (positioning-tool; news)"}
COLUMNS = ["ts", "source", "headline", "url"]

# FI query for GDELT (broad but on-topic).
GDELT_QUERY = ('(treasury OR "government bond" OR yields OR "fixed income" OR '
               '"federal reserve" OR bonds) (sourcelang:english)')
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Curated keyless RSS feeds relevant to fixed income / rates.
RSS_FEEDS = {
    "Fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    "US Treasury": "https://home.treasury.gov/rss/press.xml",
    "ECB": "https://www.ecb.europa.eu/rss/press.html",
    "ForexLive": "https://www.forexlive.com/feed/news",
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "MarketWatch Bonds": "https://feeds.content.dowjones.io/public/rss/mw_bondsnews",
}

# Keep only headlines that look FI-relevant (cheap keyword gate for the noisy feeds).
_FI_KEYWORDS = (
    "treasury", "bond", "yield", "fixed income", "fed", "fomc", "rate", "rates",
    "inflation", "cpi", "ppi", "duration", "curve", "auction", "coupon", "ecb",
    "gilt", "bund", "jgb", "sofr", "hawkish", "dovish", "tapering", "qt", "qe",
)


def _fingerprint(url: str, headline: str) -> str:
    return hashlib.md5((url or headline).encode("utf-8", "ignore")).hexdigest()


def _is_fi(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _FI_KEYWORDS)


# --------------------------------------------------------------------------- #
# GDELT
# --------------------------------------------------------------------------- #
def fetch_gdelt(timespan: str = "1m", maxrecords: int = 250) -> pd.DataFrame:
    params = {"query": GDELT_QUERY, "mode": "artlist", "format": "json",
              "timespan": timespan, "maxrecords": maxrecords, "sort": "datedesc"}
    try:
        r = requests.get(GDELT_URL, params=params, headers=_UA, timeout=30)
        r.raise_for_status()
        arts = r.json().get("articles", [])
    except (requests.RequestException, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for a in arts:
        title = a.get("title", "").strip()
        if not title:
            continue
        # seendate like "20260630T120000Z"
        try:
            ts = datetime.strptime(a.get("seendate", ""), "%Y%m%dT%H%M%SZ") \
                .replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append({"ts": ts, "source": a.get("domain", "gdelt"),
                     "headline": title, "url": a.get("url", "")})
    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------------------------------- #
# RSS
# --------------------------------------------------------------------------- #
def _parse_rss(xml_text: str, source: str) -> list[dict]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    # RSS 2.0 <item> and Atom <entry>
    items = root.iter("item")
    for it in items:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date")
        ts = _parse_date(pub)
        if title:
            out.append({"ts": ts, "source": source, "headline": title, "url": link})
    if not out:  # Atom fallback
        ns = "{http://www.w3.org/2005/Atom}"
        for e in root.iter(f"{ns}entry"):
            title = (e.findtext(f"{ns}title") or "").strip()
            link_el = e.find(f"{ns}link")
            link = link_el.get("href") if link_el is not None else ""
            ts = _parse_date(e.findtext(f"{ns}updated") or e.findtext(f"{ns}published"))
            if title:
                out.append({"ts": ts, "source": source, "headline": title, "url": link})
    return out


def _parse_date(s: str | None) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def fetch_rss() -> pd.DataFrame:
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, headers=_UA, timeout=15)
            r.raise_for_status()
            rows.extend(_parse_rss(r.text, source))
        except requests.RequestException:
            continue
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    return df[df["headline"].map(_is_fi)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Combined + rolling cache
# --------------------------------------------------------------------------- #
def fetch_news(use_gdelt: bool = True, use_rss: bool = True,
               cache_path: Path | None = None) -> pd.DataFrame:
    """Fetch fresh news, merge with the rolling cache, dedup, persist, return all.

    Robust to any single source failing (returns whatever else is available, incl. the
    existing cache) so the app never hard-fails on a flaky feed.
    """
    cache = Path(cache_path) if cache_path else _CACHE
    frames = []
    if cache.exists():
        try:
            frames.append(pd.read_csv(cache, parse_dates=["ts"]))
        except Exception:
            pass
    if use_gdelt:
        frames.append(fetch_gdelt())
    if use_rss:
        frames.append(fetch_rss())

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "headline"])
    df["fp"] = [_fingerprint(u, h) for u, h in zip(df["url"], df["headline"])]
    df = df.drop_duplicates("fp").drop(columns="fp").sort_values("ts")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df.reset_index(drop=True)
