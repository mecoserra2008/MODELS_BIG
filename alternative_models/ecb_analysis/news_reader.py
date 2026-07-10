"""
Live financial news scraper for ECB press conference monitoring.

Runs three sources concurrently in separate background threads:
  Layer 1 — FinancialJuice HTML  (20 s interval — avoids 429 rate limit)
  Layer 2 — ForexLive RSS        (8 s interval — reliable, ECB-specific)
  Layer 3 — FXStreet RSS         (12 s interval — good Lagarde quote coverage)

All three write to the same /tmp/ecb_live_feed.txt JSONL file. A shared
deduplication set (thread-safe via a lock) prevents duplicate entries.
_pull_feed() in ecb_live_app.py is unchanged — it just reads new bytes.

Usage:
    # standalone test
    python news_reader.py

    # from Streamlit
    from news_reader import start_news_feed, stop_news_feed, is_running, active_source
"""

from __future__ import annotations
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

FEED_FILE = Path("/tmp/ecb_live_feed.txt")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

ECB_KEYWORDS = [
    "lagarde", "ecb ", "e.c.b", "governing council", "euro area",
    "eurozone", "rate decision", "basis points", "inflation target",
    "data-dependent", "data dependent", "restrictive", "neutral rate",
    "deposit facility", "main refinancing", "eonia", "euribor",
    "christine", "villeroy", "lane", "schnabel", "nagel",
    "monetary policy", "interest rate", "bund", "bundesbank",
    "european central bank",
]

# ── Source definitions ────────────────────────────────────────────────────────

FINANCIALJUICE_HTML_URL = "https://www.financialjuice.com/"
FINANCIALJUICE_INTERVAL = 20   # s — respect rate limit (429 at < 15 s)

FOREXLIVE_RSS_URL  = "https://www.forexlive.com/feed/news"
FOREXLIVE_INTERVAL = 8

FXSTREET_RSS_URLS  = [
    "https://www.fxstreet.com/rss",
    "https://rss.fxstreet.com/rss/news",
]
FXSTREET_INTERVAL  = 12

# ── Module state (thread-safe) ────────────────────────────────────────────────

_stop_event    = threading.Event()
_threads: list[threading.Thread] = []
_is_running    = False

_seen_lock     = threading.Lock()
_seen: set[str] = set()

_counter_lock  = threading.Lock()
_chunk_counter = 0

_active_lock   = threading.Lock()
_active_sources: set[str] = set()


def active_source() -> str:
    with _active_lock:
        return ", ".join(sorted(_active_sources)) or "none"


def is_running() -> bool:
    return _is_running


# ── Core helpers ──────────────────────────────────────────────────────────────

def _is_ecb(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in ECB_KEYWORDS)


def _fingerprint(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _write(text: str):
    global _chunk_counter
    fp = _fingerprint(text)
    with _seen_lock:
        if fp in _seen:
            return False
        _seen.add(fp)
    with _counter_lock:
        idx = _chunk_counter
        _chunk_counter += 1
    FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "chunk_index": idx,
        "text":        text,
    })
    with open(FEED_FILE, "a", encoding="utf-8") as f:
        f.write(payload + "\n")
    print(f"[news_reader] [{idx}] {text[:90]}…")
    return True


def _get(url: str, timeout: int = 10) -> requests.Response | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        print(f"[news_reader] HTTP {r.status_code} ← {url}")
        return None
    except Exception as exc:
        print(f"[news_reader] {url}: {exc}")
        return None


# ── RSS parser ────────────────────────────────────────────────────────────────

def _parse_rss(content: str) -> list[str]:
    texts: list[str] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return texts

    ns_atom = "http://www.w3.org/2005/Atom"

    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc  = (item.findtext("description") or "").strip()
        if desc:
            try:
                desc = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)
            except Exception:
                pass
        combined = (f"{title}. {desc}" if title and desc and desc.lower() != title.lower()
                    else title or desc)
        if len(combined) >= 20:
            texts.append(combined)

    # Atom
    if not texts:
        for entry in root.iter(f"{{{ns_atom}}}entry"):
            title   = (entry.findtext(f"{{{ns_atom}}}title") or "").strip()
            summary = (entry.findtext(f"{{{ns_atom}}}summary") or "").strip()
            combined = title or summary
            if len(combined) >= 20:
                texts.append(combined)

    return texts


# ── FinancialJuice HTML parser ────────────────────────────────────────────────

def _parse_financialjuice_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    texts: list[str] = []

    # Pattern A: data-id / data-time elements (live ticker)
    for el in soup.find_all(attrs={"data-id": True}):
        t = el.get_text(" ", strip=True)
        if 20 <= len(t) <= 500:
            texts.append(t)
    if texts:
        return texts

    # Pattern B: known class names for news tickers
    for cls in ["message", "entry", "news-item", "headline", "item-text",
                "feed-item", "news-feed-item", "live-item", "ticker-item", "item"]:
        els = soup.find_all(class_=cls)
        for el in els:
            t = el.get_text(" ", strip=True)
            if 20 <= len(t) <= 500:
                texts.append(t)
        if texts:
            return texts

    # Pattern C: any short paragraph/list-item (last resort)
    for tag in ["p", "li", "span"]:
        for el in soup.find_all(tag):
            t = el.get_text(" ", strip=True)
            if 25 <= len(t) <= 300:
                texts.append(t)
    return texts


# ── Per-source worker threads ────────────────────────────────────────────────

def _financialjuice_worker(stop_event: threading.Event):
    """Poll FinancialJuice HTML every FINANCIALJUICE_INTERVAL seconds."""
    source = "FinancialJuice"
    with _active_lock:
        _active_sources.add(source)
    print(f"[news_reader] {source} worker started (interval={FINANCIALJUICE_INTERVAL}s)")

    while not stop_event.is_set():
        r = _get(FINANCIALJUICE_HTML_URL)
        if r:
            items = _parse_financialjuice_html(r.text)
            ecb   = [t for t in items if _is_ecb(t)]
            n = sum(1 for t in ecb if _write(t))
            if n:
                print(f"[news_reader] {source}: +{n} ECB items")

        # Interruptible sleep
        for _ in range(FINANCIALJUICE_INTERVAL):
            if stop_event.is_set():
                break
            time.sleep(1)

    with _active_lock:
        _active_sources.discard(source)
    print(f"[news_reader] {source} worker stopped.")


def _rss_worker(name: str, urls: list[str], interval: int, stop_event: threading.Event):
    """Generic RSS worker — tries each URL in order until one works, then sticks with it."""
    working_url: str | None = None

    with _active_lock:
        _active_sources.add(name)
    print(f"[news_reader] {name} worker started (interval={interval}s)")

    while not stop_event.is_set():
        # Re-probe if no working URL yet
        if working_url is None:
            for url in urls:
                r = _get(url)
                if r and _parse_rss(r.text):
                    working_url = url
                    print(f"[news_reader] {name}: using {url}")
                    break

        if working_url:
            r = _get(working_url)
            if r:
                items = _parse_rss(r.text)
                ecb   = [t for t in items if _is_ecb(t)]
                n = sum(1 for t in ecb if _write(t))
                if n:
                    print(f"[news_reader] {name}: +{n} ECB items")
            elif r is None:
                # URL may have changed; re-probe next cycle
                working_url = None

        for _ in range(interval):
            if stop_event.is_set():
                break
            time.sleep(1)

    with _active_lock:
        _active_sources.discard(name)
    print(f"[news_reader] {name} worker stopped.")


# ── Public API ────────────────────────────────────────────────────────────────

def start_news_feed():
    global _threads, _is_running, _chunk_counter

    if _is_running:
        print("[news_reader] Already running.")
        return

    _stop_event.clear()

    with _seen_lock:
        _seen.clear()
    with _counter_lock:
        _chunk_counter = 0
    with _active_lock:
        _active_sources.clear()

    _threads = [
        threading.Thread(
            target=_financialjuice_worker, args=(_stop_event,), daemon=True,
            name="fj-worker",
        ),
        threading.Thread(
            target=_rss_worker,
            args=("ForexLive", [FOREXLIVE_RSS_URL], FOREXLIVE_INTERVAL, _stop_event),
            daemon=True, name="forexlive-worker",
        ),
        threading.Thread(
            target=_rss_worker,
            args=("FXStreet", FXSTREET_RSS_URLS, FXSTREET_INTERVAL, _stop_event),
            daemon=True, name="fxstreet-worker",
        ),
    ]

    for t in _threads:
        t.start()

    _is_running = True
    print(f"[news_reader] Started — {len(_threads)} source threads active.")


def stop_news_feed():
    global _is_running
    _stop_event.set()
    _is_running = False
    print("[news_reader] Stopping…")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("news_reader.py — standalone test (30 s)")
    print("ECB press conference keywords filter is ACTIVE.")
    print("Outside ECB events, ForexLive/FXStreet will post ECB-adjacent news.")
    print("=" * 60)

    FEED_FILE.unlink(missing_ok=True)
    start_news_feed()

    try:
        for i in range(30):
            time.sleep(1)
            sz    = FEED_FILE.stat().st_size if FEED_FILE.exists() else 0
            lines = FEED_FILE.read_text().splitlines() if FEED_FILE.exists() else []
            print(f"\r  t+{i+1:02d}s | {len(lines)} items | {sz} bytes | src: {active_source():<30}", end="", flush=True)
    except KeyboardInterrupt:
        pass

    stop_news_feed()
    time.sleep(1)

    print(f"\n\nFinal feed ({FEED_FILE}):")
    if FEED_FILE.exists():
        lines = FEED_FILE.read_text().splitlines()
        print(f"  {len(lines)} ECB items written.")
        for line in lines:
            obj = json.loads(line)
            print(f"  [{obj['chunk_index']:02d}] {obj['text'][:100]}")
    else:
        print("  No items — no ECB press conference active right now. Expected.")
    print("\nResult:", "WORKING" if FEED_FILE.exists() else "WAITING FOR ECB EVENT")
