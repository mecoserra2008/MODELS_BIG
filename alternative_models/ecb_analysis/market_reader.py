"""
Bund futures (FGBL/RX1) live price reader.
Three sources tried in cascade:
  1. TwelveData WebSocket (free tier)
  2. EODHD WebSocket (public demo token)
  3. Stooq HTTP fast-poll (no API key)

Falls back to a manual price entry interface when all sources fail.
"""

from __future__ import annotations
import json
import sys
import time
import threading
import queue
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

# ---------------------------------------------------------------------------
# Shared price queue — consumers read from here
# ---------------------------------------------------------------------------
_price_queue: queue.Queue[dict] = queue.Queue(maxsize=500)

# State visible to Streamlit
_current_price: dict = {
    "price": None,
    "prev_price": None,
    "change_bps": None,
    "source": "none",
    "timestamp": None,
}
_lock = threading.Lock()


def get_current_price() -> dict:
    with _lock:
        return dict(_current_price)


def _update_state(price: float, source: str):
    with _lock:
        prev = _current_price.get("price")
        _current_price["prev_price"] = prev
        _current_price["price"] = price
        _current_price["source"] = source
        _current_price["timestamp"] = datetime.now(timezone.utc).isoformat()
        if prev is not None:
            # Bund futures: 1 tick = 0.01 ≈ 1 bp DV01 approximation
            _current_price["change_bps"] = round((price - prev) * 100, 1)
        _price_queue.put_nowait(dict(_current_price))


def manual_update(price: float):
    """Called from Streamlit when trader types a price."""
    _update_state(price, "manual")


# ---------------------------------------------------------------------------
# Source 1: TwelveData WebSocket
# ---------------------------------------------------------------------------
_TD_SYMBOLS = ["FGBL", "DE10Y", "EURUSD"]  # tried in order

def _twelvedata_ws(api_key: str, on_price: Callable[[float, str], None]):
    try:
        import websocket
    except ImportError:
        return

    connected = threading.Event()
    symbol_tried = 0

    def _on_message(ws, message):
        try:
            data = json.loads(message)
            if data.get("event") == "price":
                p = float(data.get("price", 0))
                if p > 0:
                    on_price(p, f"TwelveData:{data.get('symbol','')}")
        except Exception:
            pass

    def _on_error(ws, error):
        pass

    def _on_open(ws):
        sym = _TD_SYMBOLS[0]
        ws.send(json.dumps({"action": "subscribe", "params": {"symbols": sym}}))
        connected.set()

    url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={api_key}"
    ws = websocket.WebSocketApp(url, on_message=_on_message, on_error=_on_error, on_open=_on_open)
    ws.run_forever(ping_interval=30, ping_timeout=10)


def start_twelvedata(api_key: str):
    t = threading.Thread(target=_twelvedata_ws, args=(api_key, _update_state), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Source 2: EODHD WebSocket (public demo token)
# ---------------------------------------------------------------------------
_EODHD_DEMO_TOKEN = "OeAFFmMliFG5orCUuwAKQ8l4WWSH49Qn"
_EODHD_SYMBOLS = ["FGBL.EU", "BUND.EUREX", "GBL.EUREX"]

def _eodhd_ws(on_price: Callable[[float, str], None]):
    try:
        import websocket
    except ImportError:
        return

    def _on_message(ws, message):
        try:
            data = json.loads(message)
            p = float(data.get("p", 0) or data.get("close", 0) or data.get("last", 0))
            if p > 0:
                on_price(p, f"EODHD:{data.get('s','')}")
        except Exception:
            pass

    def _on_open(ws):
        for sym in _EODHD_SYMBOLS:
            ws.send(json.dumps({"action": "subscribe", "symbols": sym}))

    url = f"wss://ws.eodhistoricaldata.com/ws/futures?api_token={_EODHD_DEMO_TOKEN}"
    try:
        import websocket
        ws = websocket.WebSocketApp(url, on_message=_on_message, on_open=_on_open)
        ws.run_forever(ping_interval=30, ping_timeout=10)
    except Exception:
        pass


def start_eodhd():
    t = threading.Thread(target=_eodhd_ws, args=(_update_state,), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Source 3: Stooq HTTP fast-poll (5-second interval, no API key)
# ---------------------------------------------------------------------------
_STOOQ_SYMBOLS = ["fgbl", "gblu26", "rxa"]  # FGBL, front-month, alternative tickers
_STOOQ_BASE = "https://stooq.com/q/l/"

def _stooq_poll(stop_event: threading.Event, on_price: Callable[[float, str], None]):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/124.0 Safari/537.36",
        "Accept": "text/csv,*/*",
    }
    session = requests.Session()
    session.headers.update(headers)

    while not stop_event.is_set():
        for sym in _STOOQ_SYMBOLS:
            try:
                url = f"{_STOOQ_BASE}?s={sym}&f=sd2t2ohlcv&e=csv"
                resp = session.get(url, timeout=8)
                if resp.status_code == 200:
                    lines = resp.text.strip().splitlines()
                    if len(lines) >= 2:
                        parts = lines[1].split(",")
                        # Format: Symbol,Date,Time,Open,High,Low,Close,Volume
                        if len(parts) >= 7:
                            close_str = parts[6].strip()
                            if close_str and close_str != "N/D":
                                price = float(close_str)
                                if price > 0:
                                    on_price(price, f"Stooq:{sym.upper()}")
                                    break
            except Exception:
                continue
        stop_event.wait(5)


_stooq_stop = threading.Event()

def start_stooq():
    t = threading.Thread(target=_stooq_poll, args=(_stooq_stop, _update_state), daemon=True)
    t.start()
    return t


def stop_stooq():
    _stooq_stop.set()


# ---------------------------------------------------------------------------
# ECB SDW historical yield (reuses data_retrieval.py pattern)
# ---------------------------------------------------------------------------
def fetch_ecb_bund_history(n_days: int = 90) -> list[dict]:
    """
    Fetch German 10Y Bund yield from ECB Statistical Data Warehouse.
    Uses the same base URL pattern as alternative_models/src/data_retrieval.py.
    Returns list of {date, yield_pct} dicts.
    """
    # ECB SDW: IRS.M.DE.L.L40.CI.0.EUR.2Y.X (10Y Bund yield, monthly)
    # For daily: use IRTS or government bond daily series
    url = (
        "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
        "?startPeriod=2024-01-01&format=jsondata&lastNObservations=180"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        js = resp.json()
        obs = js["dataSets"][0]["series"]["0:0:0:0:0:0:0"]["observations"]
        times = js["structure"]["dimensions"]["observation"][0]["values"]
        result = []
        for i, t in enumerate(times):
            val = obs.get(str(i), [None])[0]
            if val is not None:
                result.append({"date": t["id"], "yield_pct": round(float(val), 4)})
        return result[-n_days:]
    except Exception as exc:
        print(f"[market_reader] ECB SDW fetch failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Auto-start: try sources in order
# ---------------------------------------------------------------------------
def autostart(twelvedata_key: str | None = None):
    """
    Try WebSocket sources first, then fall back to Stooq polling.
    Called once at app startup.
    """
    if twelvedata_key:
        start_twelvedata(twelvedata_key)
        time.sleep(3)
        with _lock:
            if _current_price["price"] is not None:
                print("[market_reader] TwelveData WebSocket active.")
                return

    start_eodhd()
    time.sleep(4)
    with _lock:
        if _current_price["price"] is not None:
            print("[market_reader] EODHD WebSocket active.")
            return

    print("[market_reader] Falling back to Stooq HTTP poll.")
    start_stooq()
