"""
Macro data retrieval for 17 target variables and their regressors.

Sources: FRED (bulk), ECB SDW, IPEA, Bank of Canada Valet.
All series cached to alternative_models/data/macro_raw/.
"""

import os
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
MACRO_RAW = DATA_DIR / "macro_raw"
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =====================================================================
# FRED bulk fetcher
# =====================================================================

FRED_SERIES = {
    # US macro
    "NFP": "PAYEMS",
    "UNRATE": "UNRATE",
    "AHE": "CES0500000003",
    "CLAIMS": "ICSA",
    "UMCSENT": "UMCSENT",
    "MICH_INF_EXP": "MICH",
    "CPI": "CPIAUCSL",
    "ISM_EMP": "MANEMP",  # Manufacturing employment (proxy for ISM emp component)
    "FED_RATE": "DFF",
    "PRODUCTIVITY": "OPHNFB",
    "PARTICIPATION": "CIVPART",
    "SP500": "SP500",
    "GAS_PRICE": "GASREGW",
    "OIL_WTI": "DCOILWTICO",
    # NFP-specific predictors (literature-grounded)
    "CONT_CLAIMS": "CCSA",           # Continued claims (weekly)
    "TEMP_HELP": "TEMPHELPS",        # Temp help services employment (monthly)
    "AWH_MFG": "AWHMAN",            # Avg weekly hours manufacturing (monthly)
    "CONSUMER_CONF": "CSCICP03USM665S",  # Consumer confidence (monthly, OECD)
    "VIX": "VIXCLS",                # VIX daily
    "CREDIT_SPREAD": "BAAFFM",      # BAA-Fed Funds spread (daily)
    "YIELD_SPREAD": "T10Y2Y",       # 10Y-2Y spread (daily)
    # Weekly macro targets
    "PPI": "PPIACO",                 # PPI Final Demand (monthly)
    "RETAIL": "RSXFS",               # Retail Sales ex-auto (monthly)
    "UK_UNEMP": "LRHUTTTTGBM156S",  # UK harmonised unemployment rate (monthly, OECD)
    "UK_WAGES": "LCRENT02GBM156S",  # UK hourly earnings manufacturing (monthly, OECD)
    # FX
    "EURUSD": "DEXUSEU",
    "USDMXN": "DEXMXUS",
    "USDBRL": "DEXBZUS",
    "CADUSD": "DEXCAUS",
    # International via FRED/OECD
    "DE_IP": "DEUPROINDMISMEI",
    "UK_BOE_RATE": "INTDSRGBM193N",  # UK interest rate (discount/bank rate)
    "BR_SELIC": "IRSTCB01BRM156N",
    "CA_BOC_RATE": "IRSTCB01CAM156N",
    "CA_UNEMP": "LRUNTTTTCAM156S",
}


def fetch_fred_bulk(
    start: str = "2000-01-01",
    end: str = "2026-05-01",
    api_key: str | None = None,
    cache_path: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch all FRED series in FRED_SERIES dict. Returns wide DataFrame."""
    if cache_path is None:
        cache_path = MACRO_RAW / "fred_bulk.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY required")

    from fredapi import Fred
    fred = Fred(api_key=api_key)

    series_data = {}
    for name, code in FRED_SERIES.items():
        print(f"  FRED: {name} ({code})...")
        for attempt in range(3):
            try:
                s = fred.get_series(code, observation_start=start, observation_end=end)
                series_data[name] = s
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    FAILED: {e}")

    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"FRED bulk cached: {df.shape}")
    return df


# =====================================================================
# ECB SDW fetcher
# =====================================================================

ECB_SERIES = {
    "EZ_RETAIL_MOM": "STS.M.I8.Y.RETT.NS0020.4.000",
    "EZ_HICP": "ICP.M.U2.N.000000.4.ANR",
    "EZ_UNEMP": "STS.M.I8.S.UNEH.RTT009.4.000",
    "ECB_RATE": "FM.M.U2.EUR.4F.KR.MRR_FR.LEV",
}


def fetch_ecb_series(
    series_key: str,
    start: str = "2000-01-01",
    end: str = "2026-05-01",
) -> pd.Series:
    """Fetch a single ECB SDW series via REST CSV."""
    url = f"https://data-api.ecb.europa.eu/service/data/{series_key}"
    params = {"startPeriod": start[:7], "endPeriod": end[:7], "format": "csvdata"}
    resp = requests.get(url, params=params, timeout=60)
    if resp.status_code != 200:
        print(f"    ECB {series_key}: HTTP {resp.status_code}")
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(resp.text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.Series(dtype=float)
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
    s = df.set_index("TIME_PERIOD")["OBS_VALUE"].astype(float)
    s.index.name = "Date"
    return s


def fetch_ecb_bulk(
    start: str = "2000-01-01",
    end: str = "2026-05-01",
    cache_path: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch all ECB series."""
    if cache_path is None:
        cache_path = MACRO_RAW / "ecb_bulk.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    series_data = {}
    for name, key in ECB_SERIES.items():
        print(f"  ECB: {name}...")
        try:
            s = fetch_ecb_series(key, start, end)
            if len(s) > 0:
                series_data[name] = s
        except Exception as e:
            print(f"    FAILED: {e}")

    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"ECB bulk cached: {df.shape}")
    return df


# =====================================================================
# IPEA fetcher (Brazil)
# =====================================================================

IPEA_SERIES = {
    "BR_IPCA": "PRECOS12_IPCAG12",
    "BR_IP": "PAN12_QIIG12",
    "BR_TRADE": "BM12_BCSALDO12",
}


def fetch_ipea_series(series_code: str) -> pd.Series:
    """Fetch single IPEA series."""
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{series_code}')"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("value", [])
    dates, values = [], []
    for rec in records:
        val = rec.get("VALVALOR")
        if val is not None:
            dates.append(pd.to_datetime(rec["VALDATA"][:10]))
            values.append(float(val))
    return pd.Series(values, index=dates, dtype=float)


def fetch_ipea_bulk(
    cache_path: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch all IPEA Brazil series."""
    if cache_path is None:
        cache_path = MACRO_RAW / "ipea_bulk.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    series_data = {}
    for name, code in IPEA_SERIES.items():
        print(f"  IPEA: {name} ({code})...")
        try:
            s = fetch_ipea_series(code)
            if len(s) > 0:
                series_data[name] = s
        except Exception as e:
            print(f"    FAILED: {e}")

    df = pd.DataFrame(series_data)
    df.index.name = "Date"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"IPEA bulk cached: {df.shape}")
    return df


# =====================================================================
# Load existing project data (yields + NS factors)
# =====================================================================

def load_ns_factors() -> pd.DataFrame:
    """Load daily NS factors from the main project."""
    path = PROJECT_ROOT / "data" / "factors" / "ns_factors_5c.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_yields() -> pd.DataFrame:
    """Load daily aligned yields from the main project."""
    path = PROJECT_ROOT / "data" / "raw" / "aligned_yields_5c.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


# =====================================================================
# Transform utilities
# =====================================================================

def compute_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    """
    Apply transforms to columns.
    transforms: {col_name: "diff" | "pct_change" | "yoy" | "log_diff" | None}
    """
    result = pd.DataFrame(index=df.index)
    for col, transform in transforms.items():
        if col not in df.columns:
            continue
        s = df[col]
        if transform is None or transform == "level":
            result[col] = s
        elif transform == "diff":
            result[col] = s.diff()
        elif transform == "pct_change":
            result[col] = s.pct_change() * 100
        elif transform == "yoy":
            result[col] = s.pct_change(12) * 100
        elif transform == "log_diff":
            result[col] = np.log(s).diff() * 100
        else:
            result[col] = s
    return result


def compute_ar1_consensus(
    series: pd.Series,
    window: int = 36,
    min_obs: int = 24,
) -> pd.DataFrame:
    """
    Compute rolling AR(1) pseudo-consensus forecast.

    For each date t, fit AR(1) on [t-window, t-1] and forecast t.
    Returns DataFrame with columns: actual, forecast, surprise.
    """
    results = []
    vals = series.dropna()

    for i in range(window, len(vals)):
        train = vals.iloc[max(0, i - window):i]
        if len(train) < min_obs:
            continue
        # AR(1): y_t = a + b * y_{t-1}
        y = train.values[1:]
        x = train.values[:-1]
        if len(y) < 2:
            continue
        # OLS: b = cov(x,y)/var(x), a = mean(y) - b*mean(x)
        b = np.cov(x, y)[0, 1] / (np.var(x) + 1e-10)
        a = np.mean(y) - b * np.mean(x)
        forecast = a + b * train.values[-1]
        actual = vals.iloc[i]
        results.append({
            "Date": vals.index[i],
            "actual": actual,
            "forecast": forecast,
            "surprise": actual - forecast,
        })

    return pd.DataFrame(results).set_index("Date")


# =====================================================================
# Master fetch
# =====================================================================

def fetch_all_macro(
    start: str = "2000-01-01",
    end: str = "2026-05-01",
    force: bool = False,
) -> dict:
    """
    Fetch all macro data from all sources. Returns dict of DataFrames.
    """
    print("=== Fetching FRED ===")
    fred = fetch_fred_bulk(start=start, end=end, force=force)

    print("\n=== Fetching ECB ===")
    ecb = fetch_ecb_bulk(start=start, end=end, force=force)

    print("\n=== Fetching IPEA ===")
    ipea = fetch_ipea_bulk(force=force)

    print("\n=== Loading NS factors ===")
    ns = load_ns_factors()

    print("\n=== Loading yields ===")
    yields = load_yields()

    return {
        "fred": fred,
        "ecb": ecb,
        "ipea": ipea,
        "ns_factors": ns,
        "yields": yields,
    }
