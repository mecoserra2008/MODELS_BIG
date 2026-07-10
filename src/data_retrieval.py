"""
Data retrieval module for UK, US, and EU yield curve data.

Sources:
- UK: Bank of England nominal spot curve (Excel download)
- US: FRED Treasury constant-maturity rates
- EU: ECB AAA-rated government bond spot rates (REST API)
"""

import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


# ---------------------------------------------------------------------------
# UK Gilts — Bank of England
# ---------------------------------------------------------------------------

# IADB series codes for nominal spot rates (zero-coupon) at specific maturities
# These are from the GLC Nominal dataset published via the IADB
_BOE_SPOT_SERIES = {
    0.5: "IUDSNZC",  # 6 months
    1: "IUDNZC1",    # 1 year (alternative codes may apply)
    2: "IUDNZC2",
    3: "IUDNZC3",
    5: "IUDNZC5",
    7: "IUDNZC7",
    10: "IUDNZC10",
    15: "IUDNZC15",
    20: "IUDNZC20",
    25: "IUDNZC25",
}


def _fetch_uk_yields_iadb(
    start_date: str,
    end_date: str,
    maturities: list[float],
    cache_path: Path,
) -> pd.DataFrame:
    """
    Fallback: fetch UK yields from the BoE ZIP archive of daily nominal
    government liability curve data.

    The ZIP contains multiple Excel files split by year range.
    We read each relevant file's spot curve sheet and concatenate.
    """
    import zipfile

    zip_url = (
        "https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
        "yield-curves/glcnominalddata.zip"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    print(f"  Downloading BoE daily nominal curve (ZIP ~38MB)...")
    resp = requests.get(zip_url, headers=headers, timeout=300)
    resp.raise_for_status()

    start_year = int(start_date[:4])
    all_dfs = []

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xlsx_names = sorted(
            n for n in zf.namelist() if n.endswith(('.xlsx', '.xls'))
        )
        for xlsx_name in xlsx_names:
            # Check if this file covers our date range
            # Filenames like "GLC Nominal daily data_2000 to 2004.xlsx"
            import re
            years = re.findall(r'\d{4}', xlsx_name)
            if years:
                file_end_year = int(years[-1]) if years[-1] != "present" else 9999
                # "present" won't match \d{4}, so last matched year is the start
                # For "2025 to present", years = ['2025']
                file_start_year = int(years[0])
                if len(years) > 1:
                    file_end_year = int(years[-1])
                else:
                    file_end_year = 9999  # "to present"
            else:
                file_start_year, file_end_year = 0, 9999

            # Skip files entirely before our start date
            if file_end_year < start_year:
                continue

            print(f"  Reading: {xlsx_name}")
            xlsx_data = zf.read(xlsx_name)

            try:
                df_part = _parse_boe_spot_excel(xlsx_data, maturities)
                if df_part is not None and len(df_part) > 0:
                    all_dfs.append(df_part)
            except Exception as e:
                print(f"    Warning: failed to parse {xlsx_name}: {e}")
                continue

    if not all_dfs:
        raise RuntimeError("Could not parse any BoE yield curve files from ZIP")

    df = pd.concat(all_dfs)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.loc[start_date:end_date]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"  UK yields cached — shape {df.shape}")
    return df


def _parse_boe_spot_excel(
    xlsx_data: bytes,
    maturities: list[float],
) -> pd.DataFrame | None:
    """Parse a single BoE yield curve Excel file and extract spot rates."""
    xls = pd.ExcelFile(io.BytesIO(xlsx_data), engine="openpyxl")

    # Find the spot curve sheet (prefer one without 'short end' in name)
    spot_sheet = None
    for name in xls.sheet_names:
        name_lower = name.lower()
        if "spot" in name_lower and "short" not in name_lower:
            spot_sheet = name
            break
    # If only short-end spot sheet exists, use it
    if spot_sheet is None:
        for name in xls.sheet_names:
            if "spot" in name.lower():
                spot_sheet = name
                break
    if spot_sheet is None:
        return None

    # Read with header=3 (most common for BoE files)
    # The structure is: rows 0-2 are metadata, row 3 has maturity headers
    df_raw = pd.read_excel(
        xls, sheet_name=spot_sheet, header=3, index_col=0
    )

    # Clean up index (dates)
    df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
    df_raw = df_raw[df_raw.index.notna()]
    df_raw = df_raw.apply(pd.to_numeric, errors="coerce")

    if df_raw.empty:
        return None

    # Map columns to float maturities
    available_cols = []
    for col in df_raw.columns:
        try:
            val = float(str(col).strip())
            available_cols.append((val, col))
        except (ValueError, TypeError):
            continue

    if not available_cols:
        return None

    selected = {}
    for m in maturities:
        best = min(available_cols, key=lambda x: abs(x[0] - m))
        # Only accept if within 0.5 years of requested maturity
        if abs(best[0] - m) <= 0.5:
            label = f"UK_{int(m)}Y" if m == int(m) else f"UK_{m}Y"
            selected[label] = df_raw[best[1]]

    if not selected:
        return None

    return pd.DataFrame(selected)


def fetch_uk_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    maturities: list[float] = [1, 2, 3, 5, 10, 20],
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download UK nominal zero-coupon (spot) rates from the Bank of England
    yield curve dataset (GLC Nominal).

    The BoE publishes a large Excel file with daily spot curves at half-yearly
    maturity intervals. We read the '4. spot curve' sheet and select columns
    for the requested maturities.

    Parameters
    ----------
    start_date : str
        Start of date range (inclusive).
    end_date : str
        End of date range (inclusive).
    maturities : list of float
        Maturities in years to extract (e.g., [1, 2, 3, 5, 10, 20]).
    cache_path : str or None
        Path to save/load cached CSV. Defaults to data/raw/uk_gilts.csv.
    force_download : bool
        If True, re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns like 'UK_1Y', 'UK_2Y', etc.
    """
    if cache_path is None:
        cache_path = RAW_DIR / "uk_gilts.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    url = (
        "https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
        "yield-curves/glcnominaldata.xlsx"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    print(f"Downloading BoE yield curve data from:\n  {url}")
    resp = requests.get(url, headers=headers, timeout=120)

    if resp.status_code != 200:
        print(f"  Excel download failed ({resp.status_code}). Falling back to ZIP archive...")
        return _fetch_uk_yields_iadb(start_date, end_date, maturities, cache_path)

    # The Excel file has multiple sheets; '4. spot curve' contains nominal spots
    # Header rows vary — typically row 4 (0-indexed) has maturity labels
    xls = pd.ExcelFile(io.BytesIO(resp.content), engine="openpyxl")

    # Find the spot curve sheet (name varies slightly across vintages)
    spot_sheet = None
    for name in xls.sheet_names:
        if "spot" in name.lower():
            spot_sheet = name
            break
    if spot_sheet is None:
        raise ValueError(
            f"Could not find spot curve sheet. Available: {xls.sheet_names}"
        )

    # Read with header detection — first column is date
    raw = pd.read_excel(
        xls, sheet_name=spot_sheet, header=None, dtype=str
    )

    # Find the header row: look for a row containing "years" or numeric maturity labels
    header_idx = None
    for i in range(min(20, len(raw))):
        row_vals = raw.iloc[i].astype(str).str.strip().tolist()
        # Check if row has many numeric-like values (maturity tenors)
        numeric_count = sum(
            1 for v in row_vals[1:]
            if v.replace(".", "").replace(",", "").isdigit()
        )
        if numeric_count > 5:
            header_idx = i
            break

    if header_idx is None:
        # Fallback: use row 3 (common in BoE files)
        header_idx = 3

    # Re-read with proper header
    df_raw = pd.read_excel(
        xls, sheet_name=spot_sheet, header=header_idx, index_col=0
    )

    # Index should be dates
    df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
    df_raw = df_raw[df_raw.index.notna()]
    df_raw = df_raw.apply(pd.to_numeric, errors="coerce")

    # Column headers are maturity tenors (as floats or strings like '0.5', '1', ...)
    # Map requested maturities to closest available columns
    available_cols = []
    for col in df_raw.columns:
        try:
            available_cols.append((float(col), col))
        except (ValueError, TypeError):
            continue

    selected = {}
    for m in maturities:
        # Find closest available maturity
        best_col = min(available_cols, key=lambda x: abs(x[0] - m))
        col_label = f"UK_{int(m)}Y" if m == int(m) else f"UK_{m}Y"
        selected[col_label] = df_raw[best_col[1]]

    df = pd.DataFrame(selected)
    df = df.sort_index()
    df = df.loc[start_date:end_date]

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"UK yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# US Treasuries — FRED
# ---------------------------------------------------------------------------

FRED_SERIES = {
    1: "DGS1",
    2: "DGS2",
    3: "DGS3",
    5: "DGS5",
    10: "DGS10",
    20: "DGS20",
}


def fetch_us_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    api_key: str | None = None,
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download US Treasury constant-maturity rates from FRED.

    Parameters
    ----------
    start_date, end_date : str
        Date range.
    api_key : str or None
        FRED API key. If None, reads from FRED_API_KEY env var.
    cache_path : str or None
        Defaults to data/raw/us_treasuries.csv.
    force_download : bool
        Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns 'US_1Y', 'US_2Y', etc.
    """
    if cache_path is None:
        cache_path = RAW_DIR / "us_treasuries.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED API key required. Set FRED_API_KEY env variable or pass api_key. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    import time
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series_data = {}

    for maturity, series_id in FRED_SERIES.items():
        print(f"  Fetching FRED series {series_id}...")
        for attempt in range(3):
            try:
                s = fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
                series_data[f"US_{maturity}Y"] = s
                break
            except Exception as e:
                if attempt < 2:
                    print(f"    Retry {attempt+1} for {series_id} ({e})")
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    FAILED {series_id} after 3 attempts: {e}")
                    raise

    df = pd.DataFrame(series_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"US yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# EU AAA Yields — ECB Statistical Data Warehouse
# ---------------------------------------------------------------------------

def fetch_eu_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    maturities: list[int] = [1, 2, 3, 5, 10, 20],
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download Euro area AAA-rated government bond spot rates from the ECB.

    Uses the ECB Data Portal REST API (SDMX-based) with CSV output.
    Dataflow: YC (Yield Curve)
    Key: B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}Y

    Parameters
    ----------
    start_date, end_date : str
        Date range.
    maturities : list of int
        Maturities in years.
    cache_path : str or None
        Defaults to data/raw/eu_aaa_yields.csv.
    force_download : bool
        Re-download even if cache exists.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns 'EU_1Y', 'EU_2Y', etc.
    """
    if cache_path is None:
        cache_path = RAW_DIR / "eu_aaa_yields.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    base_url = (
        "https://data-api.ecb.europa.eu/service/data/YC/"
        "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{m}Y"
    )

    series_data = {}
    for m in maturities:
        url = base_url.format(m=m)
        params = {
            "startPeriod": start_date,
            "endPeriod": end_date,
            "format": "csvdata",
        }
        print(f"  Fetching ECB YC SR_{m}Y...")
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()

        ecb_df = pd.read_csv(io.StringIO(resp.text))
        # ECB CSV has columns: TIME_PERIOD, OBS_VALUE, plus metadata
        ecb_df["TIME_PERIOD"] = pd.to_datetime(ecb_df["TIME_PERIOD"])
        ecb_df = ecb_df.set_index("TIME_PERIOD")
        series_data[f"EU_{m}Y"] = ecb_df["OBS_VALUE"].astype(float)

    df = pd.DataFrame(series_data)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.loc[start_date:end_date]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"EU yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Per-country Euro-area government bonds (IT / FR / ES) — stooq daily CSV
# ---------------------------------------------------------------------------

# stooq publishes daily government-bond yields as free CSV. The symbol format
# is not officially documented and has varied, so we try a few candidate
# spellings per (country, maturity) and keep the first that returns numeric
# data. NOTE: this endpoint is unofficial and per-country 5Y history/coverage
# is uneven — a leg that fails to resolve is skipped with a warning (the market
# is then dropped downstream) rather than aborting the run. For production, a
# paid feed (Bloomberg / Refinitiv) is the recommended upgrade.
EGB_COUNTRIES = {
    "IT": "it",  # Italy — BTP
    "FR": "fr",  # France — OAT
    "ES": "es",  # Spain — Bonos
}


def _stooq_symbol_candidates(cc: str, maturity: int) -> list[str]:
    """Candidate stooq yield symbols for a country code / maturity in years."""
    return [
        f"{maturity}{cc}y.b",   # e.g. 10ity.b
        f"{maturity}y{cc}.b",   # e.g. 10yit.b
        f"{maturity}{cc}by.b",  # e.g. 10itby.b
    ]


def _fetch_stooq_series(symbol: str, timeout: int = 60) -> pd.Series | None:
    """Download one stooq daily series; return Close as a Series or None."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:  # network / HTTP error
        print(f"    stooq request failed for {symbol}: {e}")
        return None

    text = resp.text.strip()
    # stooq returns the literal "No data" (or an HTML page) for bad symbols.
    if not text or "No data" in text or not text.lower().startswith("date"):
        return None

    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        return None
    if "Date" not in raw.columns or "Close" not in raw.columns:
        return None

    s = pd.Series(
        pd.to_numeric(raw["Close"], errors="coerce").values,
        index=pd.to_datetime(raw["Date"]),
    ).dropna()
    return s if len(s) > 0 else None


def fetch_egb_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    countries: dict[str, str] | None = None,
    maturities: list[int] = [5, 10],
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download per-country euro-area government bond yields (Italy BTP, France
    OAT, Spain Bonos) at the 5Y and 10Y tenors from stooq daily CSV.

    Returns a DataFrame with the {CC}_{TENOR} convention, e.g. columns
    IT_5Y, IT_10Y, FR_5Y, FR_10Y, ES_5Y, ES_10Y. Legs whose stooq symbol
    cannot be resolved are omitted (a warning is printed); downstream code
    should drop any market missing a leg.
    """
    if countries is None:
        countries = EGB_COUNTRIES
    if cache_path is None:
        cache_path = RAW_DIR / "egb_yields.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    import time

    series_data: dict[str, pd.Series] = {}
    for prefix, cc in countries.items():
        for m in maturities:
            label = f"{prefix}_{m}Y"
            found = None
            for symbol in _stooq_symbol_candidates(cc, m):
                print(f"  Fetching stooq {symbol} ({label})...")
                found = _fetch_stooq_series(symbol)
                if found is not None:
                    series_data[label] = found
                    break
                time.sleep(0.5)
            if found is None:
                print(f"    WARNING: no stooq data for {label} — leg skipped")

    if not series_data:
        print("  WARNING: fetch_egb_yields resolved no series; returning empty frame")
        return pd.DataFrame()

    df = pd.DataFrame(series_data)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.loc[start_date:end_date]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"EGB yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canada — Bank of Canada
# ---------------------------------------------------------------------------

BOC_SERIES = {
    2: "BD.CDN.2YR.DQ.YLD",
    3: "BD.CDN.3YR.DQ.YLD",
    5: "BD.CDN.5YR.DQ.YLD",
    7: "BD.CDN.7YR.DQ.YLD",
    10: "BD.CDN.10YR.DQ.YLD",
    30: "BD.CDN.LONG.DQ.YLD",
}


def fetch_ca_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download Canadian government benchmark bond yields from Bank of Canada
    Valet API.

    Available maturities: 2Y, 3Y, 5Y, 7Y, 10Y, 30Y (Long-term).

    Returns DataFrame with columns CA_2Y, CA_3Y, CA_5Y, CA_7Y, CA_10Y, CA_30Y.
    """
    if cache_path is None:
        cache_path = RAW_DIR / "ca_yields.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    import time

    series_data = {}
    for maturity, series_id in BOC_SERIES.items():
        url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/csv"
        params = {"start_date": start_date, "end_date": end_date}
        print(f"  Fetching BoC {series_id}...")

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        # Parse BoC CSV: skip header lines until "OBSERVATIONS"
        lines = resp.text.split("\n")
        obs_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('"OBSERVATIONS"') or line.strip() == "OBSERVATIONS":
                obs_start = i + 1
                break

        if obs_start is None:
            raise ValueError(f"Could not find OBSERVATIONS section in BoC response for {series_id}")

        # Read from observations header onward
        obs_text = "\n".join(lines[obs_start:])
        obs_df = pd.read_csv(io.StringIO(obs_text))
        obs_df.columns = ["Date", "Value"]
        obs_df["Date"] = pd.to_datetime(obs_df["Date"])
        obs_df = obs_df.set_index("Date")
        obs_df["Value"] = pd.to_numeric(obs_df["Value"], errors="coerce")

        label = f"CA_{maturity}Y"
        series_data[label] = obs_df["Value"]

    df = pd.DataFrame(series_data)
    df = df.sort_index()
    df = df.loc[start_date:end_date]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"CA yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Brazil — IPEA / ANBIMA DI Curve
# ---------------------------------------------------------------------------

# ANBIMA pre-fixed (DI) yield curve tenors via IPEA API
# Series: ANBIMA366_TJTLNX366 where X = years
IPEA_BR_SERIES = {
    1: "ANBIMA366_TJTLN1366",
    3: "ANBIMA366_TJTLN3366",
    6: "ANBIMA366_TJTLN6366",
    12: "ANBIMA366_TJTLN12366",
}


def fetch_br_yields(
    start_date: str = "2000-01-01",
    end_date: str = "2026-04-30",
    cache_path: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download Brazilian pre-fixed (DI) yield curve from IPEA/ANBIMA.

    Available tenors: 1Y, 3Y, 6Y, 12Y.
    Yields are in annual DI convention (252 business day compounding),
    expressed in percent (e.g., 14.3 = 14.3%).

    Returns DataFrame with columns BR_1Y, BR_3Y, BR_6Y, BR_12Y.
    """
    if cache_path is None:
        cache_path = RAW_DIR / "br_yields.csv"
    else:
        cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.loc[start_date:end_date]

    import time

    series_data = {}
    for maturity, series_code in IPEA_BR_SERIES.items():
        url = (
            f"http://www.ipeadata.gov.br/api/odata4/"
            f"ValoresSerie(SERCODIGO='{series_code}')"
        )
        print(f"  Fetching IPEA {series_code} ({maturity}Y)...")

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

        data = resp.json()
        records = data.get("value", [])

        dates = []
        values = []
        for rec in records:
            val = rec.get("VALVALOR")
            if val is not None:
                dt = pd.to_datetime(rec["VALDATA"][:10])
                dates.append(dt)
                values.append(float(val))

        label = f"BR_{maturity}Y"
        series_data[label] = pd.Series(values, index=dates, name=label)

    df = pd.DataFrame(series_data)
    df.index.name = "Date"
    df = df.sort_index()
    df = df.loc[start_date:end_date]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"BR yields cached to {cache_path} — shape {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Alignment (generalized)
# ---------------------------------------------------------------------------

def align_yield_data(
    *yield_dfs: pd.DataFrame,
    freq: str | None = None,
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """
    Merge any number of yield DataFrames on their date index.

    Parameters
    ----------
    *yield_dfs : pd.DataFrame
        Individual yield DataFrames with DatetimeIndex.
        Column names must follow the {CC}_{TENOR} convention
        (e.g., UK_10Y, US_5Y, BR_3Y).
    freq : str or None
        If provided, resample to this frequency (e.g., 'W-FRI' for weekly).
    ffill_limit : int
        Maximum number of consecutive NaN to forward-fill (handles holidays).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all columns, NaN-filled gaps handled.
    """
    combined = pd.concat(yield_dfs, axis=1)
    combined = combined.sort_index()

    # Forward-fill small gaps from different holiday calendars
    combined = combined.ffill(limit=ffill_limit)

    if freq is not None:
        combined = combined.resample(freq).last()

    # Auto-detect region prefixes from column names
    prefixes = sorted(set(c.split("_")[0] for c in combined.columns))

    # Keep rows where every region has at least one non-NaN value
    mask = pd.Series(True, index=combined.index)
    for prefix in prefixes:
        cols = [c for c in combined.columns if c.startswith(f"{prefix}_")]
        if cols:
            mask = mask & combined[cols].notna().any(axis=1)
    combined = combined[mask]

    return combined
