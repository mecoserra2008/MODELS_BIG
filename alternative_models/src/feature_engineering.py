"""
Feature engineering for MIDAS and XGBoost macro models.

Provides:
- Beta polynomial MIDAS weight functions
- High-frequency to low-frequency feature alignment
- XGBoost rolling statistic features
- Target variable registry with regressor specifications
"""

import numpy as np
import pandas as pd
from scipy.special import beta as beta_func


# =====================================================================
# Beta polynomial weights (Ghysels et al. 2007)
# =====================================================================

def beta_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """
    Compute normalized Beta polynomial weights for MIDAS.

    w(k) = f(k/K; theta1, theta2) / sum_j f(j/K; theta1, theta2)
    f(x; a, b) = x^(a-1) * (1-x)^(b-1)

    Parameters
    ----------
    K : int
        Number of high-frequency lags.
    theta1, theta2 : float
        Beta distribution shape parameters (must be > 1.0 for well-behaved weights).

    Returns
    -------
    np.ndarray of shape (K,), normalized weights summing to 1.
    Weights are ordered: w[0] = most recent, w[K-1] = most distant.
    """
    # Grid: k=1..K mapped to (0, 1) interval
    x = np.linspace(1 / (K + 1), K / (K + 1), K)
    log_w = (theta1 - 1) * np.log(x + 1e-10) + (theta2 - 1) * np.log(1 - x + 1e-10)
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def almon_weights(K: int, c: np.ndarray) -> np.ndarray:
    """
    Exponential Almon polynomial weights (fallback).

    w(k) = exp(c0 + c1*k + c2*k^2 + ...) / sum(exp(...))

    Parameters
    ----------
    K : int
        Number of lags.
    c : array-like
        Polynomial coefficients (typically length 2 or 3).

    Returns
    -------
    np.ndarray of shape (K,), normalized weights.
    """
    k = np.arange(K, dtype=float)
    poly = sum(c[j] * k**j for j in range(len(c)))
    poly -= poly.max()
    w = np.exp(poly)
    return w / w.sum()


# =====================================================================
# MIDAS feature construction
# =====================================================================

def align_hf_to_lf(
    hf_series: pd.Series,
    lf_dates: pd.DatetimeIndex,
    K: int,
) -> np.ndarray:
    """
    For each low-frequency date, extract K high-frequency lags.

    Returns array of shape (len(lf_dates), K).
    hf_series[i, 0] = most recent HF obs on or before lf_dates[i].
    hf_series[i, K-1] = K-1 steps back.

    Missing values filled with NaN.
    """
    hf = hf_series.dropna().sort_index()
    result = np.full((len(lf_dates), K), np.nan)

    for i, dt in enumerate(lf_dates):
        # Find last HF observation on or before dt
        mask = hf.index <= dt
        if mask.sum() < K:
            continue
        recent = hf[mask].iloc[-K:]
        # Reverse so [0] = most recent
        result[i] = recent.values[::-1]

    return result


def build_midas_feature(
    hf_series: pd.Series,
    lf_dates: pd.DatetimeIndex,
    K: int,
    theta1: float = 1.0,
    theta2: float = 5.0,
) -> pd.Series:
    """
    Build a single MIDAS feature: weighted average of K HF lags.

    Parameters
    ----------
    hf_series : daily/weekly Series
    lf_dates : monthly DatetimeIndex of target variable
    K : number of HF lags per LF period
    theta1, theta2 : beta polynomial parameters

    Returns
    -------
    pd.Series indexed by lf_dates.
    """
    hf_lags = align_hf_to_lf(hf_series, lf_dates, K)  # (T, K)
    w = beta_weights(K, theta1, theta2)  # (K,)
    weighted = (hf_lags * w).sum(axis=1)  # (T,)
    # NaN if any lag was missing
    weighted[np.isnan(hf_lags).any(axis=1)] = np.nan
    return pd.Series(weighted, index=lf_dates, name=hf_series.name)


def build_midas_features_multi(
    hf_dict: dict[str, pd.Series],
    lf_dates: pd.DatetimeIndex,
    K: int,
    theta_init: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build MIDAS features for multiple HF regressors.

    Returns:
    - DataFrame of weighted features (columns = regressor names)
    - 3D array of raw HF lags: shape (T, n_regressors, K)
    """
    if theta_init is None:
        theta_init = {}

    features = {}
    all_lags = []

    for name, series in hf_dict.items():
        t1 = theta_init.get(name, (1.0, 5.0))[0]
        t2 = theta_init.get(name, (1.0, 5.0))[1]
        features[name] = build_midas_feature(series, lf_dates, K, t1, t2)
        all_lags.append(align_hf_to_lf(series, lf_dates, K))

    X_midas = pd.DataFrame(features, index=lf_dates)
    raw_lags = np.stack(all_lags, axis=1) if all_lags else np.empty((len(lf_dates), 0, K))
    return X_midas, raw_lags


# =====================================================================
# XGBoost feature construction
# =====================================================================

def build_xgb_features(
    hf_series: pd.Series,
    lf_dates: pd.DatetimeIndex,
    K: int,
    name: str | None = None,
) -> pd.DataFrame:
    """
    Compute rolling statistics of HF series over K-day window for each LF date.

    Returns DataFrame with columns:
    {name}_mean, {name}_std, {name}_last, {name}_first,
    {name}_max, {name}_min, {name}_change, {name}_trend
    """
    if name is None:
        name = hf_series.name or "hf"

    hf_lags = align_hf_to_lf(hf_series, lf_dates, K)  # (T, K)

    feats = pd.DataFrame(index=lf_dates)
    feats[f"{name}_mean"] = np.nanmean(hf_lags, axis=1)
    feats[f"{name}_std"] = np.nanstd(hf_lags, axis=1)
    feats[f"{name}_last"] = hf_lags[:, 0]  # most recent
    feats[f"{name}_first"] = hf_lags[:, -1]  # most distant
    feats[f"{name}_max"] = np.nanmax(hf_lags, axis=1)
    feats[f"{name}_min"] = np.nanmin(hf_lags, axis=1)
    feats[f"{name}_change"] = hf_lags[:, 0] - hf_lags[:, -1]

    # OLS trend slope
    k = np.arange(K, dtype=float)
    k_demean = k - k.mean()
    slopes = np.full(len(lf_dates), np.nan)
    for i in range(len(lf_dates)):
        row = hf_lags[i]
        if not np.isnan(row).any():
            slopes[i] = np.sum(k_demean * (row - row.mean())) / (np.sum(k_demean**2) + 1e-10)
    feats[f"{name}_trend"] = slopes

    return feats


def build_xgb_features_multi(
    hf_dict: dict[str, pd.Series],
    lf_dates: pd.DatetimeIndex,
    K: int,
) -> pd.DataFrame:
    """Build XGB features for multiple HF regressors. Returns wide DataFrame."""
    dfs = []
    for name, series in hf_dict.items():
        dfs.append(build_xgb_features(series, lf_dates, K, name=name))
    return pd.concat(dfs, axis=1) if dfs else pd.DataFrame(index=lf_dates)


def add_same_freq_lags(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 2, 3],
) -> pd.DataFrame:
    """Add lagged values of same-frequency regressors."""
    result = df.copy()
    for col in columns:
        if col in df.columns:
            for lag in lags:
                result[f"{col}_lag{lag}"] = df[col].shift(lag)
    return result


# =====================================================================
# Target Variable Registry
# =====================================================================

# Frequency mapping for MIDAS K values
K_MAP = {
    ("D", "M"): 22,   # daily -> monthly: ~22 business days
    ("D", "W"): 5,    # daily -> weekly
    ("W", "M"): 4,    # weekly -> monthly
    ("Q", "M"): 1,    # quarterly -> monthly: use step function
}

# Beat/miss thresholds per target
THRESHOLDS = {
    "US_NFP": 25.0,           # thousands
    "US_UNRATE": 0.1,         # percentage points
    "US_AHE_MOM": 0.1,        # percentage points
    "US_UMCSENT": 2.0,        # index points
    "US_MICH_INF": 0.2,       # percentage points
    "US_CLAIMS": 10.0,        # thousands
    "UK_CONSTR_PMI": 1.0,     # index points
    "EZ_RETAIL_MOM": 0.3,     # percentage points
    "DE_FACTORY_MOM": 1.0,    # percentage points
    "DE_IP_MOM": 0.5,         # percentage points
    "MX_CPI_YOY": 0.1,       # percentage points
    "MX_BANXICO": 0.25,       # percentage points
    "BR_IP_MOM": 0.5,         # percentage points
    "BR_IPCA_YOY": 0.1,      # percentage points
    "BR_TRADE": 1.0,          # USD billions
    "CA_EMPLOYMENT": 10.0,    # thousands
    "CA_IVEY_PMI": 1.0,       # index points
}

# Target registry: defines regressors for each target
TARGETS = {
    "US_NFP": {
        "target_series": "NFP",
        "transform": "diff",
        "freq": "M",
        "hf_regressors": {
            "FED_RATE": {"source": "fred", "K": 22, "hf_freq": "D"},
            "SP500": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
            "US_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "US_Slope": {"source": "ns", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["CLAIMS", "ISM_EMP", "UNRATE"],
        "same_freq_lags": [1, 2],
    },
    "US_UNRATE": {
        "target_series": "UNRATE",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "FED_RATE": {"source": "fred", "K": 22, "hf_freq": "D"},
            "US_Slope": {"source": "ns", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["NFP", "PARTICIPATION", "CLAIMS"],
        "same_freq_lags": [1, 2],
    },
    "US_AHE_MOM": {
        "target_series": "AHE",
        "transform": "pct_change",
        "freq": "M",
        "hf_regressors": {
            "US_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["UNRATE", "CPI"],
        "same_freq_lags": [1, 2],
    },
    "US_UMCSENT": {
        "target_series": "UMCSENT",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "SP500": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
            "GAS_PRICE": {"source": "fred", "K": 4, "hf_freq": "W"},
        },
        "same_freq_regressors": ["UNRATE", "CPI"],
        "same_freq_lags": [1],
    },
    "US_MICH_INF": {
        "target_series": "MICH_INF_EXP",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
            "FED_RATE": {"source": "fred", "K": 22, "hf_freq": "D"},
            "GAS_PRICE": {"source": "fred", "K": 4, "hf_freq": "W"},
        },
        "same_freq_regressors": ["CPI"],
        "same_freq_lags": [1, 2],
    },
    "US_CLAIMS": {
        "target_series": "CLAIMS",
        "transform": None,
        "freq": "W",
        "hf_regressors": {
            "FED_RATE": {"source": "fred", "K": 5, "hf_freq": "D"},
            "SP500": {"source": "fred", "K": 5, "hf_freq": "D"},
        },
        "same_freq_regressors": ["CLAIMS"],
        "same_freq_lags": [1, 2, 3, 4],
    },
    "UK_CONSTR_PMI": {
        "target_series": "UK_CONSTR_PMI",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "UK_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "UK_Slope": {"source": "ns", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["UK_BOE_RATE"],
        "same_freq_lags": [1, 2],
    },
    "EZ_RETAIL_MOM": {
        "target_series": "EZ_RETAIL_MOM",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "EU_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "EURUSD": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["EZ_HICP", "EZ_UNEMP", "ECB_RATE"],
        "same_freq_lags": [1],
    },
    "DE_FACTORY_MOM": {
        "target_series": "DE_FACTORY_MOM",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "EURUSD": {"source": "fred", "K": 22, "hf_freq": "D"},
            "EU_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["DE_IP"],
        "same_freq_lags": [1, 2],
    },
    "DE_IP_MOM": {
        "target_series": "DE_IP",
        "transform": "pct_change",
        "freq": "M",
        "hf_regressors": {
            "EURUSD": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["DE_IP"],
        "same_freq_lags": [1, 2, 3],
    },
    "MX_CPI_YOY": {
        "target_series": "MX_CPI_YOY",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "USDMXN": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["MX_CPI_YOY"],
        "same_freq_lags": [1, 2, 3],
    },
    "BR_IP_MOM": {
        "target_series": "BR_IP",
        "transform": "pct_change",
        "freq": "M",
        "hf_regressors": {
            "USDBRL": {"source": "fred", "K": 22, "hf_freq": "D"},
            "BR_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["BR_SELIC"],
        "same_freq_lags": [1, 2],
    },
    "BR_IPCA_YOY": {
        "target_series": "BR_IPCA",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "USDBRL": {"source": "fred", "K": 22, "hf_freq": "D"},
            "BR_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["BR_IPCA", "BR_SELIC"],
        "same_freq_lags": [1, 2, 3],
    },
    "BR_TRADE": {
        "target_series": "BR_TRADE",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "USDBRL": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["BR_TRADE"],
        "same_freq_lags": [1, 2],
    },
    "CA_EMPLOYMENT": {
        "target_series": "CA_EMPLOYMENT",
        "transform": "diff",
        "freq": "M",
        "hf_regressors": {
            "CADUSD": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
            "CA_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["CA_BOC_RATE", "CA_UNEMP"],
        "same_freq_lags": [1, 2],
    },
    "CA_IVEY_PMI": {
        "target_series": "CA_IVEY_PMI",
        "transform": None,
        "freq": "M",
        "hf_regressors": {
            "CADUSD": {"source": "fred", "K": 22, "hf_freq": "D"},
            "OIL_WTI": {"source": "fred", "K": 22, "hf_freq": "D"},
            "CA_Level": {"source": "ns", "K": 22, "hf_freq": "D"},
        },
        "same_freq_regressors": ["ISM_EMP"],
        "same_freq_lags": [1],
    },
}


# =====================================================================
# Master feature matrix builder
# =====================================================================

def build_feature_matrix(
    target_key: str,
    macro_data: dict,
    mode: str = "midas",
    theta_init: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build (y, X) for a given target variable.

    Parameters
    ----------
    target_key : str, e.g. "US_NFP"
    macro_data : dict from fetch_all_macro()
    mode : "midas" or "xgb"
    theta_init : optional dict of {regressor_name: (theta1, theta2)} for MIDAS

    Returns
    -------
    y : pd.Series (target variable, transformed)
    X : pd.DataFrame (features, aligned to y's dates)
    """
    config = TARGETS[target_key]
    fred = macro_data["fred"]
    ecb = macro_data.get("ecb", pd.DataFrame())
    ipea = macro_data.get("ipea", pd.DataFrame())
    ns = macro_data["ns_factors"]

    # --- Build target series ---
    target_name = config["target_series"]

    # Find target in available data
    all_monthly = pd.concat([fred, ecb, ipea], axis=1)
    if target_name not in all_monthly.columns:
        raise ValueError(f"Target series '{target_name}' not found in macro data")

    y_raw = all_monthly[target_name].dropna()

    # Apply transform
    transform = config.get("transform")
    if transform == "diff":
        y = y_raw.diff().dropna()
    elif transform == "pct_change":
        y = y_raw.pct_change().dropna() * 100
    elif transform == "yoy":
        y = y_raw.pct_change(12).dropna() * 100
    else:
        y = y_raw.dropna()

    # Resample to target frequency if needed
    if config["freq"] == "M":
        y = y.resample("ME").last().dropna()
    elif config["freq"] == "W":
        y = y.resample("W-FRI").last().dropna()

    lf_dates = y.index

    # --- Build HF features ---
    hf_dict = {}
    for reg_name, reg_config in config.get("hf_regressors", {}).items():
        K = reg_config["K"]
        source = reg_config["source"]

        if source == "fred":
            if reg_name in fred.columns:
                hf_dict[reg_name] = (fred[reg_name].dropna(), K)
        elif source == "ns":
            if reg_name in ns.columns:
                hf_dict[reg_name] = (ns[reg_name].dropna(), K)

    # Build features based on mode
    feature_dfs = []

    for reg_name, (series, K) in hf_dict.items():
        if mode == "midas":
            t1 = theta_init.get(reg_name, (1.0, 5.0))[0] if theta_init else 1.0
            t2 = theta_init.get(reg_name, (1.0, 5.0))[1] if theta_init else 5.0
            feat = build_midas_feature(series, lf_dates, K, t1, t2)
            feat.name = f"midas_{reg_name}"
            feature_dfs.append(feat.to_frame())
        elif mode == "xgb":
            feat = build_xgb_features(series, lf_dates, K, name=reg_name)
            feature_dfs.append(feat)

    # --- Add same-frequency regressors ---
    same_freq_cols = config.get("same_freq_regressors", [])
    same_freq_lags = config.get("same_freq_lags", [1])

    for col in same_freq_cols:
        if col in all_monthly.columns:
            src = all_monthly[col].dropna().sort_index()
            aligned = src.reindex(lf_dates, method="ffill")
            for lag in same_freq_lags:
                lag_name = f"{col}_lag{lag}"
                feature_dfs.append(aligned.shift(lag).rename(lag_name).to_frame())

    # Combine all features
    if feature_dfs:
        X = pd.concat(feature_dfs, axis=1)
    else:
        X = pd.DataFrame(index=lf_dates)

    # Align and drop NaN
    combined = pd.concat([y.rename("target"), X], axis=1).dropna()
    y_clean = combined["target"]
    X_clean = combined.drop(columns=["target"])

    return y_clean, X_clean
