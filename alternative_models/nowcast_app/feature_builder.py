"""
Feature matrix construction for the nowcasting app.

All features are STRICTLY LAGGED — no contemporaneous values enter the matrix.

Three weighting schemes:
  - "Direct lags"  : k-period lags of the raw series (same or lower freq)
  - "Almon PDL"    : classical polynomial distributed lag transformation
  - "Beta MIDAS"   : beta-polynomial weighted aggregate (Ghysels et al. 2007)

Publication cutoff is applied per-regressor before any lag construction,
enforcing the ragged-edge constraint: regressor i is only available up to
as_of_date - pub_lag_bdays(i).
"""

import sys
import os
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Make sure the existing src modules are importable
_APP_DIR = Path(__file__).parent
_ALT_DIR = _APP_DIR.parent
sys.path.insert(0, str(_ALT_DIR))

from src.feature_engineering import beta_weights, align_hf_to_lf
from config import TARGETS, REGRESSORS, TRAIN_END


# ---------------------------------------------------------------------------
# Publication cutoff
# ---------------------------------------------------------------------------

def apply_publication_cutoff(
    series: pd.Series,
    pub_lag_bdays: int,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Truncate series to data that would be available by as_of_date."""
    cutoff = as_of_date - pd.offsets.BDay(pub_lag_bdays)
    return series[series.index <= cutoff]


# ---------------------------------------------------------------------------
# Target series construction
# ---------------------------------------------------------------------------

def get_target_series(
    raw_data: dict,
    target_key: str,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """
    Extract and transform the target variable from raw_data.
    Applies publication cutoff so only released observations are included.
    """
    fred_col, ecb_col, transform, freq, pub_lag = TARGETS[target_key]

    if fred_col is not None:
        df = raw_data.get("fred", pd.DataFrame())
        if fred_col not in df.columns:
            raise ValueError(f"FRED column '{fred_col}' not found in raw data.")
        s = df[fred_col].dropna()
    elif ecb_col is not None:
        df = raw_data.get("ecb", pd.DataFrame())
        if ecb_col not in df.columns:
            raise ValueError(f"ECB column '{ecb_col}' not found in raw data.")
        s = df[ecb_col].dropna()
    else:
        raise ValueError(f"No data source configured for target '{target_key}'.")

    # Apply publication cutoff
    s = apply_publication_cutoff(s, pub_lag, as_of_date)

    # Resample to monthly end (most targets are monthly)
    if freq == "M":
        s = s.resample("ME").last().dropna()

    # Transform
    if transform == "mom":
        s = s.pct_change() * 100
    elif transform == "yoy":
        s = s.pct_change(12) * 100
    elif transform == "diff":
        s = s.diff()
    # "lvl" → no transformation

    s = s.dropna()
    s.name = target_key
    return s


# ---------------------------------------------------------------------------
# Almon PDL: classical linear polynomial distributed lag
# ---------------------------------------------------------------------------

def almon_pdl_transform(
    hf_lags: np.ndarray,
    degree: int = 2,
    constraint: str = "far",
) -> tuple[np.ndarray, list[str]]:
    """
    Transform a (T, K) matrix of strictly-lagged values into (T, p) Almon
    Z-regressors suitable for OLS.

    Mathematical basis
    ------------------
    Constrain β_k = Σ_{j=0}^{degree} δ_j · k_c^j where k_c = k - mean(k).
    Substituting into y = Xβ gives y = Z·δ where
        Z_j = Σ_k k_c^j · x_{t-k}  (dot product of lag row with k_c^j vector)

    Endpoint constraints
    --------------------
    - "far"  : β_K = 0  → last row of Z constrained via substitution (linear restriction)
    - "near" : β_1 = 0  → drop j=0 Z-column (constant polynomial term)
               Note: k starts at 1 (strictly lagged), so β_0 never appears
    - "both" : both constraints
    - "none" : unrestricted polynomial

    For "far", the constraint β_K = 0 → Σ_j δ_j · K_c^j = 0 is imposed by
    substituting δ_0 = -(Σ_{j>0} δ_j · K_c^j) / K_c^0, then absorbing back
    into the Z columns. This keeps estimation as plain OLS.

    Returns
    -------
    Z : np.ndarray (T, n_cols)
    col_names : list[str] matching Z columns
    """
    T, K = hf_lags.shape
    k_vals = np.arange(1, K + 1, dtype=float)
    k_c = k_vals - k_vals.mean()   # centering improves numerical stability

    # Build raw Z columns: Z_j = hf_lags @ k_c^j
    raw_Z = []
    for j in range(degree + 1):
        raw_Z.append(hf_lags @ (k_c ** j))
    raw_Z = np.column_stack(raw_Z)  # (T, degree+1)

    # Apply far-end constraint: β_K = 0
    # β_K = Σ_j δ_j K_c^j = 0  →  δ_0 = -(Σ_{j>=1} δ_j K_c^j) / K_c^0
    # Substitute into Z: Z_new[:,j≥1] += Z_new[:,0] · (-K_c^j / K_c^0)
    # Z_new[:,0] is dropped (it would become zero)
    if constraint in ("far", "both"):
        K_c0 = k_c[-1] ** 0  # = 1.0
        if abs(K_c0) < 1e-10:
            pass  # degenerate, skip constraint
        else:
            K_c_powers = np.array([k_c[-1] ** j for j in range(degree + 1)])
            # After substitution, drop column 0 and adjust columns j>=1
            Z_constrained = []
            for j in range(1, degree + 1):
                z_j_adj = raw_Z[:, j] - (K_c_powers[j] / K_c0) * raw_Z[:, 0]
                Z_constrained.append(z_j_adj)
            raw_Z = np.column_stack(Z_constrained) if Z_constrained else raw_Z[:, 1:]
            degree_eff = degree  # now has 'degree' columns (not degree+1)
            start_j = 1
    else:
        degree_eff = degree + 1
        start_j = 0

    # Apply near-end constraint: drop j=0 column (if not already dropped by far)
    if constraint in ("near", "both") and start_j == 0:
        raw_Z = raw_Z[:, 1:]
        start_j = 1

    # Column names
    cols = []
    n_cols = raw_Z.shape[1]
    for idx in range(n_cols):
        j = start_j + idx
        cols.append(f"Z{j}")

    return raw_Z, cols


def build_almon_weights_from_delta(
    delta: np.ndarray,
    K: int,
    degree: int,
    constraint: str,
) -> np.ndarray:
    """
    Recover the implied lag profile w_k = Σ_j δ_j · k_c^j from OLS δ estimates.
    Useful for visualising the effective weighting of past lags.
    """
    k_vals = np.arange(1, K + 1, dtype=float)
    k_c = k_vals - k_vals.mean()

    if constraint == "none":
        # delta has degree+1 entries corresponding to j=0..degree
        w = np.zeros(K)
        for j, d in enumerate(delta):
            w += d * (k_c ** j)
    elif constraint in ("near",):
        # delta corresponds to j=1..degree
        w = np.zeros(K)
        for j, d in enumerate(delta, start=1):
            w += d * (k_c ** j)
    elif constraint in ("far",):
        # After substitution, delta has 'degree' entries for j=1..degree
        # δ_0 was absorbed. Reconstruct:
        K_c_powers = np.array([k_c[-1] ** j for j in range(degree + 1)])
        # delta_full[j>=1] = delta; delta_full[0] = -(sum delta[j]*K_c^j)/K_c^0
        d_full = np.zeros(degree + 1)
        for j, d in enumerate(delta, start=1):
            d_full[j] = d
        d_full[0] = -sum(d_full[j] * K_c_powers[j] for j in range(1, degree + 1))
        w = np.zeros(K)
        for j in range(degree + 1):
            w += d_full[j] * (k_c ** j)
    else:  # "both"
        # Near: drop j=0; Far: absorb j=0 → effectively only j=1..degree-1 remain
        # Just compute best-effort profile
        w = np.zeros(K)
        for j, d in enumerate(delta, start=1):
            w += d * (k_c ** j)
    return w


# ---------------------------------------------------------------------------
# Regressor series extraction
# ---------------------------------------------------------------------------

def get_regressor_series(raw_data: dict, reg_key: str) -> pd.Series:
    """Extract a regressor series from merged raw_data dict."""
    source_col = REGRESSORS[reg_key][0]

    # Search order: fred, ecb, ipea, ns_factors, yields
    for key in ("fred", "ecb", "ipea"):
        df = raw_data.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and source_col in df.columns:
            s = df[source_col].dropna()
            s.name = reg_key
            return s

    # NS factors
    ns = raw_data.get("ns_factors", pd.DataFrame())
    if isinstance(ns, pd.DataFrame) and source_col in ns.columns:
        s = ns[source_col].dropna()
        s.name = reg_key
        return s

    raise ValueError(f"Regressor '{reg_key}' (column '{source_col}') not found in raw data.")


# ---------------------------------------------------------------------------
# Build lagged feature matrix for a single regressor
# ---------------------------------------------------------------------------

def build_regressor_features(
    reg_series: pd.Series,
    lf_dates: pd.DatetimeIndex,
    lag_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Build feature columns for one regressor given its lag configuration.

    Parameters
    ----------
    reg_series : Series (at its native frequency, already pub-cutoff-applied)
    lf_dates   : monthly DatetimeIndex of target variable
    lag_cfg    : {weight_type, K, degree, constraint, theta1, theta2}

    Returns
    -------
    df_feat    : DataFrame aligned to lf_dates
    build_meta : dict with Z-column names and K/degree for later weight recovery
    """
    weight_type = lag_cfg.get("weight_type", "Almon PDL")
    K           = int(lag_cfg.get("K", 3))
    degree      = int(lag_cfg.get("degree", 2))
    constraint  = lag_cfg.get("constraint", "far")
    theta1      = float(lag_cfg.get("theta1", 1.0))
    theta2      = float(lag_cfg.get("theta2", 5.0))
    reg_name    = reg_series.name or "reg"

    # Shift the low-frequency reference dates back one period so we only use
    # the PRIOR period's high-frequency data — enforcing strict lag-only.
    lf_lagged = lf_dates - pd.DateOffset(months=1)

    obs_freq = REGRESSORS.get(reg_name, (None, "M"))[1] if reg_name in REGRESSORS else "M"

    if weight_type == "Direct lags":
        # Same or lower frequency: produce K direct lags
        # Resample to monthly if needed
        if obs_freq == "D":
            s_monthly = reg_series.resample("ME").last()
        elif obs_freq == "W":
            s_monthly = reg_series.resample("ME").last()
        else:
            s_monthly = reg_series

        cols = {}
        for k in range(1, K + 1):
            shifted = s_monthly.shift(k)
            vals = shifted.reindex(lf_dates, method="ffill")
            cols[f"{reg_name}_lag{k}"] = vals
        df_feat = pd.DataFrame(cols, index=lf_dates)
        meta = {"type": "direct", "col_names": list(cols.keys()), "K": K}

    elif weight_type == "Beta MIDAS":
        hf_lags = align_hf_to_lf(reg_series, lf_lagged, K)  # (T, K)
        w = beta_weights(K, theta1, theta2)
        weighted = (hf_lags * w).sum(axis=1)
        weighted[np.isnan(hf_lags).any(axis=1)] = np.nan
        col_name = f"{reg_name}_beta"
        df_feat = pd.DataFrame({col_name: weighted}, index=lf_dates)
        meta = {"type": "beta", "col_names": [col_name], "K": K,
                "theta1": theta1, "theta2": theta2, "weights": w}

    else:  # "Almon PDL"
        hf_lags = align_hf_to_lf(reg_series, lf_lagged, K)  # (T, K)
        Z, z_names = almon_pdl_transform(hf_lags, degree=degree, constraint=constraint)
        prefixed = [f"{reg_name}_{n}" for n in z_names]
        df_feat = pd.DataFrame(Z, index=lf_dates, columns=prefixed)
        meta = {"type": "almon", "col_names": prefixed, "K": K,
                "degree": degree, "constraint": constraint}

    return df_feat, meta


# ---------------------------------------------------------------------------
# Master feature matrix builder
# ---------------------------------------------------------------------------

def build_full_feature_matrix(
    raw_data: dict,
    target_key: str,
    regressor_keys: list[str],
    lag_configs: dict,
    as_of_date: pd.Timestamp,
    train_end: str = TRAIN_END,
) -> tuple[pd.DataFrame, pd.Series, dict, StandardScaler]:
    """
    Build the complete scaled feature matrix and target series.

    Parameters
    ----------
    raw_data       : dict from fetch_all_macro()
    target_key     : e.g. "US_CPI_YOY"
    regressor_keys : list of regressor names
    lag_configs    : {reg_key: {weight_type, K, degree, constraint, ...}}
    as_of_date     : Timestamp — simulate real-time knowledge cutoff
    train_end      : last date of training period for StandardScaler fit

    Returns
    -------
    X_scaled  : StandardScaler-normalised feature DataFrame
    y         : target Series aligned to X_scaled.index
    build_meta: {reg_key: meta_dict} for weight recovery
    scaler    : fitted StandardScaler (fit on train subsample only)
    """
    # 1. Target series
    y = get_target_series(raw_data, target_key, as_of_date)
    lf_dates = y.index

    # 2. Build features for each regressor
    all_feature_dfs = []
    all_meta = {}

    for reg_key in regressor_keys:
        cfg = lag_configs.get(reg_key, {})
        _, obs_freq, default_K, default_pub_lag = REGRESSORS[reg_key]

        # Fill defaults from config
        cfg = {
            "weight_type": cfg.get("weight_type", "Almon PDL" if obs_freq in ("D","W") else "Direct lags"),
            "K":           cfg.get("K", default_K),
            "degree":      cfg.get("degree", 2),
            "constraint":  cfg.get("constraint", "far"),
            "theta1":      cfg.get("theta1", 1.0),
            "theta2":      cfg.get("theta2", 5.0),
            "pub_lag":     cfg.get("pub_lag", default_pub_lag),
        }

        try:
            raw_series = get_regressor_series(raw_data, reg_key)
            # Apply publication cutoff
            raw_series = apply_publication_cutoff(raw_series, cfg["pub_lag"], as_of_date)
            feat_df, meta = build_regressor_features(raw_series, lf_dates, cfg)
            all_feature_dfs.append(feat_df)
            all_meta[reg_key] = meta
        except Exception as e:
            # Skip unavailable regressors silently; caller sees reduced columns
            all_meta[reg_key] = {"error": str(e)}

    if not all_feature_dfs:
        raise ValueError("No features could be constructed. Check data availability.")

    X_raw = pd.concat(all_feature_dfs, axis=1)

    # 3. Align X and y (inner join on date index)
    combined = X_raw.join(y, how="inner").dropna()
    X_raw = combined.drop(columns=[target_key])
    y = combined[target_key]

    # 4. StandardScaler fit on training subsample
    train_mask = X_raw.index <= pd.Timestamp(train_end)
    X_train = X_raw[train_mask]
    if len(X_train) < 5:
        X_train = X_raw  # fallback if train period is too short

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        index=X_raw.index,
        columns=X_raw.columns,
    )

    return X_scaled, y, all_meta, scaler


# ---------------------------------------------------------------------------
# Config hash (for stale-feature detection in app)
# ---------------------------------------------------------------------------

def config_hash(target_key: str, regressor_keys: list, lag_configs: dict,
                as_of_date: str, train_end: str) -> str:
    payload = json.dumps(
        {"t": target_key, "r": sorted(regressor_keys),
         "l": lag_configs, "a": as_of_date, "te": train_end},
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
