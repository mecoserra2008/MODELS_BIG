"""
VAR model estimation, impulse response functions, forecast error variance
decomposition, and Granger causality tests for yield curve spillover analysis.

Uses statsmodels VAR implementation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson


# ---------------------------------------------------------------------------
# Variable ordering (Cholesky: most exogenous first)
# ---------------------------------------------------------------------------

FACTOR_NAMES = ["Level", "Slope", "Curvature"]

# Legacy 3-country ordering (backward compat)
VAR_ORDER = [
    "US_Level", "US_Slope", "US_Curvature",
    "EU_Level", "EU_Slope", "EU_Curvature",
    "UK_Level", "UK_Slope", "UK_Curvature",
]


def build_var_order(countries: list[str], target: str) -> list[str]:
    """
    Build Cholesky ordering: non-target countries first (in given order),
    then target last (most endogenous).

    Parameters
    ----------
    countries : list of str
        All country codes, e.g. ["US", "EU", "CA", "BR", "UK"].
    target : str
        The country to place last.

    Returns
    -------
    list of str, e.g. ["US_Level", "US_Slope", ..., "UK_Curvature"]
    """
    ordered = [c for c in countries if c != target] + [target]
    return [f"{c}_{f}" for c in ordered for f in FACTOR_NAMES]


def prepare_var_data(
    factors_dict: dict[str, pd.DataFrame] | None = None,
    *legacy_args,
    target: str | None = None,
    countries: list[str] | None = None,
    freq: str = "W-FRI",
    # Legacy positional args for backward compat
    _uk_factors=None,
    _us_factors=None,
    _eu_factors=None,
) -> pd.DataFrame:
    """
    Merge NS factors from multiple regions and resample.

    Can be called in two ways:

    New API:
        prepare_var_data({"UK": uk_df, "US": us_df, ...}, target="UK")

    Legacy API (backward compat):
        prepare_var_data(uk_factors, us_factors, eu_factors, freq="W-FRI")

    Parameters
    ----------
    factors_dict : dict mapping country code -> DataFrame with
        columns ['Level', 'Slope', 'Curvature'].
    target : str
        Country code to place last in Cholesky ordering.
    countries : list of str or None
        Explicit ordering of non-target countries. If None, sorted alphabetically.
    freq : str
        Resample frequency.

    Returns
    -------
    DataFrame with N*3 columns in Cholesky ordering.
    """
    # Handle legacy 3-positional-arg call:
    # prepare_var_data(uk_factors, us_factors, eu_factors)
    if factors_dict is not None and isinstance(factors_dict, pd.DataFrame):
        # Called with positional DataFrames (old API)
        uk_f = factors_dict
        us_f = legacy_args[0] if len(legacy_args) > 0 else _us_factors
        eu_f = legacy_args[1] if len(legacy_args) > 1 else _eu_factors
        if len(legacy_args) > 2 and isinstance(legacy_args[2], str):
            freq = legacy_args[2]
        factors_dict = {"UK": uk_f, "US": us_f, "EU": eu_f}
        if target is None:
            target = "UK"

    if factors_dict is None:
        raise ValueError("Must provide factors_dict or positional DataFrames")

    if target is None:
        target = list(factors_dict.keys())[-1]

    if countries is None:
        countries = list(factors_dict.keys())

    var_order = build_var_order(countries, target)

    # Prefix and concatenate
    dfs = []
    for code in countries:
        if code in factors_dict:
            dfs.append(factors_dict[code].add_prefix(f"{code}_"))
    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()

    if freq is not None:
        combined = combined.resample(freq).last()

    combined = combined.ffill(limit=2)
    combined = combined.dropna()

    # Reorder per Cholesky convention
    available = [c for c in var_order if c in combined.columns]
    combined = combined[available]
    return combined


# ---------------------------------------------------------------------------
# Stationarity testing
# ---------------------------------------------------------------------------

def adf_tests(data: pd.DataFrame, maxlag: int | None = None) -> pd.DataFrame:
    """
    Run Augmented Dickey-Fuller test on each column.

    Returns DataFrame with columns: ['ADF_stat', 'p_value', 'lags_used', 'nobs',
    'critical_1%', 'critical_5%', 'critical_10%', 'stationary_5%'].
    """
    results = []
    for col in data.columns:
        series = data[col].dropna()
        adf_result = adfuller(series, maxlag=maxlag, autolag="AIC")
        stat, pval, lags, nobs, crit, icbest = adf_result
        results.append({
            "Variable": col,
            "ADF_stat": stat,
            "p_value": pval,
            "lags_used": lags,
            "nobs": nobs,
            "critical_1%": crit["1%"],
            "critical_5%": crit["5%"],
            "critical_10%": crit["10%"],
            "stationary_5%": pval < 0.05,
        })
    return pd.DataFrame(results).set_index("Variable")


# ---------------------------------------------------------------------------
# VAR estimation
# ---------------------------------------------------------------------------

def select_var_lag(data: pd.DataFrame, max_lags: int = 12) -> pd.DataFrame:
    """
    Compute information criteria for VAR lag selection.

    Returns DataFrame indexed by lag with columns [AIC, BIC, HQIC, FPE].
    """
    model = VAR(data)
    lag_order = model.select_order(maxlags=max_lags)
    return lag_order.summary().data  # Returns the summary; we'll also return raw


def select_var_lag_detailed(data: pd.DataFrame, max_lags: int = 12) -> dict:
    """
    Returns dict with 'summary' (the lag order results object) and
    'recommended' (dict of criterion -> optimal lag).
    """
    model = VAR(data)
    lag_order = model.select_order(maxlags=max_lags)
    recommended = {
        "AIC": lag_order.aic,
        "BIC": lag_order.bic,
        "HQIC": lag_order.hqic,
        "FPE": lag_order.fpe,
    }
    return {"summary": lag_order, "recommended": recommended}


def estimate_var(data: pd.DataFrame, lags: int):
    """
    Fit VAR(lags) model.

    Parameters
    ----------
    data : DataFrame with 9 columns (in VAR_ORDER).
    lags : int
        Number of lags.

    Returns
    -------
    VARResults object.
    """
    model = VAR(data)
    results = model.fit(maxlags=lags, ic=None)
    return results


# ---------------------------------------------------------------------------
# Impulse Response Functions
# ---------------------------------------------------------------------------

def compute_irfs(var_results, periods: int = 40, orth: bool = True):
    """
    Compute impulse response functions.

    Parameters
    ----------
    var_results : VARResults
    periods : int
        Horizon in periods (e.g., 40 weeks).
    orth : bool
        If True, use orthogonalized (Cholesky) IRFs.

    Returns
    -------
    IRAnalysis object with .irfs, .plot(), .plot_cum() methods.
    """
    irf = var_results.irf(periods=periods)
    return irf


def irf_to_dataframe(
    irf_obj,
    impulse: str,
    response: str,
    var_names: list[str] = None,
) -> pd.DataFrame:
    """
    Extract a specific impulse-response pair as a DataFrame.

    Returns DataFrame with columns ['irf', 'lower', 'upper'] indexed by period.
    """
    if var_names is None:
        var_names = list(irf_obj.model.names)

    imp_idx = var_names.index(impulse)
    resp_idx = var_names.index(response)

    irfs = irf_obj.irfs[:, resp_idx, imp_idx]

    df = pd.DataFrame({"irf": irfs}, index=range(len(irfs)))
    df.index.name = "Period"

    # Add confidence bands if available
    if hasattr(irf_obj, "ci") and irf_obj.ci is not None:
        lower = irf_obj.ci[:, resp_idx, imp_idx, 0]
        upper = irf_obj.ci[:, resp_idx, imp_idx, 1]
        df["lower"] = lower
        df["upper"] = upper

    return df


# ---------------------------------------------------------------------------
# Forecast Error Variance Decomposition
# ---------------------------------------------------------------------------

def compute_fevd(var_results, periods: int = 52):
    """
    Compute Forecast Error Variance Decomposition.

    Returns FEVD object. Access decomposition via .decomp attribute
    (array of shape (periods, n_vars, n_vars)).
    """
    fevd = var_results.fevd(periods=periods)
    return fevd


def fevd_summary(
    fevd_obj,
    response: str,
    horizons: list[int] = [1, 4, 12, 26, 52],
    var_names: list[str] = None,
) -> pd.DataFrame:
    """
    Extract FEVD for a specific response variable at given horizons.

    Returns DataFrame: rows = horizons, columns = contributing variables,
    values = % of variance explained.
    """
    if var_names is None:
        var_names = list(fevd_obj.names)

    resp_idx = var_names.index(response)
    decomp = fevd_obj.decomp  # shape: (n_equations, periods, n_equations)

    rows = []
    for h in horizons:
        if h <= decomp.shape[1]:
            row = decomp[resp_idx, h - 1, :] * 100  # convert to %
            rows.append(row)
        else:
            rows.append(np.full(len(var_names), np.nan))

    df = pd.DataFrame(rows, index=horizons, columns=var_names)
    df.index.name = "Horizon"
    return df


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------

def granger_causality_tests(
    var_results,
    causing: list[str],
    caused: list[str] | None = None,
    target: str | None = None,
    signif: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether 'causing' variables Granger-cause each of the 'caused' variables.

    Parameters
    ----------
    var_results : VARResults
    causing : list of str
        Variable names of potential causes.
    caused : list of str or None
        Variables to test as effects. If None, auto-generated from target.
    target : str or None
        Country code (e.g., "UK"). Used to build caused list if caused is None.
    signif : float
        Significance level.

    Returns
    -------
    DataFrame with columns: ['caused', 'causing', 'F_stat', 'p_value', 'significant']
    """
    if caused is None:
        if target is not None:
            caused = [f"{target}_{f}" for f in FACTOR_NAMES]
        else:
            caused = ["UK_Level", "UK_Slope", "UK_Curvature"]

    results = []
    for target in caused:
        try:
            test_result = var_results.test_causality(
                target, causing=causing, kind="f"
            )
            results.append({
                "caused": target,
                "causing": ", ".join(causing),
                "F_stat": test_result.test_statistic,
                "p_value": test_result.pvalue,
                "significant": test_result.pvalue < signif,
            })
        except Exception as e:
            results.append({
                "caused": target,
                "causing": ", ".join(causing),
                "F_stat": np.nan,
                "p_value": np.nan,
                "significant": False,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def var_diagnostics(var_results) -> dict:
    """
    Compute standard VAR diagnostics.

    Returns dict with:
    - 'durbin_watson': DW stats per equation
    - 'portmanteau': Portmanteau test (autocorrelation)
    - 'normality': Jarque-Bera test
    """
    # Durbin-Watson
    resids = var_results.resid
    dw_stats = {}
    for i, name in enumerate(var_results.names):
        dw_stats[name] = durbin_watson(resids.iloc[:, i])

    # Portmanteau test for residual autocorrelation
    try:
        port_test = var_results.test_whiteness(nlags=20, adjusted=True)
        portmanteau = {
            "statistic": port_test.test_statistic,
            "p_value": port_test.pvalue,
        }
    except Exception:
        portmanteau = {"statistic": np.nan, "p_value": np.nan}

    # Jarque-Bera normality test
    try:
        norm_test = var_results.test_normality()
        normality = {
            "statistic": norm_test.test_statistic,
            "p_value": norm_test.pvalue,
        }
    except Exception:
        normality = {"statistic": np.nan, "p_value": np.nan}

    return {
        "durbin_watson": dw_stats,
        "portmanteau": portmanteau,
        "normality": normality,
    }
