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

VAR_ORDER = [
    "US_Level", "US_Slope", "US_Curvature",
    "EU_Level", "EU_Slope", "EU_Curvature",
    "UK_Level", "UK_Slope", "UK_Curvature",
]


def prepare_var_data(
    uk_factors: pd.DataFrame,
    us_factors: pd.DataFrame,
    eu_factors: pd.DataFrame,
    freq: str = "W-FRI",
) -> pd.DataFrame:
    """
    Merge NS factors from 3 regions and resample to a common frequency.

    Parameters
    ----------
    uk_factors, us_factors, eu_factors : DataFrame
        Each with columns ['Level', 'Slope', 'Curvature'] and DatetimeIndex.
    freq : str
        Resample frequency. 'W-FRI' (weekly Friday) recommended.

    Returns
    -------
    DataFrame with 9 columns in Cholesky ordering, resampled, NaN-dropped.
    """
    # Prefix columns by region
    uk = uk_factors.add_prefix("UK_")
    us = us_factors.add_prefix("US_")
    eu = eu_factors.add_prefix("EU_")

    combined = pd.concat([us, eu, uk], axis=1)
    combined = combined.sort_index()

    # Resample to target frequency (last observation in each period)
    if freq is not None:
        combined = combined.resample(freq).last()

    # Forward-fill small gaps, then drop remaining NaN
    combined = combined.ffill(limit=2)
    combined = combined.dropna()

    # Reorder columns per Cholesky convention
    combined = combined[VAR_ORDER]
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
        Variables to test as effects. If None, tests all UK variables.
    signif : float
        Significance level.

    Returns
    -------
    DataFrame with columns: ['caused', 'causing', 'F_stat', 'p_value', 'significant']
    """
    if caused is None:
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
