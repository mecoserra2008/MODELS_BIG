"""
Nelson-Siegel yield curve decomposition into Level, Slope, and Curvature factors.

Reference: Diebold, F.X. and Li, C. (2006), "Forecasting the term structure of
government bond yields", Journal of Econometrics, 130(2), 337-364.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize


def ns_factor_loadings(
    maturities: np.ndarray,
    lam: float = 0.0609,
) -> np.ndarray:
    """
    Compute the (N x 3) Nelson-Siegel factor loading matrix.

    Parameters
    ----------
    maturities : array of shape (N,)
        Maturities in years.
    lam : float
        Decay parameter (lambda). Default 0.0609 per Diebold-Li.

    Returns
    -------
    X : array of shape (N, 3)
        Column 0: Level loading (all ones)
        Column 1: Slope loading = (1 - exp(-lam*tau)) / (lam*tau)
        Column 2: Curvature loading = slope_loading - exp(-lam*tau)
    """
    tau = np.asarray(maturities, dtype=float)
    lt = lam * tau

    # Avoid division by zero for tau=0
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_loading = np.where(lt > 1e-10, (1 - np.exp(-lt)) / lt, 1.0)
        curvature_loading = slope_loading - np.exp(-lt)

    X = np.column_stack([np.ones_like(tau), slope_loading, curvature_loading])
    return X


def fit_ns_ols(
    yields: pd.DataFrame,
    maturities: np.ndarray,
    lam: float = 0.0609,
) -> pd.DataFrame:
    """
    Fit Nelson-Siegel by OLS for each date (row), with fixed lambda.

    Since lambda is fixed, the model is linear in (beta1, beta2, beta3)
    and we solve via least squares for each date.

    Parameters
    ----------
    yields : DataFrame of shape (T, N)
        T dates, N maturities. Values in percent (or consistent units).
    maturities : array of shape (N,)
        Maturity in years for each column.
    lam : float
        Fixed decay parameter.

    Returns
    -------
    factors : DataFrame with columns ['Level', 'Slope', 'Curvature']
        Same DatetimeIndex as input.
    """
    maturities = np.asarray(maturities, dtype=float)
    X = ns_factor_loadings(maturities, lam)

    # Pseudo-inverse (constant across dates since lambda is fixed)
    # betas = (X'X)^{-1} X' y  for each row y
    pinv = np.linalg.pinv(X)  # shape (3, N)

    # Convert yields to numpy, handle NaN by row
    Y = yields.values  # (T, N)

    betas = np.full((Y.shape[0], 3), np.nan)
    for t in range(Y.shape[0]):
        row = Y[t]
        valid = ~np.isnan(row)
        if valid.sum() >= 3:  # need at least 3 obs to fit 3 params
            X_valid = X[valid]
            y_valid = row[valid]
            b, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
            betas[t] = b

    factors = pd.DataFrame(
        betas,
        index=yields.index,
        columns=["Level", "Slope", "Curvature"],
    )
    return factors


def fit_ns_ols_vectorized(
    yields: pd.DataFrame,
    maturities: np.ndarray,
    lam: float = 0.0609,
) -> pd.DataFrame:
    """
    Vectorized NS OLS fit — faster but drops rows with any NaN.

    Use this when data is complete (no missing values).
    """
    maturities = np.asarray(maturities, dtype=float)
    X = ns_factor_loadings(maturities, lam)
    pinv = np.linalg.pinv(X)  # (3, N)

    Y = yields.dropna().values  # (T_clean, N)
    betas = (pinv @ Y.T).T  # (T_clean, 3)

    factors = pd.DataFrame(
        betas,
        index=yields.dropna().index,
        columns=["Level", "Slope", "Curvature"],
    )
    return factors


def fit_ns_nonlinear(
    yields_row: np.ndarray,
    maturities: np.ndarray,
    lam_bounds: tuple = (0.01, 0.5),
) -> dict:
    """
    Fit Nelson-Siegel with lambda estimated via optimization (single date).

    Parameters
    ----------
    yields_row : array of shape (N,)
        Yields for one date.
    maturities : array of shape (N,)
        Maturities in years.
    lam_bounds : tuple
        Bounds for lambda search.

    Returns
    -------
    dict with keys: 'Level', 'Slope', 'Curvature', 'lambda', 'rmse'
    """
    maturities = np.asarray(maturities, dtype=float)
    yields_row = np.asarray(yields_row, dtype=float)

    def objective(lam):
        X = ns_factor_loadings(maturities, lam)
        betas, residuals, _, _ = np.linalg.lstsq(X, yields_row, rcond=None)
        fitted = X @ betas
        return np.sqrt(np.mean((yields_row - fitted) ** 2))

    result = minimize_scalar(objective, bounds=lam_bounds, method="bounded")
    best_lam = result.x

    X = ns_factor_loadings(maturities, best_lam)
    betas, _, _, _ = np.linalg.lstsq(X, yields_row, rcond=None)
    fitted = X @ betas
    rmse = np.sqrt(np.mean((yields_row - fitted) ** 2))

    return {
        "Level": betas[0],
        "Slope": betas[1],
        "Curvature": betas[2],
        "lambda": best_lam,
        "rmse": rmse,
    }


def compute_fitted_yields(
    factors: pd.DataFrame,
    maturities: np.ndarray,
    lam: float = 0.0609,
) -> pd.DataFrame:
    """
    Reconstruct fitted yields from NS factors.

    Parameters
    ----------
    factors : DataFrame with columns ['Level', 'Slope', 'Curvature']
    maturities : array of maturities in years
    lam : decay parameter

    Returns
    -------
    DataFrame of fitted yields (same index as factors, one column per maturity)
    """
    X = ns_factor_loadings(np.asarray(maturities), lam)
    betas = factors[["Level", "Slope", "Curvature"]].values  # (T, 3)
    fitted = betas @ X.T  # (T, N)

    cols = [f"{m}Y" for m in maturities]
    return pd.DataFrame(fitted, index=factors.index, columns=cols)


def compute_rmse(
    actual: pd.DataFrame,
    factors: pd.DataFrame,
    maturities: np.ndarray,
    lam: float = 0.0609,
) -> pd.Series:
    """
    Compute RMSE of NS fit for each date.

    Returns Series with same index as factors.
    """
    fitted_df = compute_fitted_yields(factors, maturities, lam)
    # Align indices
    common_idx = actual.index.intersection(fitted_df.index)
    actual_vals = actual.loc[common_idx].values
    fitted_vals = fitted_df.loc[common_idx].values

    rmse = np.sqrt(np.nanmean((actual_vals - fitted_vals) ** 2, axis=1))
    return pd.Series(rmse, index=common_idx, name="RMSE")


def select_lambda(
    yields: pd.DataFrame,
    maturities: np.ndarray,
    lambda_grid: list[float] = [0.04, 0.05, 0.0609, 0.07, 0.08, 0.10],
) -> dict:
    """
    Select optimal lambda by minimizing average RMSE across all dates.

    Returns dict with 'best_lambda', 'rmse_by_lambda'.
    """
    results = {}
    for lam in lambda_grid:
        factors = fit_ns_ols(yields, maturities, lam)
        rmse_series = compute_rmse(yields, factors, maturities, lam)
        results[lam] = rmse_series.mean()

    best_lam = min(results, key=results.get)
    return {"best_lambda": best_lam, "rmse_by_lambda": results}
