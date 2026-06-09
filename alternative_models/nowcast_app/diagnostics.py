"""
Model assumption diagnostics suite.

Tests run post-estimation on residuals (and optionally the feature matrix X):
  - Normality:          Jarque-Bera (statsmodels), Shapiro-Wilk (scipy)
  - Heteroskedasticity: Breusch-Pagan, White (limited to 6 features)
  - Autocorrelation:    Durbin-Watson, Ljung-Box, Breusch-Godfrey
  - Stability:          CUSUM of OLS residuals

Plots (Plotly): Q-Q, residual vs fitted, residual over time, CUSUM.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    acorr_ljungbox,
    acorr_breusch_godfrey,
)


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def _test_jarque_bera(residuals: np.ndarray) -> dict:
    try:
        stat, pval, skew, kurt = jarque_bera(residuals)
        return {"stat": float(stat), "pval": float(pval),
                "skewness": float(skew), "kurtosis": float(kurt),
                "reject_h0": pval < 0.05}
    except Exception as e:
        return {"error": str(e)}


def _test_shapiro_wilk(residuals: np.ndarray) -> dict:
    try:
        # Shapiro-Wilk limited to 5000 obs
        r = residuals[:5000]
        stat, pval = scipy_stats.shapiro(r)
        return {"stat": float(stat), "pval": float(pval), "reject_h0": pval < 0.05}
    except Exception as e:
        return {"error": str(e)}


def _test_breusch_pagan(residuals: np.ndarray, X_sm: np.ndarray) -> dict:
    try:
        lm, lm_pval, f, f_pval = het_breuschpagan(residuals, X_sm)
        return {"lm_stat": float(lm), "pval": float(lm_pval),
                "f_stat": float(f), "f_pval": float(f_pval),
                "reject_h0": lm_pval < 0.05}
    except Exception as e:
        return {"error": str(e)}


def _test_white(residuals: np.ndarray, X_sm: np.ndarray) -> dict:
    try:
        # White test with cross-products is memory-intensive; limit to 6 features
        X_lim = X_sm[:, :6] if X_sm.shape[1] > 6 else X_sm
        lm, lm_pval, f, f_pval = het_white(residuals, X_lim)
        note = "Limited to first 6 features to avoid rank deficiency." if X_sm.shape[1] > 6 else ""
        return {"lm_stat": float(lm), "pval": float(lm_pval),
                "f_stat": float(f), "f_pval": float(f_pval),
                "reject_h0": lm_pval < 0.05, "note": note}
    except Exception as e:
        return {"error": str(e)}


def _test_durbin_watson(residuals: np.ndarray) -> dict:
    try:
        dw = durbin_watson(residuals)
        if dw < 1.5:
            interp = "Positive autocorrelation detected (DW < 1.5)"
        elif dw > 2.5:
            interp = "Negative autocorrelation detected (DW > 2.5)"
        else:
            interp = "No strong autocorrelation (1.5 ≤ DW ≤ 2.5)"
        return {"stat": float(dw), "interpretation": interp}
    except Exception as e:
        return {"error": str(e)}


def _test_ljung_box(residuals: np.ndarray, lags: list = None) -> dict:
    if lags is None:
        lags = [4, 8, 12]
    try:
        res = acorr_ljungbox(residuals, lags=lags, return_df=True)
        return {"lags": lags,
                "stats": [float(v) for v in res["lb_stat"].values],
                "pvals": [float(v) for v in res["lb_pvalue"].values]}
    except Exception as e:
        return {"error": str(e)}


def _test_breusch_godfrey(residuals: np.ndarray, X_sm: np.ndarray, nlags: int = 4) -> dict:
    try:
        lm, lm_pval, f, f_pval = acorr_breusch_godfrey(
            sm.OLS(residuals, X_sm).fit(), nlags=nlags
        )
        return {"lm_stat": float(lm), "pval": float(lm_pval),
                "f_stat": float(f), "f_pval": float(f_pval),
                "reject_h0": lm_pval < 0.05}
    except Exception as e:
        return {"error": str(e)}


def _test_cusum(residuals: np.ndarray) -> dict:
    try:
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        n = len(residuals)
        # CUSUM of squared residuals
        cusum = np.cumsum(residuals - residuals.mean())
        cusum_std = np.std(residuals) * np.sqrt(n)
        cusum_norm = cusum / (cusum_std + 1e-10)
        # 5% critical boundaries: ±(0.948 + 2*i/n) scaled
        idx = np.arange(1, n + 1)
        boundary = 0.948 * (1 + 2 * idx / n)

        is_stable = bool(np.all(np.abs(cusum_norm) < boundary))
        return {
            "cusum_path": cusum_norm.tolist(),
            "boundary":   boundary.tolist(),
            "is_stable":  is_stable,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Master diagnostics runner
# ---------------------------------------------------------------------------

def run_all_diagnostics(fit_result: dict, X: pd.DataFrame) -> dict:
    """
    Run all diagnostic tests on model residuals.

    Parameters
    ----------
    fit_result : dict with keys 'residuals', 'y_index', 'fitted'
    X          : feature DataFrame aligned to fit_result['y_index']
    """
    residuals = np.array(fit_result["residuals"])
    y_index   = fit_result.get("y_index", None)

    # Align X to residual index
    if y_index is not None and isinstance(X, pd.DataFrame):
        X_aligned = X.reindex(y_index).dropna(how="all")
        # Drop rows where residuals also exist
        min_len = min(len(residuals), len(X_aligned))
        residuals = residuals[:min_len]
        X_vals = X_aligned.values[:min_len]
    else:
        X_vals = X.values[:len(residuals)]

    X_sm = sm.add_constant(X_vals, has_constant="add")

    return {
        "jarque_bera":     _test_jarque_bera(residuals),
        "shapiro_wilk":    _test_shapiro_wilk(residuals),
        "breusch_pagan":   _test_breusch_pagan(residuals, X_sm),
        "white_test":      _test_white(residuals, X_sm),
        "durbin_watson":   _test_durbin_watson(residuals),
        "ljung_box":       _test_ljung_box(residuals),
        "breusch_godfrey": _test_breusch_godfrey(residuals, X_sm),
        "cusum":           _test_cusum(residuals),
    }


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def make_qq_plot(residuals: np.ndarray) -> go.Figure:
    """Normal Q-Q plot of residuals."""
    res = np.sort(residuals)
    n = len(res)
    theoretical = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=res,
        mode="markers",
        marker=dict(color="#3498db", size=5, opacity=0.7),
        name="Quantiles",
    ))
    # 45-degree reference line
    mn = min(theoretical.min(), res.min())
    mx = max(theoretical.max(), res.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color="red", dash="dash"), name="Normal line",
    ))
    fig.update_layout(
        title="Normal Q-Q Plot",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=380, plot_bgcolor="white",
    )
    return fig


def make_residual_fitted_plot(fitted: np.ndarray, residuals: np.ndarray) -> go.Figure:
    """Residual vs fitted values plot."""
    fig = go.Figure(go.Scatter(
        x=fitted, y=residuals,
        mode="markers",
        marker=dict(color="#2ecc71", size=5, opacity=0.7),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residuals vs Fitted",
        xaxis_title="Fitted Values",
        yaxis_title="Residuals",
        height=380, plot_bgcolor="white",
    )
    return fig


def make_residual_time_plot(index, residuals: np.ndarray) -> go.Figure:
    """Residuals over time plot."""
    fig = go.Figure(go.Scatter(
        x=list(index), y=residuals,
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(color="#9b59b6", width=1),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residuals Over Time",
        xaxis_title="Date",
        yaxis_title="Residual",
        height=380, plot_bgcolor="white",
    )
    return fig


def make_cusum_plot(cusum_result: dict, index) -> go.Figure:
    """CUSUM stability plot with 5% critical boundaries."""
    path = cusum_result.get("cusum_path", [])
    boundary = cusum_result.get("boundary", [])
    n = len(path)
    x = list(index)[:n] if index is not None else list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=path,
        mode="lines", name="CUSUM", line=dict(color="#2980b9", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=boundary,
        mode="lines", name="5% boundary (+)",
        line=dict(color="red", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[-b for b in boundary],
        mode="lines", name="5% boundary (−)",
        line=dict(color="red", dash="dash"),
        showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    status = "Stable" if cusum_result.get("is_stable") else "Instability detected"
    fig.update_layout(
        title=f"CUSUM Stability Test — {status}",
        xaxis_title="Date",
        yaxis_title="Normalised CUSUM",
        height=380, plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
    )
    return fig
