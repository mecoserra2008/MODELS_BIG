"""
Shared Plotly figure helpers used by multiple app tabs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def make_target_series_plot(y: pd.Series, target_key: str) -> go.Figure:
    """Line chart of the target series with rolling mean."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values,
        mode="lines", name=target_key,
        line=dict(color="#2980b9", width=1.5),
    ))
    if len(y) >= 12:
        roll = y.rolling(12).mean()
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values,
            mode="lines", name="12m rolling mean",
            line=dict(color="#e74c3c", width=2, dash="dot"),
        ))
    fig.update_layout(
        title=f"{target_key} — Historical Series",
        xaxis_title="Date",
        yaxis_title=target_key,
        height=350,
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    return fig


def make_feature_matrix_heatmap(X: pd.DataFrame, max_cols: int = 30) -> go.Figure:
    """Heatmap preview of the feature matrix (first max_cols columns)."""
    X_show = X.iloc[:, :max_cols]
    fig = px.imshow(
        X_show.T,
        color_continuous_scale="RdBu_r",
        zmin=-3, zmax=3,
        aspect="auto",
        title=f"Feature Matrix Preview (standardised, first {len(X_show.columns)} cols)",
        labels={"x": "Date", "y": "Feature", "color": "z-score"},
    )
    fig.update_layout(height=max(300, 18 * len(X_show.columns)), margin=dict(l=10, r=10))
    return fig


def make_coefficient_plot(coef_df: pd.DataFrame, method: str) -> go.Figure:
    """
    Horizontal bar chart of coefficients/importances.
    Shows confidence intervals if available (OLS / QuantReg).
    """
    coef_col = "importance" if "importance" in coef_df.columns else "coef"
    df = coef_df.copy().sort_values(coef_col, key=abs, ascending=True).tail(30)

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df[coef_col]]

    has_ci = ("ci_low" in df.columns and df["ci_low"].notna().any()
              and "ci_high" in df.columns)

    if has_ci:
        err_minus = (df[coef_col] - df["ci_low"]).abs().fillna(0).values
        err_plus  = (df["ci_high"] - df[coef_col]).abs().fillna(0).values
        error_x   = dict(type="data", symmetric=False,
                         array=err_plus, arrayminus=err_minus)
    else:
        error_x = None

    fig = go.Figure(go.Bar(
        x=df[coef_col].values,
        y=df["feature"].values,
        orientation="h",
        marker_color=colors,
        error_x=error_x,
        text=[f"{v:.3f}" for v in df[coef_col]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    label = "Importance" if coef_col == "importance" else "Coefficient"
    fig.update_layout(
        title=f"{method} — {label}s (top 30 by magnitude)",
        xaxis_title=label,
        yaxis_title="",
        height=max(350, 22 * len(df)),
        margin=dict(l=150, r=60, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


def make_fitted_vs_actual_plot(
    y_true: pd.Series,
    fitted: np.ndarray,
    method: str,
) -> go.Figure:
    """Actual vs fitted time-series chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true.values,
        mode="lines", name="Actual",
        line=dict(color="#2980b9", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=y_true.index, y=fitted,
        mode="lines", name="Fitted",
        line=dict(color="#e67e22", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title=f"{method} — Actual vs Fitted",
        xaxis_title="Date",
        yaxis_title="Value",
        height=380,
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    return fig


def make_almon_weight_profile(
    weights: np.ndarray,
    reg_name: str,
    K: int,
) -> go.Figure:
    """Bar chart of implied Almon PDL lag weights w_k."""
    k_labels = [f"t-{k}" for k in range(1, K + 1)]
    colors = ["#e74c3c" if w < 0 else "#3498db" for w in weights]
    fig = go.Figure(go.Bar(
        x=k_labels, y=weights,
        marker_color=colors,
        name="Lag weight",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Almon PDL Implied Lag Profile — {reg_name}",
        xaxis_title="Lag",
        yaxis_title="Weight β_k",
        height=320,
        plot_bgcolor="white",
    )
    return fig


def make_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Horizontal bar of feature importances (XGBoost / RandomForest)."""
    df = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#9b59b6",
    ))
    fig.update_layout(
        title=f"Feature Importance (top {top_n})",
        xaxis_title="Importance",
        height=max(300, 22 * len(df)),
        margin=dict(l=150, r=40, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


def make_probability_chart(probs: np.ndarray, index, labels: list) -> go.Figure:
    """Stacked area chart of OrderedLogit class probabilities over time."""
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    fig = go.Figure()
    for j, lbl in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=list(index), y=probs[:, j],
            mode="lines",
            name=lbl,
            line=dict(width=0),
            stackgroup="one",
            fillcolor=colors[j] if j < len(colors) else None,
            opacity=0.7,
        ))
    fig.update_layout(
        title="OrderedLogit Class Probabilities Over Time",
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig
