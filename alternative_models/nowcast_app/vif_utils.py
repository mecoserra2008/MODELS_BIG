"""
VIF (Variance Inflation Factor) utilities for multicollinearity analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for each column of X.
    Returns DataFrame with columns [feature, vif] sorted descending.
    """
    X_vals = X.dropna().values
    if X_vals.shape[0] <= X_vals.shape[1]:
        return pd.DataFrame({"feature": list(X.columns), "vif": [np.nan] * len(X.columns)})

    vifs = []
    for i in range(X_vals.shape[1]):
        try:
            v = variance_inflation_factor(X_vals, i)
        except Exception:
            v = np.nan
        vifs.append(v)

    df = pd.DataFrame({"feature": list(X.columns), "vif": vifs})
    return df.sort_values("vif", ascending=False).reset_index(drop=True)


def iterative_vif_elimination(
    X: pd.DataFrame,
    threshold: float = 10.0,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Iteratively drop the highest-VIF feature until all VIF < threshold.
    Returns (kept_cols, dropped_list) where dropped_list is [(col, vif_at_drop), ...].
    """
    remaining = list(X.columns)
    dropped = []

    while True:
        X_sub = X[remaining].dropna()
        if X_sub.shape[0] <= X_sub.shape[1]:
            break

        vif_df = compute_vif(X_sub)
        max_row = vif_df.iloc[0]
        if max_row["vif"] < threshold or np.isnan(max_row["vif"]):
            break

        col_to_drop = max_row["feature"]
        dropped.append((col_to_drop, float(max_row["vif"])))
        remaining.remove(col_to_drop)

    return remaining, dropped


def make_vif_bar_chart(vif_df: pd.DataFrame, threshold: float = 10.0) -> go.Figure:
    """Horizontal bar chart of VIF values with threshold line."""
    df = vif_df.sort_values("vif", ascending=True)
    colors = ["#e74c3c" if v >= threshold else "#2ecc71" for v in df["vif"]]

    fig = go.Figure(go.Bar(
        x=df["vif"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" if not np.isnan(v) else "N/A" for v in df["vif"]],
        textposition="outside",
    ))
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Variance Inflation Factor per Feature",
        xaxis_title="VIF",
        yaxis_title="",
        height=max(300, 30 * len(df)),
        margin=dict(l=150, r=60, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


def make_correlation_heatmap(X: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of the feature correlation matrix."""
    corr = X.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix",
    )
    fig.update_layout(
        height=max(400, 25 * len(corr)),
        coloraxis_colorbar=dict(title="r"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(tickangle=45)
    return fig
