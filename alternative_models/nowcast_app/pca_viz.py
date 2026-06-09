"""
PCA analysis with full eigenvalue/eigenvector visualisation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


def run_pca(
    X: pd.DataFrame,
    n_components: int,
    train_end: str,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA on the training subsample and transform the full X.

    Returns
    -------
    pca_obj   : fitted sklearn PCA object
    scores_df : DataFrame (T, n_components) — PC scores for all observations
    loadings_df : DataFrame (n_features, n_components) — eigenvector loadings
    """
    X_clean = X.dropna()
    train_mask = X_clean.index <= pd.Timestamp(train_end)
    X_train = X_clean[train_mask]
    if len(X_train) < max(5, n_components):
        X_train = X_clean  # fallback

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    scores = pca.transform(X_clean)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=X_clean.index, columns=pc_cols)

    # Loadings: eigenvectors (components_) shape (n_comp, n_features) → transpose
    loadings = pca.components_.T  # (n_features, n_components)
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=pc_cols)

    return pca, scores_df, loadings_df


def make_eigenvalue_table(pca: PCA) -> pd.DataFrame:
    """Return eigenvalue table with variance explained and cumulative %."""
    evs = pca.explained_variance_
    evr = pca.explained_variance_ratio_ * 100
    rows = []
    cum = 0.0
    for i, (ev, evr_i) in enumerate(zip(evs, evr)):
        cum += evr_i
        rows.append({
            "PC": f"PC{i+1}",
            "Eigenvalue": round(float(ev), 4),
            "% Variance": round(float(evr_i), 2),
            "Cumulative %": round(float(cum), 2),
        })
    return pd.DataFrame(rows)


def make_scree_plot(pca: PCA) -> go.Figure:
    """Scree plot: bar chart of eigenvalues + cumulative explained variance line."""
    n = len(pca.explained_variance_)
    pcs = [f"PC{i+1}" for i in range(n)]
    evr = pca.explained_variance_ratio_ * 100
    cum_evr = np.cumsum(evr)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=pcs, y=pca.explained_variance_,
        name="Eigenvalue", marker_color="#3498db",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pcs, y=cum_evr,
        name="Cumulative Variance %", mode="lines+markers",
        marker_color="#e74c3c", line_width=2,
    ), secondary_y=True)

    fig.add_hline(y=80, line_dash="dot", line_color="gray",
                  annotation_text="80% threshold",
                  secondary_y=True)

    fig.update_layout(
        title="Scree Plot — Eigenvalues & Explained Variance",
        xaxis_title="Principal Component",
        legend=dict(x=0.6, y=0.1),
        plot_bgcolor="white",
        height=380,
    )
    fig.update_yaxes(title_text="Eigenvalue", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True,
                     range=[0, 105])
    return fig


def make_loadings_heatmap(loadings_df: pd.DataFrame) -> go.Figure:
    """Heatmap of PCA loadings: rows = features, cols = PCs."""
    fig = px.imshow(
        loadings_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="PCA Loadings — Eigenvector Components",
        labels={"x": "Principal Component", "y": "Feature", "color": "Loading"},
    )
    fig.update_layout(
        height=max(350, 22 * len(loadings_df)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(tickangle=0)
    return fig


def make_biplot(
    scores_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    pc_x: int,
    pc_y: int,
    color_series: pd.Series | None = None,
) -> go.Figure:
    """
    Biplot: scatter of PC scores + loading vectors for each original feature.

    pc_x, pc_y are 1-indexed PC numbers.
    """
    cx = f"PC{pc_x}"
    cy = f"PC{pc_y}"

    scores_aligned = scores_df[[cx, cy]].dropna()

    # Color
    if color_series is not None:
        c_aligned = color_series.reindex(scores_aligned.index)
        colors = c_aligned.values
        colorscale = "RdYlBu_r"
        marker_kw = dict(color=colors, colorscale=colorscale, showscale=True,
                         colorbar=dict(title=color_series.name or ""))
    else:
        marker_kw = dict(color="#3498db", opacity=0.6)

    hover_text = [str(d.date()) for d in scores_aligned.index]

    fig = go.Figure()

    # Observation scatter
    fig.add_trace(go.Scatter(
        x=scores_aligned[cx],
        y=scores_aligned[cy],
        mode="markers",
        marker=dict(size=6, **marker_kw),
        text=hover_text,
        hovertemplate="%{text}<br>" + cx + "=%{x:.3f}<br>" + cy + "=%{y:.3f}<extra></extra>",
        name="Observations",
    ))

    # Loading arrows (scaled to fit score range)
    scale = max(
        scores_aligned[cx].abs().max(),
        scores_aligned[cy].abs().max(),
    ) * 0.7

    load_x = loadings_df[cx] if cx in loadings_df.columns else None
    load_y = loadings_df[cy] if cy in loadings_df.columns else None

    if load_x is not None and load_y is not None:
        for feat in loadings_df.index:
            lx = float(load_x[feat]) * scale
            ly = float(load_y[feat]) * scale
            fig.add_annotation(
                ax=0, ay=0, x=lx, y=ly,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowwidth=1.5,
                arrowcolor="#e74c3c", opacity=0.75,
            )
            fig.add_trace(go.Scatter(
                x=[lx * 1.08], y=[ly * 1.08],
                mode="text",
                text=[feat],
                textfont=dict(size=9, color="#c0392b"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=f"Biplot — {cx} vs {cy}",
        xaxis_title=cx,
        yaxis_title=cy,
        height=500,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="lightgray")
    fig.add_vline(x=0, line_dash="dot", line_color="lightgray")
    return fig
