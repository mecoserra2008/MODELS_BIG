"""
t-SNE visualisation of the lagged feature matrix.

t-SNE reveals non-linear cluster structure that PCA misses.
Results are exploratory — topology depends heavily on perplexity and
random seed, so robustness checks (multiple runs) are recommended.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def run_tsne(
    X: pd.DataFrame,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run t-SNE on X (observations × features).

    Parameters
    ----------
    X            : DataFrame of scaled features
    perplexity   : t-SNE perplexity (typical range 5–50)
    n_iter       : number of optimisation iterations
    random_state : seed for reproducibility

    Returns
    -------
    embedding : np.ndarray (T, 2)
    """
    X_clean = X.dropna().values
    n = len(X_clean)

    # Clamp perplexity to a valid range (must be < n/3 and >= 5)
    perplexity = min(float(perplexity), max(5.0, (n - 1) / 3.0 - 1))

    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    # sklearn >= 1.4 uses max_iter; earlier versions used n_iter
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) >= Version("1.4"):
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter
    tsne = TSNE(**tsne_kwargs)
    embedding = tsne.fit_transform(X_clean)
    return embedding


def make_tsne_plot(
    embedding: np.ndarray,
    index: pd.DatetimeIndex,
    color_series: pd.Series | None = None,
    color_label: str = "",
) -> go.Figure:
    """
    Scatter plot of t-SNE embedding.

    Parameters
    ----------
    embedding    : (T, 2) array from run_tsne
    index        : DatetimeIndex aligned to embedding rows
    color_series : optional Series aligned to index for colouring
    color_label  : label for colour axis
    """
    n = len(embedding)
    hover_text = [str(d.date()) for d in index[:n]]
    x = embedding[:, 0]
    y = embedding[:, 1]

    if color_series is not None:
        c_aligned = color_series.reindex(index[:n])
        colors = c_aligned.values
        colorscale = "RdYlBu_r"
        marker = dict(
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=color_label),
            size=7,
            opacity=0.8,
        )
    else:
        marker = dict(
            color="#3498db",
            size=7,
            opacity=0.7,
        )

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=marker,
        text=hover_text,
        hovertemplate="%{text}<br>dim1=%{x:.2f}<br>dim2=%{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="t-SNE Embedding of Lagged Feature Matrix",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        height=500,
        plot_bgcolor="white",
        xaxis=dict(zeroline=False, showgrid=True, gridcolor="lightgray"),
        yaxis=dict(zeroline=False, showgrid=True, gridcolor="lightgray"),
    )
    return fig
