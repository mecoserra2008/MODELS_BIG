"""
t-SNE map: phrases in semantic space, coloured by expected FGBL Bund reaction.
Pipeline: TF-IDF (max 200 features) -> PCA (30 components) -> t-SNE (2D) -> Plotly scatter.

Reuses the run_tsne pattern from alternative_models/nowcast_app/tsne_viz.py.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from lexicon import score_segment, to_barometer, TOPIC_MAP

REACTION_COLORS = {
    "lower": "#c0392b",
    "neutral_lower": "#e67e22",
    "neutral": "#7f8c8d",
    "neutral_higher": "#27ae60",
    "higher": "#2980b9",
    "sharply higher": "#1a237e",
}

CLUSTER_LABELS = {
    0: "Energy/Inflation",
    1: "Wage/Services",
    2: "Forward Guidance",
    3: "Growth/Uncertainty",
    4: "Rates/Credibility",
}


def _assign_reaction(score: float) -> str:
    bar = to_barometer(score)
    if bar >= 68:
        return "lower"
    elif bar >= 58:
        return "neutral_lower"
    elif bar >= 47:
        return "neutral"
    elif bar >= 37:
        return "neutral_higher"
    else:
        return "higher"


def build_tsne_frame(segments: list[dict]) -> pd.DataFrame:
    """Score every segment and prepare DataFrame for t-SNE."""
    rows = []
    for seg in segments:
        raw = score_segment(seg["text"])
        reaction = seg.get("bund_reaction") or _assign_reaction(raw)
        rows.append({
            "meeting_date": seg["meeting_date"],
            "text": seg["text"],
            "raw_score": raw,
            "barometer": to_barometer(raw),
            "bund_reaction": reaction,
            "speaker": seg.get("speaker", "Lagarde"),
        })
    return pd.DataFrame(rows)


def run_tsne_pipeline(df: pd.DataFrame, n_clusters: int = 5, seed: int = 42
                      ) -> tuple[pd.DataFrame, go.Figure]:
    """
    Full TF-IDF -> PCA -> t-SNE pipeline.
    Returns augmented DataFrame (with tsne_x, tsne_y, cluster) and Plotly figure.
    """
    texts = df["text"].tolist()
    n = len(texts)

    # --- TF-IDF ---
    vectoriser = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
    )
    X_tfidf = vectoriser.fit_transform(texts).toarray()  # (n, 200)

    # --- PCA pre-reduction ---
    n_pca = min(30, n - 1, X_tfidf.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed)
    X_pca = pca.fit_transform(X_tfidf)

    # --- t-SNE ---
    perplexity = max(5, min(30, n // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    X_2d = tsne.fit_transform(X_pca)

    # --- K-Means clusters ---
    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    clusters = km.fit_predict(X_pca)

    df = df.copy()
    df["tsne_x"] = X_2d[:, 0]
    df["tsne_y"] = X_2d[:, 1]
    df["cluster"] = clusters
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS).fillna("Other")

    fig = _build_tsne_figure(df)
    return df, fig


def _build_tsne_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for reaction, colour in REACTION_COLORS.items():
        mask = df["bund_reaction"] == reaction
        if not mask.any():
            continue
        sub = df[mask]
        label_name = {
            "lower": "FGBL Lower (Hawkish)",
            "neutral_lower": "FGBL Neutral/Lower",
            "neutral": "FGBL Neutral",
            "neutral_higher": "FGBL Neutral/Higher",
            "higher": "FGBL Higher (Dovish)",
            "sharply higher": "FGBL Sharply Higher",
        }.get(reaction, reaction)

        # Marker size = abs(barometer - 50) / 5, clamped
        sizes = ((sub["barometer"] - 50).abs() / 5 + 6).clip(6, 18)

        hover = (
            sub["text"].str[:100] + "..."
            + "<br>Score: " + sub["barometer"].round(1).astype(str)
            + "<br>Meeting: " + sub["meeting_date"]
        )

        fig.add_trace(go.Scatter(
            x=sub["tsne_x"], y=sub["tsne_y"],
            mode="markers+text",
            marker=dict(color=colour, size=sizes, opacity=0.8,
                        line=dict(width=0.5, color="white")),
            text=sub["text"].str.split().str[:3].str.join(" "),
            textposition="top center",
            textfont=dict(size=7, color="#555"),
            hovertext=hover,
            hoverinfo="text",
            name=label_name,
        ))

    # Cluster centroid labels
    for cl, label in CLUSTER_LABELS.items():
        mask = df["cluster"] == cl
        if not mask.any():
            continue
        cx = df.loc[mask, "tsne_x"].mean()
        cy = df.loc[mask, "tsne_y"].mean()
        fig.add_annotation(
            x=cx, y=cy, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="#FF8200"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#FF8200", borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        title=dict(
            text="t-SNE Semantic Map: Lagarde Phrases × FGBL Bund Reaction",
            font=dict(size=14, color="#3C3C3C")
        ),
        xaxis=dict(showticklabels=False, title="t-SNE Dim 1"),
        yaxis=dict(showticklabels=False, title="t-SNE Dim 2"),
        template="plotly_white",
        font=dict(family="Helvetica Neue, Arial", size=11),
        legend=dict(
            orientation="v", x=1.01, y=1,
            bordercolor="#ddd", borderwidth=1,
        ),
        margin=dict(l=40, r=180, t=60, b=40),
    )
    return fig


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster summary: mean barometer, dominant reaction, representative phrase."""
    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        dom_reaction = sub["bund_reaction"].mode().iloc[0] if len(sub) else "neutral"
        rep_phrase = sub.loc[sub["barometer"].sub(sub["barometer"].mean()).abs().idxmin(), "text"]
        rows.append({
            "Cluster": CLUSTER_LABELS.get(cl, f"Cluster {cl}"),
            "N Segments": len(sub),
            "Mean Barometer": round(sub["barometer"].mean(), 1),
            "Dominant Reaction": dom_reaction,
            "Representative Phrase": rep_phrase[:80] + "...",
        })
    return pd.DataFrame(rows)
