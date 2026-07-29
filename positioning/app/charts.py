"""Plotly figure factories for the step-by-step tabs.

Every figure: single y-axis (no dual-axis), recessive grid, hover, validated palette.
Categorical series are direct-labeled (relief for the sub-3:1 hues); diverging heatmaps
use the orange↔blue signed scale; magnitude uses the orange sequential ramp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from positioning.config import CATEGORY_LABELS, CONTRACTS, SCORING_CATEGORIES
from positioning.core.normalize import rolling_zscore
from . import theme

INSTRUMENT_ORDER = [c.key for c in CONTRACTS]


# --------------------------------------------------------------------------- #
# Step 6 — composite index with hysteresis bands + stance shading
# --------------------------------------------------------------------------- #
def composite_fig(res, mode: str, k_entry: float) -> go.Figure:
    p = theme.pal(mode)
    comp = res.composite.dropna()
    st_ = res.stance.reindex(comp.index).fillna(0)
    fig = go.Figure()
    # stance shading as clean vrect bands over contiguous runs (no NaN-fill artifacts)
    labelled = {-1: False, 1: False}
    for x0, x1, sgn in _runs(st_):
        color = p["bear"] if sgn < 0 else p["bull"]
        name = "short-duration stance" if sgn < 0 else "long-duration stance"
        fig.add_vrect(x0=x0, x1=x1, fillcolor=_rgba(color, 0.16), line_width=0,
                      layer="below")
        if not labelled[sgn]:  # one legend proxy per stance sign
            fig.add_trace(go.Scatter(x=[comp.index[0]], y=[None], mode="markers",
                          marker=dict(color=_rgba(color, 0.5), size=10, symbol="square"),
                          name=name, hoverinfo="skip", showlegend=True))
            labelled[sgn] = True
    fig.add_trace(go.Scatter(x=comp.index, y=comp.values, mode="lines",
                  line=dict(color=theme.BRAND, width=2), name="Composite z",
                  hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>"))
    for k in (k_entry, -k_entry):
        fig.add_hline(y=k, line=dict(color=p["bear"] if k > 0 else p["bull"],
                      width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    return theme.plotly_layout(fig, mode, height=380)


def _runs(stance: pd.Series):
    """Yield (x0, x1, sign) for each contiguous non-zero stance run."""
    idx, vals = stance.index, stance.to_numpy()
    i, n = 0, len(vals)
    while i < n:
        if vals[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        yield idx[i], idx[j], int(vals[i])
        i = j + 1


# --------------------------------------------------------------------------- #
# Step 2 — net DV01 by tenor (diverging bars)
# --------------------------------------------------------------------------- #
def net_dv01_bars(res, mode: str) -> go.Figure:
    p = theme.pal(mode)
    latest_date = res.fut_dv01["date"].max()
    d = res.fut_dv01[res.fut_dv01["date"] == latest_date]
    net = (d.groupby("instrument")["net_dv01"].sum()
           .reindex(INSTRUMENT_ORDER) / 1e6)
    colors = [p["pos_long"] if v >= 0 else p["pos_short"] for v in net.values]
    fig = go.Figure(go.Bar(x=net.index, y=net.values, marker_color=colors,
                    marker_line_width=0,
                    hovertemplate="%{x}<br>%{y:.2f} $mm/bp<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=1))
    fig.update_layout(showlegend=False)
    return theme.plotly_layout(fig, mode, height=320,
                               title=f"Market net DV01 by tenor · {str(latest_date)[:10]}")


# --------------------------------------------------------------------------- #
# Step 3 — per-category duration positioning (categorical, direct-labeled)
# --------------------------------------------------------------------------- #
def category_duration_fig(res, mode: str) -> go.Figure:
    p = theme.pal(mode)
    cat = res.category_duration
    fig = go.Figure()
    for c in SCORING_CATEGORIES:
        if c not in cat.columns:
            continue
        s = (cat[c] / 1e6).dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                      line=dict(color=p[c], width=2), name=CATEGORY_LABELS[c],
                      hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} $mm/bp<extra></extra>"))
        # direct label at the last point
        fig.add_annotation(x=s.index[-1], y=s.values[-1], text=CATEGORY_LABELS[c],
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=p[c], size=11))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    return theme.plotly_layout(fig, mode, height=360)


# --------------------------------------------------------------------------- #
# Step 3 — curve-crowding z (front minus long-end DV01; + = crowded steepener)
# --------------------------------------------------------------------------- #
def curve_crowding_fig(res, mode: str, k: float = 1.5) -> go.Figure:
    """Per-category curve-crowding z + aggregate. Categories wear their hues; the
    aggregate is a neutral ink emphasis line (blue↔purple fails CVD on dark), with
    direct labels carrying identity."""
    p = theme.pal(mode)
    cc = res.curve_crowding
    fig = go.Figure()
    for c in SCORING_CATEGORIES:
        if c not in cc.columns:
            continue
        s = cc[c].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                      line=dict(color=p[c], width=1.4), name=CATEGORY_LABELS[c],
                      hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>"))
        fig.add_annotation(x=s.index[-1], y=s.values[-1], text=CATEGORY_LABELS[c],
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=p[c], size=11))
    agg = cc["aggregate_restd" if "aggregate_restd" in cc.columns else "aggregate"].dropna()
    fig.add_trace(go.Scatter(x=agg.index, y=agg.values, mode="lines",
                  line=dict(color=p["ink"], width=2.4), name="Aggregate (re-z)",
                  hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>"))
    fig.add_annotation(x=agg.index[-1], y=agg.values[-1], text="Aggregate (re-z)",
                       showarrow=False, xanchor="left", xshift=6,
                       font=dict(color=p["ink"], size=11))
    for kk in (k, -k):
        fig.add_hline(y=kk, line=dict(color=p["muted"], width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    return theme.plotly_layout(fig, mode, height=360,
                               title="Curve-crowding z · + = crowded steepener, − = flattener")


# --------------------------------------------------------------------------- #
# Step 4 — feature panel z heatmap over time (diverging)
# --------------------------------------------------------------------------- #
def feature_heatmap(res, mode: str) -> go.Figure:
    feat = res.feature_panel.dropna(how="all").tail(156)
    fig = go.Figure(go.Heatmap(
        z=feat.T.values, x=feat.index, y=list(feat.columns),
        colorscale=theme.diverging_scale(mode), zmid=0, zmin=-3, zmax=3,
        colorbar=dict(title="z", thickness=12),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>z %{z:.2f}<extra></extra>"))
    return theme.plotly_layout(fig, mode, height=320, title="Standardized features (z), last 3y")


# --------------------------------------------------------------------------- #
# Step 2 — DV01-by-tenor×trader z heatmap (the desk "map")
# --------------------------------------------------------------------------- #
def positioning_heatmap(res, mode: str) -> go.Figure:
    piv = res.fut_dv01.pivot_table(index="date", columns=["category", "instrument"],
                                   values="net_dv01", aggfunc="sum").sort_index()
    z = piv.apply(lambda c: rolling_zscore(c, 156))
    latest = z.dropna(how="all").iloc[-1]
    cats = [c for c in CATEGORY_LABELS if c in piv.columns.get_level_values(0)]
    M = np.full((len(cats), len(INSTRUMENT_ORDER)), np.nan)
    for i, cat in enumerate(cats):
        for j, ins in enumerate(INSTRUMENT_ORDER):
            if (cat, ins) in latest.index:
                M[i, j] = latest[(cat, ins)]
    fig = go.Figure(go.Heatmap(
        z=M, x=INSTRUMENT_ORDER, y=[CATEGORY_LABELS[c] for c in cats],
        colorscale=theme.diverging_scale(mode), zmid=0, zmin=-3, zmax=3,
        text=[[f"{v:+.1f}" if np.isfinite(v) else "" for v in row] for row in M],
        texttemplate="%{text}", textfont={"size": 11},
        colorbar=dict(title="z", thickness=12),
        hovertemplate="%{y} · %{x}<br>z %{z:.2f}<extra></extra>"))
    return theme.plotly_layout(fig, mode, height=280,
                               title="Net-DV01 positioning z · trader × tenor")


# --------------------------------------------------------------------------- #
# Step 5 — PCA loadings + variance explained
# --------------------------------------------------------------------------- #
def loadings_fig(res, mode: str) -> go.Figure:
    L = res.factors.loadings
    fig = go.Figure(go.Heatmap(
        z=L.values, x=list(L.columns), y=list(L.index),
        colorscale=theme.diverging_scale(mode), zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in L.values],
        texttemplate="%{text}", textfont={"size": 11},
        colorbar=dict(thickness=12),
        hovertemplate="%{y} · %{x}<br>loading %{z:.2f}<extra></extra>"))
    return theme.plotly_layout(fig, mode, height=300, title="PCA loadings")


def variance_fig(res, mode: str) -> go.Figure:
    ev = res.factors.explained_var
    xs = [f"PC{i+1}" for i in range(len(ev))]
    fig = go.Figure(go.Bar(x=xs, y=[v * 100 for v in ev], marker_color=theme.BRAND,
                    text=[f"{v:.0%}" for v in ev], textposition="outside",
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"))
    fig.update_layout(showlegend=False)
    return theme.plotly_layout(fig, mode, height=300, title="Variance explained")


# --------------------------------------------------------------------------- #
# Step 7 — predictive overlay
# --------------------------------------------------------------------------- #
def forward_prob_fig(res, mode: str) -> go.Figure:
    p = theme.pal(mode)
    pr = res.predictive.proba
    fig = go.Figure(go.Scatter(x=pr.index, y=pr.values, mode="lines",
                    line=dict(color=theme.BRAND, width=1.4), name="P(selloff)",
                    hovertemplate="%{x|%Y-%m-%d}<br>P %{y:.2f}<extra></extra>"))
    fig.add_hline(y=res.predictive.metrics["base_rate"],
                  line=dict(color=p["muted"], width=1, dash="dash"))
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(showlegend=False)
    return theme.plotly_layout(fig, mode, height=300,
                               title="Walk-forward P(selloff, next 4w)")


def calibration_fig(res, mode: str) -> go.Figure:
    p = theme.pal(mode)
    c = res.predictive.calibration
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                  line=dict(color=p["muted"], width=1, dash="dash"),
                  name="perfect", hoverinfo="skip"))
    if not c.empty:
        fig.add_trace(go.Scatter(x=c["pred_mean"], y=c["realized"], mode="lines+markers",
                      line=dict(color=theme.BRAND, width=2),
                      marker=dict(size=8), name="model",
                      hovertemplate="pred %{x:.2f}<br>realized %{y:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="predicted")
    fig.update_yaxes(range=[0, 1], title="realized")
    fig.update_layout(showlegend=False)
    return theme.plotly_layout(fig, mode, height=300, title="Calibration")


def conditional_baserates_fig(res, mode: str) -> go.Figure:
    """Honest headline overlay: historical contrarian hit rate at crowded extremes, by
    horizon. Long side = P(selloff|crowded long); short side = P(rally|crowded short).
    The asymmetry (long side works, short side ~coin-flip) is the real, interpretable read."""
    p = theme.pal(mode)
    cond = res.latest.get("conditional_base_rates", {})
    fig = go.Figure()
    if cond:
        Hs = sorted(cond, key=int)
        xs = [f"{h}w" for h in Hs]
        p_long = [cond[h]["crowded_long"]["p_selloff"] for h in Hs]
        p_short = [cond[h]["crowded_short"]["p_rally"] for h in Hs]
        bp_long = [cond[h]["crowded_long"]["mean_bp"] for h in Hs]
        fig.add_trace(go.Bar(x=xs, y=p_long, name="P(selloff | crowded LONG)",
                      marker_color="#c00000",
                      text=[f"{b:+.0f}bp" for b in bp_long], textposition="outside",
                      hovertemplate="%{x}: %{y:.0%}<extra>crowded long</extra>"))
        fig.add_trace(go.Bar(x=xs, y=p_short, name="P(rally | crowded SHORT)",
                      marker_color="#2e7d32", opacity=0.65,
                      hovertemplate="%{x}: %{y:.0%}<extra>crowded short</extra>"))
        fig.add_hline(y=0.5, line=dict(color=p["muted"], width=1, dash="dash"))
    fig.update_yaxes(range=[0, 1], title="historical hit rate")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12))
    return theme.plotly_layout(fig, mode, height=300,
                               title="Contrarian hit rate at crowded extremes (|z|>1.5)")


def sentiment_series_fig(series, mode: str) -> go.Figure:
    p = theme.pal(mode)
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines",
                    line=dict(color=theme.BRAND, width=2), name="daily sentiment",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    fig.update_yaxes(range=[-100, 100])
    fig.update_layout(showlegend=False)
    return theme.plotly_layout(fig, mode, height=240,
                               title="Daily FI news sentiment (bullish + / bearish −), last 30d")


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
