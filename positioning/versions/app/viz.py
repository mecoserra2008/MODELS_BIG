"""Plotly figure factories for the methodologies app (reuse the desk theme).

Every figure runs through `theme.plotly_layout` for the shared recessive chrome and the
validated BIG palette. Sign convention across the board matches base.py: POSITIVE = crowded
net LONG (for V4 it is crowded STEEPENER). Crowd bands: +/-1.5 for z-family, 80/20 for the
bounded [0,100] versions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from positioning.app import theme

# Distinct line colours for the Compare overlays (validated categorical + two extras).
_OVERLAY_COLORS = ["#FF8200", "#3987e5", "#199e70", "#7030A0", "#0F9D9D", "#d03b3b"]


def _rgba(hex_color: str, alpha: float) -> str:
    """#RRGGBB -> rgba(r,g,b,alpha) for translucent band fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
# Composite time series with crowd bands
# --------------------------------------------------------------------------- #
def composite_ts(series: pd.Series, bounded: bool, mode: str,
                 title: str | None = None) -> go.Figure:
    """Composite line with crowd bands (80/20 if bounded, else +/-1.5)."""
    p = theme.pal(mode)
    s = series.dropna()
    fig = go.Figure()

    if bounded:
        fig.add_hrect(y0=80, y1=100, fillcolor=_rgba(p["pos_long"], 0.10),
                      line_width=0, layer="below")
        fig.add_hrect(y0=0, y1=20, fillcolor=_rgba(p["pos_short"], 0.10),
                      line_width=0, layer="below")
        for y in (80.0, 20.0):
            fig.add_hline(y=y, line=dict(color=p["muted"], width=1, dash="dash"))
        fig.add_hline(y=50, line=dict(color=p["axis"], width=0.6))
    else:
        span = 2.0
        if len(s):
            span = max(2.0, float(np.nanmax(np.abs(s.values))) * 1.05)
        fig.add_hrect(y0=1.5, y1=span, fillcolor=_rgba(p["pos_long"], 0.10),
                      line_width=0, layer="below")
        fig.add_hrect(y0=-span, y1=-1.5, fillcolor=_rgba(p["pos_short"], 0.10),
                      line_width=0, layer="below")
        for y in (1.5, -1.5):
            fig.add_hline(y=y, line=dict(color=p["muted"], width=1, dash="dash"))
        fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))

    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color=theme.BRAND, width=2), name="composite",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"))
    if bounded:
        fig.update_yaxes(range=[0, 100])
    return theme.plotly_layout(fig, mode, height=380, title=title)


# --------------------------------------------------------------------------- #
# Conditional predictive panel — historical base rate at crowded extremes
# --------------------------------------------------------------------------- #
def conditional_bar(stats: dict, mode: str, uncond: dict | None = None,
                    title: str | None = None) -> go.Figure:
    """P(yield up | crowded long) by horizon, with the unconditional P(up) reference line.

    `stats` is the dict from predictive.conditional_extreme_stats (keys = horizon strings).
    `uncond` optionally maps horizon-string -> unconditional P(yield up) for the ref line.
    """
    p = theme.pal(mode)
    fig = go.Figure()
    Hs = sorted(stats, key=int)
    xs = [f"{h}w" for h in Hs]

    def _grab(h, key):
        v = stats[h].get("crowded_long", {})
        return v.get(key)

    p_long = [_grab(h, "p_selloff") for h in Hs]
    ns = [_grab(h, "n") for h in Hs]
    bp = [_grab(h, "mean_bp") for h in Hs]
    txt = [f"{b:+.0f}bp" if b is not None else "" for b in bp]

    fig.add_trace(go.Bar(
        x=xs, y=p_long, marker_color=theme.BRAND,
        text=txt, textposition="outside",
        customdata=ns,
        hovertemplate="%{x}: P(up)=%{y:.0%}<br>mean Δy in text · n=%{customdata}<extra></extra>",
        name="P(yield up | crowded long)"))

    # unconditional reference line(s): prefer stats' own uncond_p_up, else supplied dict, else 0.5
    ref = None
    stat_unc = stats[Hs[0]].get("uncond_p_up") if Hs else None
    if stat_unc is not None:
        ref = float(stat_unc)
    elif uncond:
        vals = [uncond.get(h) for h in Hs if uncond.get(h) is not None]
        ref = float(np.mean(vals)) if vals else None
    if ref is None:
        ref = 0.5
    fig.add_hline(y=ref, line=dict(color=p["muted"], width=1, dash="dash"),
                  annotation_text=f"unconditional P(up) ≈ {ref:.0%}",
                  annotation_position="top left",
                  annotation_font=dict(color=p["muted"], size=10))

    fig.update_yaxes(range=[0, 1], title="hit rate")
    return theme.plotly_layout(fig, mode, height=300, title=title)


# --------------------------------------------------------------------------- #
# Compare overlay
# --------------------------------------------------------------------------- #
def overlay(series_map: dict, bounded: bool, mode: str,
            title: str | None = None) -> go.Figure:
    """Overlay several composites on one axis. `series_map` = {label: pd.Series}."""
    p = theme.pal(mode)
    fig = go.Figure()
    for (label, s), col in zip(series_map.items(), _OVERLAY_COLORS):
        s = s.dropna()
        if not len(s):
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=label,
                                 line=dict(color=col, width=1.6),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + label + "</extra>"))
    if bounded:
        for y in (80.0, 20.0):
            fig.add_hline(y=y, line=dict(color=p["muted"], width=1, dash="dash"))
        fig.update_yaxes(range=[0, 100])
    else:
        for y in (1.5, -1.5):
            fig.add_hline(y=y, line=dict(color=p["muted"], width=1, dash="dash"))
        fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    return theme.plotly_layout(fig, mode, height=440, title=title)


# --------------------------------------------------------------------------- #
# Predictive coefficient panels (descriptive associations — not tradeable alpha)
# --------------------------------------------------------------------------- #
def _fmt(v, nd: int = 2) -> str:
    """Format a possibly-None/NaN number for hovertext/labels."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if np.isnan(f) else f"{f:.{nd}f}"


def _empty_fig(mode: str, title: str, msg: str = "computing… / unavailable") -> go.Figure:
    """Axis-free placeholder so a stub-returning-empty read never blanks a chart slot."""
    p = theme.pal(mode)
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(color=p["muted"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return theme.plotly_layout(fig, mode, height=300, title=title)


def coef_table(reg: dict, mode: str) -> go.Figure:
    """Horizontal bar of forward-Δ10Y regression β (bp per +1σ) per horizon.

    `reg` = {str(H): {beta_bp_per_sigma, t, p, r2, n, interpretation}} (predict.forward_regression).
    Bars coloured by significance: |t|>1.96 -> brand (strong), else muted. Empty -> placeholder.
    """
    title = "Forward Δ10Y regression: bp per +1σ of the signal"
    if not reg:
        return _empty_fig(mode, title, "computing… (regression unavailable)")

    p = theme.pal(mode)
    Hs = sorted(reg, key=int)
    ys = [f"{h}w" for h in Hs]
    betas, colors, texts, hovers = [], [], [], []
    for h in Hs:
        d = reg[h] or {}
        b = d.get("beta_bp_per_sigma")
        try:
            bf = float(b)
            bf = None if np.isnan(bf) else bf
        except (TypeError, ValueError):
            bf = None
        betas.append(bf)
        t = d.get("t")
        try:
            sig = t is not None and not np.isnan(float(t)) and abs(float(t)) > 1.96
        except (TypeError, ValueError):
            sig = False
        colors.append(theme.BRAND if sig else p["muted"])
        texts.append(f"{bf:+.1f}" if bf is not None else "")
        hovers.append(
            f"β = {_fmt(d.get('beta_bp_per_sigma'), 1)} bp/σ"
            f"<br>t = {_fmt(d.get('t'))} · p = {_fmt(d.get('p'), 3)}"
            f"<br>R² = {_fmt(d.get('r2'), 3)} · n = {d.get('n')}")

    fig = go.Figure(go.Bar(
        x=betas, y=ys, orientation="h", marker_color=colors,
        text=texts, textposition="outside",
        hovertext=hovers, hovertemplate="%{y}<br>%{hovertext}<extra></extra>",
        name="β (bp/σ)"))
    fig.add_vline(x=0, line=dict(color=p["axis"], width=0.8))
    fig.update_xaxes(title="bp per +1σ")
    fig.update_yaxes(autorange="reversed")            # 4w on top, 13w below
    return theme.plotly_layout(fig, mode, height=300, title=title)


def logit_table(logit: dict, mode: str) -> go.Figure:
    """Signed horizontal bar of logistic P(selloff) coefficients (log-odds) per feature.

    `logit` = {coef:{feat:val}, interpretation:{feat:str}, metrics:{auc,n,base_rate}}.
    Brand for +, blue for −; AUC / n / base_rate summarised in the title. Empty -> placeholder.
    """
    base_title = "Logistic P(selloff): signed coefficients (log-odds)"
    coef = (logit or {}).get("coef", {}) or {}
    if not coef:
        return _empty_fig(mode, base_title, "computing… (logistic read unavailable)")

    p = theme.pal(mode)
    metrics = (logit or {}).get("metrics", {}) or {}

    def _finite(v):
        try:
            return np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    bits = []
    if _finite(metrics.get("auc")):
        bits.append(f"AUC={_fmt(metrics.get('auc'))}")
    if metrics.get("n") is not None:
        bits.append(f"n={metrics.get('n')}")
    if _finite(metrics.get("base_rate")):
        bits.append(f"base={float(metrics['base_rate']):.0%}")
    title = base_title + (" · " + " · ".join(bits) if bits else "")

    items = sorted(coef.items(), key=lambda kv: (float(kv[1]) if kv[1] is not None else 0.0))
    feats = [k for k, _ in items]
    vals = [(float(v) if v is not None else 0.0) for _, v in items]
    colors = [theme.BRAND if v >= 0 else p["pos_short"] for v in vals]
    texts = [f"{v:+.2f}" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=texts, textposition="outside",
        hovertemplate="%{y}: log-odds %{x:+.3f}<extra></extra>",
        name="coef (log-odds)"))
    fig.add_vline(x=0, line=dict(color=p["axis"], width=0.8))
    fig.update_xaxes(title="log-odds per +1σ")
    return theme.plotly_layout(fig, mode, height=300, title=title)


# --------------------------------------------------------------------------- #
# V5 Kalman auxiliary: trend (slope) & acceleration when meta carries series
# --------------------------------------------------------------------------- #
def kalman_aux(meta: dict, mode: str) -> go.Figure | None:
    """Small secondary chart of the Kalman trend & acceleration, IF meta carries them as
    Series. Returns None when only latest scalars are available (metrics row covers those)."""
    p = theme.pal(mode)
    specs = (("slope", theme.BRAND, "trend (slope)"),
             ("acceleration", p["asset_mgr"], "acceleration"))
    traces = []
    for key, col, label in specs:
        v = (meta or {}).get(key)
        if isinstance(v, pd.Series):
            s = v.dropna()
            if len(s):
                traces.append((s, col, label))
    if not traces:
        return None
    fig = go.Figure()
    for s, col, label in traces:
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(color=col, width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra>" + label + "</extra>"))
    fig.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    return theme.plotly_layout(fig, mode, height=240,
                               title="Kalman trend & acceleration")
