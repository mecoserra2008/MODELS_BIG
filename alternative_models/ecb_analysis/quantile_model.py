"""
Quantile regression of Lagarde's hawkishness score over speech-time.
Fits τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90} curves per meeting and pooled.

Reuses MacroQuantileRegression from alternative_models/src/probabilistic.py
via direct QuantReg call (statsmodels).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.regression.quantile_regression import QuantReg

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lexicon import score_segment, to_barometer, top_terms

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
COLORS = {
    0.10: "rgba(31,119,180,0.15)",
    0.25: "rgba(31,119,180,0.25)",
    0.50: "#FF8200",
    0.75: "rgba(31,119,180,0.25)",
    0.90: "rgba(31,119,180,0.15)",
}


def _build_design(n: int, degree: int = 2) -> np.ndarray:
    """Design matrix: [1, t, t^2] where t ∈ [0,1]."""
    t = np.linspace(0, 1, n)
    cols = [np.ones(n)]
    for d in range(1, degree + 1):
        cols.append(t ** d)
    return np.column_stack(cols), t


def score_corpus(segments: list[dict]) -> pd.DataFrame:
    """Score every segment; return DataFrame with per-segment barometer values."""
    rows = []
    for seg in segments:
        raw = score_segment(seg["text"])
        bar = to_barometer(raw)
        rows.append({
            "meeting_date": seg["meeting_date"],
            "segment_index": seg["segment_index"],
            "speaker": seg.get("speaker", ""),
            "text": seg["text"],
            "raw_score": raw,
            "barometer": bar,
            "bund_reaction": seg.get("bund_reaction", "neutral"),
            "label": seg.get("bund_reaction", ""),
            "word_count": seg.get("word_count", len(seg["text"].split())),
            "top_term": top_terms(seg["text"], n=1)[0][0] if top_terms(seg["text"], n=1) else "",
        })
    return pd.DataFrame(rows)


def fit_quantile_regressions(df: pd.DataFrame) -> dict[str, dict]:
    """
    Fit per-meeting and pooled quantile regressions.
    Returns dict: meeting_date -> {tau -> (alpha, beta_t, beta_t2, fitted_values)}
    """
    results: dict[str, dict] = {}

    groups = {"pooled": df}
    for d in df["meeting_date"].unique():
        groups[d] = df[df["meeting_date"] == d].reset_index(drop=True)

    for key, grp in groups.items():
        n = len(grp)
        if n < 6:
            continue
        X, t = _build_design(n)
        y = grp["barometer"].values
        meeting_results: dict[float, dict] = {}

        for tau in QUANTILES:
            try:
                model = QuantReg(y, X)
                res = model.fit(q=tau, max_iter=2000)
                fitted = res.fittedvalues
                meeting_results[tau] = {
                    "params": res.params.tolist(),
                    "fitted": fitted.tolist(),
                    "t": t.tolist(),
                }
            except Exception as exc:
                # fallback: simple quantile of y
                q_val = float(np.quantile(y, tau))
                meeting_results[tau] = {
                    "params": [q_val, 0.0, 0.0],
                    "fitted": [q_val] * n,
                    "t": np.linspace(0, 1, n).tolist(),
                }

        results[key] = meeting_results
    return results


def top_shift_drivers(df: pd.DataFrame, window: int = 3) -> list[dict]:
    """
    Identify segments where the Q50 barometer jumps most (positive or negative)
    and return the top-3 driving terms for each jump.
    """
    if len(df) < window + 1:
        return []

    scores = df["barometer"].values
    shifts = []
    for i in range(window, len(scores)):
        delta = scores[i] - scores[i - window]
        if abs(delta) > 5.0:
            seg_text = df.iloc[i]["text"]
            drivers = top_terms(seg_text, n=3)
            shifts.append({
                "segment_index": int(df.iloc[i]["segment_index"]),
                "meeting_date": df.iloc[i]["meeting_date"],
                "delta": round(float(delta), 2),
                "direction": "more hawkish" if delta > 0 else "more dovish",
                "text_snippet": seg_text[:120] + "...",
                "drivers": [(t, round(w, 2)) for t, w in drivers],
            })
    shifts.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return shifts[:10]


def build_fan_chart(df: pd.DataFrame, qr_results: dict, meeting: str = "pooled") -> go.Figure:
    """
    Plotly fan chart: shaded quantile bands + Q50 line, annotated with shift drivers.
    """
    grp = df if meeting == "pooled" else df[df["meeting_date"] == meeting].reset_index(drop=True)
    n = len(grp)
    if n == 0 or meeting not in qr_results:
        return go.Figure()

    res = qr_results[meeting]
    t_vals = np.linspace(0, 1, n) * 100  # x-axis: % through speech

    fig = go.Figure()

    # Shaded bands: Q10–Q90 outer, Q25–Q75 inner
    q10 = np.array(res[0.10]["fitted"])
    q25 = np.array(res[0.25]["fitted"])
    q50 = np.array(res[0.50]["fitted"])
    q75 = np.array(res[0.75]["fitted"])
    q90 = np.array(res[0.90]["fitted"])

    fig.add_trace(go.Scatter(
        x=np.concatenate([t_vals, t_vals[::-1]]),
        y=np.concatenate([q90, q10[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.08)",
        line=dict(width=0), name="Q10–Q90", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_vals, t_vals[::-1]]),
        y=np.concatenate([q75, q25[::-1]]),
        fill="toself", fillcolor="rgba(31,119,180,0.18)",
        line=dict(width=0), name="Q25–Q75", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=t_vals, y=q50, mode="lines",
        line=dict(color="#FF8200", width=2.5), name="Median (Q50)"
    ))

    # Raw segment scores
    raw_x = np.linspace(0, 100, n)
    fig.add_trace(go.Scatter(
        x=raw_x, y=grp["barometer"].values,
        mode="markers", marker=dict(color="#333", size=5, opacity=0.5),
        name="Segment score", hovertext=grp["text"].str[:80]
    ))

    # Shift annotations
    shifts = top_shift_drivers(grp)
    for sh in shifts[:4]:
        xi = sh["segment_index"] / max(n - 1, 1) * 100
        yi = grp.iloc[min(sh["segment_index"], n - 1)]["barometer"]
        driver_str = ", ".join(t for t, _ in sh["drivers"])
        fig.add_annotation(
            x=xi, y=yi + 3,
            text=f"{'↑' if sh['delta'] > 0 else '↓'} {driver_str}",
            showarrow=True, arrowhead=2, font=dict(size=9, color="#FF8200"),
            arrowcolor="#FF8200", ax=0, ay=-25
        )

    # Reference lines
    for level, label, colour in [(65, "Hawkish", "red"), (35, "Dovish", "blue"), (50, "Neutral", "grey")]:
        fig.add_hline(y=level, line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=label, annotation_position="right")

    title = "Lagarde Tone Distribution — Pooled (All Meetings)" if meeting == "pooled" else f"Tone Distribution — {meeting}"
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#3C3C3C")),
        xaxis_title="Speech Progress (%)",
        yaxis_title="Hawkishness Barometer (0=Dovish, 100=Hawkish)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        font=dict(family="Helvetica Neue, Arial", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-meeting summary: mean, median, Q25, Q75, Q10, Q90 barometer values."""
    rows = []
    for d, grp in df.groupby("meeting_date"):
        b = grp["barometer"]
        rows.append({
            "Meeting": d,
            "N Segments": len(grp),
            "Mean": round(b.mean(), 1),
            "Median": round(b.median(), 1),
            "Q10": round(b.quantile(0.10), 1),
            "Q25": round(b.quantile(0.25), 1),
            "Q75": round(b.quantile(0.75), 1),
            "Q90": round(b.quantile(0.90), 1),
            "Label": "Hawkish" if b.mean() > 60 else ("Dovish" if b.mean() < 40 else "Neutral"),
        })
    return pd.DataFrame(rows)
