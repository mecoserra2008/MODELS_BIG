"""
Scenario analysis: topic cluster × tone framing → FGBL Bund futures expected move (bps).
Calibrated from the MD file reaction matrix (5 scenarios) and lexicon weights.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------
# Each cell: expected FGBL price change in ticks (1 tick = 0.01 = ~10 EUR/contract)
# Positive = Bunds rally (dovish), Negative = Bunds sell off (hawkish)
SCENARIO_MATRIX: dict[str, dict[str, float]] = {
    "Inflation": {
        "Persistent / above target": -45,
        "On track / stabilising": 0,
        "Falling / temporary": +35,
    },
    "Energy": {
        "Feeding into core": -35,
        "Monitoring / unclear": -5,
        "Supply shock fading": +30,
    },
    "Wages": {
        "Second-round effects / strong": -40,
        "Stabilising": 0,
        "No pass-through": +25,
    },
    "Services": {
        "Elevated / persistent": -30,
        "Moderate": 0,
        "Easing": +20,
    },
    "Growth": {
        "Resilient / strong demand": -10,
        "Moderate / balanced": 0,
        "Downside risks dominant": +35,
    },
    "Forward Guidance": {
        "Further hikes warranted": -60,
        "Meeting-by-meeting / data-dependent": 0,
        "One-and-done / pause": +55,
    },
    "Uncertainty": {
        "Inflation uncertainty dominates": -20,
        "Balanced risks": 0,
        "Growth uncertainty dominates": +25,
    },
}

# Framing labels ordered hawkish → dovish per topic
TOPIC_FRAMINGS: dict[str, list[str]] = {k: list(v.keys()) for k, v in SCENARIO_MATRIX.items()}

# Topic weights (how much each topic moves FGBL, relative)
TOPIC_WEIGHTS: dict[str, float] = {
    "Forward Guidance": 0.30,
    "Inflation": 0.22,
    "Wages": 0.15,
    "Energy": 0.13,
    "Growth": 0.10,
    "Services": 0.06,
    "Uncertainty": 0.04,
}

# Pre-defined named scenarios matching the MD file reaction matrix
NAMED_SCENARIOS: list[dict] = [
    {
        "name": "Hawkish Hike",
        "description": "25bp + strong next-hike signal",
        "selections": {
            "Inflation": "Persistent / above target",
            "Energy": "Feeding into core",
            "Wages": "Second-round effects / strong",
            "Services": "Elevated / persistent",
            "Growth": "Resilient / strong demand",
            "Forward Guidance": "Further hikes warranted",
            "Uncertainty": "Inflation uncertainty dominates",
        },
        "fgbl_label": "FGBL sharply lower",
    },
    {
        "name": "Hike + Data-Dependent",
        "description": "25bp + guidance less committal",
        "selections": {
            "Inflation": "On track / stabilising",
            "Energy": "Monitoring / unclear",
            "Wages": "Stabilising",
            "Services": "Moderate",
            "Growth": "Moderate / balanced",
            "Forward Guidance": "Meeting-by-meeting / data-dependent",
            "Uncertainty": "Balanced risks",
        },
        "fgbl_label": "FGBL neutral to higher",
    },
    {
        "name": "Dovish Hike",
        "description": "25bp + growth downside emphasis",
        "selections": {
            "Inflation": "Falling / temporary",
            "Energy": "Supply shock fading",
            "Wages": "No pass-through",
            "Services": "Easing",
            "Growth": "Downside risks dominant",
            "Forward Guidance": "One-and-done / pause",
            "Uncertainty": "Growth uncertainty dominates",
        },
        "fgbl_label": "FGBL higher",
    },
    {
        "name": "No Hike (Surprise)",
        "description": "Dovish surprise — no rate change",
        "selections": {
            "Inflation": "Falling / temporary",
            "Energy": "Supply shock fading",
            "Wages": "No pass-through",
            "Services": "Easing",
            "Growth": "Downside risks dominant",
            "Forward Guidance": "One-and-done / pause",
            "Uncertainty": "Growth uncertainty dominates",
        },
        "fgbl_label": "FGBL sharply higher",
        "override_bps": +120,
    },
    {
        "name": "Persistent Inflation Narrative",
        "description": "Upgraded projections + wage/services concerns",
        "selections": {
            "Inflation": "Persistent / above target",
            "Energy": "Feeding into core",
            "Wages": "Second-round effects / strong",
            "Services": "Elevated / persistent",
            "Growth": "Resilient / strong demand",
            "Forward Guidance": "Further hikes warranted",
            "Uncertainty": "Inflation uncertainty dominates",
        },
        "fgbl_label": "FGBL lower; curve bear-flattens",
        "override_bps": -80,
    },
]


def compute_composite_bps(selections: dict[str, str]) -> float:
    """Weighted average of FGBL bps move across topics for given framing selections."""
    total_bps = 0.0
    total_weight = 0.0
    for topic, framing in selections.items():
        bps = SCENARIO_MATRIX.get(topic, {}).get(framing, 0.0)
        w = TOPIC_WEIGHTS.get(topic, 0.05)
        total_bps += bps * w
        total_weight += w
    return round(total_bps / max(total_weight, 1e-9), 1)


def compute_barometer_from_selections(selections: dict[str, str]) -> float:
    """Map scenario selections to a barometer value [0, 100]."""
    bps = compute_composite_bps(selections)
    # Map from [-80, +80] bps range to [0, 100] (inverted: positive bps = dovish = lower barometer)
    raw = 50.0 - (bps / 80.0) * 35.0
    return float(np.clip(raw, 0, 100))


def named_scenario_table() -> pd.DataFrame:
    """DataFrame of named scenarios with composite bps and barometer."""
    rows = []
    for sc in NAMED_SCENARIOS:
        bps = sc.get("override_bps", compute_composite_bps(sc["selections"]))
        bar = 50.0 - (bps / 80.0) * 35.0
        rows.append({
            "Scenario": sc["name"],
            "Description": sc["description"],
            "Composite FGBL (bps)": bps,
            "Barometer": round(float(np.clip(bar, 0, 100)), 1),
            "FGBL Interpretation": sc["fgbl_label"],
        })
    return pd.DataFrame(rows)


def build_heatmap(user_selections: dict[str, str] | None = None) -> go.Figure:
    """
    Interactive Plotly heatmap: topics (rows) × framings (cols) → FGBL bps.
    Highlights user-selected cells.
    """
    topics = list(SCENARIO_MATRIX.keys())
    # Use 3 columns: hawkish / neutral / dovish
    framings_aligned = ["Hawkish Framing", "Neutral Framing", "Dovish Framing"]

    z = np.zeros((len(topics), 3))
    text_labels = [[""] * 3 for _ in range(len(topics))]

    for i, topic in enumerate(topics):
        vals = list(SCENARIO_MATRIX[topic].values())
        keys = list(SCENARIO_MATRIX[topic].keys())
        for j in range(min(3, len(vals))):
            z[i, j] = vals[j]
            text_labels[i][j] = f"{keys[j]}<br>{'+' if vals[j]>0 else ''}{vals[j]:.0f}bp"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=framings_aligned,
        y=topics,
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=[
            [0.0, "#c0392b"],    # very hawkish (large negative)
            [0.4, "#e67e22"],
            [0.5, "#f5f5f5"],    # neutral
            [0.6, "#27ae60"],
            [1.0, "#1a237e"],    # very dovish (large positive)
        ],
        zmid=0,
        colorbar=dict(
            title="FGBL bps",
            thickness=12,
            len=0.8,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Expected FGBL: %{z:+.0f} bps<extra></extra>",
    ))

    # Highlight user selections
    if user_selections:
        for i, topic in enumerate(topics):
            sel = user_selections.get(topic)
            if sel:
                keys = list(SCENARIO_MATRIX[topic].keys())
                if sel in keys:
                    j = keys.index(sel)
                    if j < 3:
                        fig.add_shape(
                            type="rect",
                            x0=j - 0.48, x1=j + 0.48,
                            y0=i - 0.48, y1=i + 0.48,
                            line=dict(color="#FF8200", width=3),
                        )

    fig.update_layout(
        title=dict(
            text="Scenario Matrix: Topic × Tone Framing → FGBL Bund Futures (bps)",
            font=dict(size=13, color="#3C3C3C"),
        ),
        xaxis_title="Tone Framing",
        yaxis_title="Topic",
        template="plotly_white",
        font=dict(family="Helvetica Neue, Arial", size=11),
        margin=dict(l=130, r=100, t=60, b=40),
        height=420,
    )
    return fig


def sensitivity_table(topic: str, fixed_selections: dict[str, str]) -> pd.DataFrame:
    """How sensitive FGBL is to changing one topic's framing."""
    rows = []
    for framing, bps in SCENARIO_MATRIX.get(topic, {}).items():
        test_sel = {**fixed_selections, topic: framing}
        composite = compute_composite_bps(test_sel)
        rows.append({
            "Topic": topic,
            "Framing": framing,
            "Standalone FGBL (bps)": bps,
            "Composite FGBL (bps)": composite,
        })
    return pd.DataFrame(rows)
