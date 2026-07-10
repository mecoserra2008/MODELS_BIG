"""Visual components: crowding thermometer, positioning/signal chips, asset tiles.

Chips always pair color WITH a word/icon (status-color rule) so meaning is never
carried by color alone. The thermometer is a plotly gauge on the composite percentile.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import theme

_POS_TXT = {"long": "Crowded Long", "short": "Crowded Short", "light": "Light"}
_POS_CLS = {"long": "chip-long", "short": "chip-short", "light": "chip-light"}
_SIG_TXT = {"bull": "▲ Bullish", "bear": "▼ Bearish", "neutral": "● Neutral"}
_SIG_CLS = {"bull": "chip-bull", "bear": "chip-bear", "neutral": "chip-neutral"}


def _chip(text: str, cls: str) -> str:
    return f'<span class="chip {cls}">{text}</span>'


def positioning_chip(kind: str) -> str:
    return _chip(_POS_TXT[kind], _POS_CLS[kind])


def signal_chip(kind: str) -> str:
    return _chip(_SIG_TXT[kind], _SIG_CLS[kind])


# --------------------------------------------------------------------------- #
# Crowding thermometer (composite percentile 0-100, diverging zones)
# --------------------------------------------------------------------------- #
def thermometer(percentile: float, composite_z: float, mode: str) -> go.Figure:
    p = theme.pal(mode)
    val = float(percentile) * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": " pct", "font": {"size": 30, "color": p["ink"]},
                "valueformat": ".0f"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100], "tickvals": [0, 10, 50, 90, 100],
                     "ticktext": ["", "short", "light", "long", ""],
                     "tickcolor": p["muted"], "tickfont": {"color": p["muted"], "size": 11}},
            "bar": {"color": theme.BRAND, "thickness": 0.55},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 10], "color": p["pos_short"]},
                {"range": [10, 35], "color": _mix(p["pos_short"], p["surface"])},
                {"range": [35, 65], "color": p["surface"]},
                {"range": [65, 90], "color": _mix(p["pos_long"], p["surface"])},
                {"range": [90, 100], "color": p["pos_long"]},
            ],
            "threshold": {"line": {"color": p["ink"], "width": 3}, "thickness": 0.85,
                          "value": val},
        },
    ))
    fig.update_layout(
        height=130, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        title={"text": f"<b>Crowding</b> · z {composite_z:+.2f}",
               "font": {"size": 13, "color": p["ink2"]}, "x": 0.02},
    )
    return fig


def _mix(a: str, b: str) -> str:
    """Blend two hex colors 55/45 for the intermediate thermometer zones."""
    ah, bh = a.lstrip("#"), b.lstrip("#")
    ar, ag, ab = int(ah[0:2], 16), int(ah[2:4], 16), int(ah[4:6], 16)
    br, bg, bb = int(bh[0:2], 16), int(bh[2:4], 16), int(bh[4:6], 16)
    m = lambda x, y: int(0.55 * x + 0.45 * y)
    return f"#{m(ar,br):02x}{m(ag,bg):02x}{m(ab,bb):02x}"


# --------------------------------------------------------------------------- #
# Asset tile (two chips + sparkline)
# --------------------------------------------------------------------------- #
def asset_tile(row, mode: str) -> None:
    p = theme.pal(mode)
    st.markdown(f"""
      <div class="tile">
        <div class="name">{row.name} <span class="tenor">{row.sub}</span></div>
        <div class="chips">
          {positioning_chip(row.positioning)}
          {signal_chip(row.signal)}
        </div>
        <div class="zval">buy-side crowding z&nbsp; <b>{row.z:+.2f}</b></div>
      </div>
    """, unsafe_allow_html=True)
    spark = go.Figure(go.Scatter(
        y=row.spark.values, mode="lines",
        line=dict(color=theme.BRAND, width=1.6), hoverinfo="skip"))
    spark.update_layout(height=42, margin=dict(l=0, r=0, t=2, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
    spark.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
    st.plotly_chart(spark, width='stretch',
                    config={"displayModeBar": False}, key=f"spark_{row.name}")


_READ_CHIP = {"bull": ("chip-bull", "▲ Bullish"), "bear": ("chip-bear", "▼ Bearish"),
              "neutral": ("chip-neutral", "● Neutral")}


def indicator_tile(ind, mode: str) -> None:
    """Compact tile for an extra positioning Indicator (z + read chip + sparkline)."""
    if ind.read in _READ_CHIP:
        cls, txt = _READ_CHIP[ind.read]
        chip = _chip(txt, cls)
    else:  # regime words (calm/stressed/normal/elevated leverage/...)
        chip = _chip(ind.read, "chip-light")
    unavail = "" if ind.available else "<span class='tenor'>· unavailable</span>"
    ztxt = f"{ind.z:+.2f}" if ind.available else "—"
    st.markdown(f"""
      <div class="tile">
        <div class="name" title="{ind.note}">{ind.name} {unavail}</div>
        <div class="chips" style="margin-top:8px;">{chip}</div>
        <div class="zval">z&nbsp; <b>{ztxt}</b></div>
      </div>
    """, unsafe_allow_html=True)
    if ind.available and len(ind.series):
        p = theme.pal(mode)
        spark = go.Figure(go.Scatter(y=ind.series.values, mode="lines",
                          line=dict(color=theme.BRAND, width=1.6), hoverinfo="skip"))
        spark.update_layout(height=42, margin=dict(l=0, r=0, t=2, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False), yaxis=dict(visible=False))
        spark.add_hline(y=0, line=dict(color=p["axis"], width=0.8))
        st.plotly_chart(spark, width="stretch", config={"displayModeBar": False},
                        key=f"ind_{ind.key}")


def sentiment_feed(feed, mode: str, limit: int = 18) -> None:
    """Rendered list of recent scored headlines with bull/bear chips."""
    rows = []
    for _, r in feed.head(limit).iterrows():
        cls, _ = _READ_CHIP.get(r["label"], ("chip-neutral", ""))
        ts = str(r["ts"])[:16]
        url = r.get("url") or "#"
        head = r["headline"]
        rows.append(
            f"<div style='display:flex;gap:10px;align-items:baseline;padding:5px 0;"
            f"border-bottom:1px solid var(--border);'>"
            f"<span class='chip {cls}' style='min-width:58px;justify-content:center;'>"
            f"{r['score']:+.0f}</span>"
            f"<a href='{url}' target='_blank' style='color:var(--ink);text-decoration:none;"
            f"flex:1;font-size:0.86rem;'>{head}</a>"
            f"<span class='tenor' style='white-space:nowrap;'>{r['source']} · {ts}</span></div>")
    st.markdown("<div>" + "".join(rows) + "</div>", unsafe_allow_html=True)


def master_call(latest: dict, mode: str) -> None:
    """Aggregate US-duration call: positioning + contrarian signal chips."""
    z = latest["composite_z"]
    pos = "long" if z > 0.5 else "short" if z < -0.5 else "light"
    sig = "bear" if latest["stance"] < 0 else "bull" if latest["stance"] > 0 else "neutral"
    st.markdown(f"""
      <div class="master">
        <div class="lab">US Treasury duration — aggregate</div>
        <div class="big">{_POS_TXT[pos]}</div>
        <div class="chips" style="display:flex; gap:8px; margin-bottom:10px;">
          {positioning_chip(pos)} {signal_chip(sig)}
        </div>
        <div class="lab">Stance</div>
        <div style="color:var(--ink2); font-size:0.9rem; margin-top:2px;">
          {latest['stance_text']}
        </div>
      </div>
    """, unsafe_allow_html=True)
