"""Theme: validated palette + brand CSS injection for the positioning desk app.

The palette was validated with the dataviz skill's checker (CVD, contrast, bands) —
do not hand-edit hexes without re-validating. Semantic split that drives the whole UI:

    orange / blue   -> WHERE the market is positioned (descriptive)
    green  / red    -> OUR call (contrarian trade signal), always with an icon + label

Two selected modes (dark default, light toggle); each categorical hue is stepped for its
own surface, not auto-flipped.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

BRAND = "#FF6002"          # BIG orange
BRAND_DEEP = "#B23F00"
LOGO_PATH = Path(__file__).resolve().parents[2] / "Logo-BIG.png"


# --------------------------------------------------------------------------- #
# Palette per mode
# --------------------------------------------------------------------------- #
PALETTES = {
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d", "card": "#1f1f1e",
        "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
        # categorical (trader types) — validated dark set
        "lev_money": "#E5611F", "asset_mgr": "#3987e5", "dealer": "#199e70",
        # diverging poles (signed positioning): long=orange, short=blue; gray mid
        "pos_long": "#E5611F", "pos_short": "#3987e5", "pos_mid": "#383835",
        # status (trade signal)
        "bull": "#0ca30c", "bear": "#d03b3b", "neutral": "#898781",
    },
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7", "card": "#ffffff",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
        "lev_money": "#FF6002", "asset_mgr": "#2a78d6", "dealer": "#1baf7a",
        "pos_long": "#FF6002", "pos_short": "#2a78d6", "pos_mid": "#f0efec",
        "bull": "#0ca30c", "bear": "#d03b3b", "neutral": "#898781",
    },
}

CATEGORY_COLOR_KEYS = {"lev_money": "lev_money", "asset_mgr": "asset_mgr", "dealer": "dealer"}


def pal(mode: str) -> dict:
    return PALETTES["dark" if mode == "dark" else "light"]


def diverging_scale(mode: str) -> list:
    """Plotly colorscale short(blue) -> gray -> long(orange) for signed z heatmaps."""
    p = pal(mode)
    return [[0.0, p["pos_short"]], [0.5, p["pos_mid"]], [1.0, p["pos_long"]]]


def orange_sequential(mode: str) -> list:
    """Light->deep orange ramp for crowding intensity (magnitude)."""
    if mode == "dark":
        return [[0.0, "#2a1a10"], [0.5, "#B23F00"], [1.0, "#FF7A2E"]]
    return [[0.0, "#FFE8D6"], [0.5, "#FF8A3D"], [1.0, "#B23F00"]]


# --------------------------------------------------------------------------- #
# Logo + CSS
# --------------------------------------------------------------------------- #
def _logo_b64() -> str | None:
    try:
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except Exception:
        return None


def inject_css(mode: str) -> None:
    """Global brand CSS: header lockup, chips, tiles, tabs, metric accents."""
    p = pal(mode)
    st.markdown(f"""
    <style>
      :root {{ --brand:{BRAND}; --ink:{p['ink']}; --ink2:{p['ink2']};
               --muted:{p['muted']}; --card:{p['card']}; --border:{p['border']};
               --plane:{p['plane']}; --surface:{p['surface']};
               --bull:{p['bull']}; --bear:{p['bear']}; --neutral:{p['neutral']};
               --long:{p['pos_long']}; --short:{p['pos_short']}; }}
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }}
      /* header bar */
      .big-header {{ display:flex; align-items:center; gap:16px;
        border-bottom:3px solid var(--brand); padding:6px 2px 12px; margin-bottom:8px; }}
      .big-header img {{ height:44px; }}
      .big-header .ttl {{ font-size:1.5rem; font-weight:800; color:var(--ink);
        letter-spacing:-0.01em; }}
      .big-header .sub {{ color:var(--muted); font-size:0.82rem; margin-left:auto;
        text-align:right; line-height:1.35; }}
      /* chips */
      .chip {{ display:inline-flex; align-items:center; gap:6px; font-weight:700;
        font-size:0.74rem; padding:3px 9px; border-radius:999px; white-space:nowrap; }}
      .chip-long {{ background:color-mix(in srgb, var(--long) 20%, transparent);
        color:var(--long); border:1px solid var(--long); }}
      .chip-short {{ background:color-mix(in srgb, var(--short) 20%, transparent);
        color:var(--short); border:1px solid var(--short); }}
      .chip-light {{ background:transparent; color:var(--muted); border:1px solid var(--muted); }}
      .chip-bull {{ background:color-mix(in srgb, var(--bull) 20%, transparent);
        color:var(--bull); border:1px solid var(--bull); }}
      .chip-bear {{ background:color-mix(in srgb, var(--bear) 20%, transparent);
        color:var(--bear); border:1px solid var(--bear); }}
      .chip-neutral {{ background:transparent; color:var(--muted); border:1px solid var(--muted); }}
      /* asset tile */
      .tile {{ background:var(--card); border:1px solid var(--border);
        border-radius:12px; padding:10px 12px 12px; }}
      .tile .name {{ font-weight:800; font-size:0.95rem; color:var(--ink); }}
      .tile .tenor {{ color:var(--muted); font-size:0.72rem; font-weight:600; }}
      .tile .chips {{ display:flex; flex-direction:column; gap:5px; margin-top:8px; }}
      .tile .zval {{ font-variant-numeric:tabular-nums; color:var(--ink2);
        font-size:0.72rem; margin-top:6px; }}
      /* master call card */
      .master {{ background:var(--card); border:1px solid var(--border);
        border-left:4px solid var(--brand); border-radius:12px; padding:14px 16px; }}
      .master .lab {{ color:var(--muted); font-size:0.72rem; font-weight:700;
        text-transform:uppercase; letter-spacing:0.04em; }}
      .master .big {{ font-size:1.35rem; font-weight:800; color:var(--ink); margin:4px 0 10px; }}
      /* app owns bg + text so it doesn't depend on .streamlit/config.toml (not loaded
         when run from repo root) and tracks the runtime light/dark toggle. Streamlit
         paints the background on stAppViewContainer/stMain, so target those. */
      [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {{
        background:var(--plane) !important; color:var(--ink); }}
      [data-testid="stHeader"] {{ background:transparent !important; }}
      [data-testid="stSidebar"] {{ background:var(--surface) !important; }}
      [data-testid="stSidebar"] * {{ color:var(--ink); }}
      /* text inputs / widgets: dark field, legible value */
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
        background:var(--card) !important; color:var(--ink) !important;
        border:1px solid var(--border) !important; }}
      [data-testid="stExpander"] {{ background:var(--card); border:1px solid var(--border);
        border-radius:8px; }}
      [data-testid="stDataFrame"] {{ background:var(--card); }}
      .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color:var(--ink); }}
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
        color:var(--muted) !important; }}
      /* streamlit metric orange accent + mode-aware text colours (fixes dark-on-dark) */
      div[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--border);
        border-left:3px solid var(--brand); border-radius:10px; padding:10px 12px; }}
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color:var(--ink) !important; }}
      div[data-testid="stMetric"] [data-testid="stMetricLabel"],
      div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {{ color:var(--muted) !important; }}
      div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {{ color:var(--ink2) !important; }}
      .stTabs [aria-selected="true"] {{ color:var(--brand) !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background:var(--brand) !important; }}
      .step-note {{ color:var(--ink2); font-size:0.86rem; background:var(--card);
        border-left:3px solid var(--brand); border-radius:6px; padding:8px 12px; margin:2px 0 12px; }}
      .freshpill {{ display:inline-block; font-size:0.68rem; font-weight:700; padding:2px 8px;
        border-radius:999px; border:1px solid var(--border); color:var(--ink2); }}
    </style>
    """, unsafe_allow_html=True)


def header(as_of: str, freshness: str) -> None:
    b64 = _logo_b64()
    img = f'<img src="data:image/png;base64,{b64}"/>' if b64 else ""
    st.markdown(f"""
    <div class="big-header">
      {img}
      <span class="ttl">Fixed Income Positioning Monitor</span>
      <span class="sub">US Treasury futures complex · CFTC TFF + NY Fed + FRED<br>
        as&nbsp;of <b>{as_of}</b> &nbsp;·&nbsp; <span class="freshpill">{freshness}</span></span>
    </div>
    """, unsafe_allow_html=True)


def plotly_layout(fig, mode: str, height: int = 360, title: str | None = None):
    """Apply consistent, recessive chrome to a plotly figure (single-axis, brand ink)."""
    p = pal(mode)
    fig.update_layout(
        template=None, height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=p["ink2"], family="system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12),
        margin=dict(l=48, r=20, t=40 if title else 16, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["axis"], linecolor=p["axis"],
                     showgrid=True, tickfont=dict(color=p["muted"], size=11))
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["axis"], linecolor=p["axis"],
                     showgrid=True, tickfont=dict(color=p["muted"], size=11))
    return fig
