"""Fixed Income Positioning Monitor — trading-desk Streamlit app (BIG, orange).

Run from the repo root:
    streamlit run positioning/app/streamlit_app.py

Presentation layer over positioning.core.score.run_score. Shows a crowding thermometer,
a per-asset Bull/Bear board (positioning + contrarian signal), and the full methodology
step by step. Dark default with a light toggle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from positioning.app import charts, components, data, theme  # noqa: E402
from positioning.config import (  # noqa: E402
    CATEGORY_LABELS, DEFAULT_START, SCORING_CATEGORIES, Z_LOOKBACK_LONG,
)

st.set_page_config(page_title="FI Positioning Monitor", page_icon="🟠",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #
def sidebar() -> dict:
    with st.sidebar:
        mode = "dark" if st.toggle("Dark theme", value=True) else "light"
        st.markdown("### Data")
        start = st.text_input("History start", DEFAULT_START)
        force = st.button("↻ Refresh data (re-download)", width='stretch')

        st.markdown("### Methodology")
        train_end = st.text_input("PCA train-end", "2019-12-31")
        horizon = st.slider("Predictive horizon (weeks)", 2, 16, 4, 1)
        k_entry = st.slider("Signal entry band (z)", 0.5, 3.0, 1.5, 0.1)
        k_exit = st.slider("Signal exit band (z)", 0.0, 2.0, 0.5, 0.1)
        st.caption(f"Normalization lookback fixed at {Z_LOOKBACK_LONG}w for a stable factor.")

        st.markdown("### Board thresholds")
        pos_thr = st.slider("Crowded if |z| >", 0.0, 2.0, 0.5, 0.1)
        sig_thr = st.slider("Signal if |z| >", 0.5, 3.0, 1.0, 0.1)

        st.markdown("### News sources")
        use_gdelt = st.checkbox("GDELT (1W/1M history)", value=True)
        use_rss = st.checkbox("RSS (live headlines)", value=True)

        with st.expander("Methodology glossary"):
            st.markdown(
                "- **Composite (PC1)**: signed crowding z from PCA over DV01-normalized "
                "positioning. `+` = crowded net long duration.\n"
                "- **Signal**: contrarian — crowded long → fade → bearish duration.\n"
                "- **Buy-side crowding**: leveraged funds + asset managers net DV01, z-scored.\n"
                "- **DV01**: $ P&L per 1bp; makes tenors additive.")
    return dict(mode=mode, start=start, force=force, train_end=train_end,
                horizon=horizon, k_entry=k_entry, k_exit=k_exit,
                pos_thr=pos_thr, sig_thr=sig_thr,
                use_gdelt=use_gdelt, use_rss=use_rss)


# --------------------------------------------------------------------------- #
# Headline deck
# --------------------------------------------------------------------------- #
def headline(res, mode: str) -> None:
    L = res.latest
    left, mid, right = st.columns([1.15, 2.0, 1.15])
    with left:
        st.plotly_chart(components.thermometer(L["percentile"], L["composite_z"], mode),
                        width='stretch', config={"displayModeBar": False},
                        key="thermo")
        pos = "Crowded Long" if L["composite_z"] > 0.5 else \
              "Crowded Short" if L["composite_z"] < -0.5 else "Light / Neutral"
        st.markdown(f"<div style='text-align:center;color:var(--muted);font-size:0.8rem'>"
                    f"{L['percentile']*100:.0f}th pctile · <b>{pos}</b></div>",
                    unsafe_allow_html=True)
    with mid:
        r1 = st.columns(3)
        r1[0].metric("Composite z", f"{L['composite_z']:+.2f}",
                     f"{L['composite_z']:+.2f} vs 0")
        r1[1].metric("Percentile", f"{L['percentile']*100:.0f}%")
        r1[2].metric("Weeks in stance", _weeks_in_stance(res))
        r2 = st.columns(3)
        r2[0].metric("Curve positioning z", f"{L['curve_z']:+.2f}" if L['curve_z'] else "n/a")
        r2[1].metric("Concentration z",
                     f"{L['concentration_z']:+.2f}" if L['concentration_z'] else "n/a")
        r2[2].metric("Cash (dealer) z", f"{L['cash_z']:+.2f}" if L['cash_z'] else "n/a")
    with right:
        components.master_call(L, mode)


def _weeks_in_stance(res) -> str:
    st_ = res.stance.dropna()
    if st_.empty:
        return "—"
    cur = st_.iloc[-1]
    n = 0
    for v in st_.values[::-1]:
        if v == cur:
            n += 1
        else:
            break
    return f"{n}"


# --------------------------------------------------------------------------- #
# Bull/Bear board
# --------------------------------------------------------------------------- #
def board(res, mode: str, pos_thr: float, sig_thr: float) -> None:
    rows = data.board_rows(res, pos_thr=pos_thr, sig_thr=sig_thr)
    st.markdown("#### Bull / Bear board — positioning vs our call")
    tabs = st.tabs(["Contracts", "Curve buckets", "Segments", "Table"])
    with tabs[0]:
        cols = st.columns(len(rows["contracts"]))
        for c, row in zip(cols, rows["contracts"]):
            with c:
                components.asset_tile(row, mode)
    with tabs[1]:
        cols = st.columns(len(rows["buckets"]))
        for c, row in zip(cols, rows["buckets"]):
            with c:
                components.asset_tile(row, mode)
    with tabs[2]:
        cols = st.columns(len(rows["segments"]))
        for c, row in zip(cols, rows["segments"]):
            with c:
                components.asset_tile(row, mode)
    with tabs[3]:
        st.dataframe(data.board_table(rows), width='stretch', hide_index=True)
        st.caption("Table view = accessibility relief for color-only encodings + desk sorting.")


# --------------------------------------------------------------------------- #
# News & Sentiment
# --------------------------------------------------------------------------- #
def news_sentiment(cfg: dict, mode: str) -> None:
    st.markdown("#### News & Sentiment — fixed-income headlines")
    try:
        sent = data.load_sentiment(use_gdelt=cfg["use_gdelt"], use_rss=cfg["use_rss"])
    except Exception as e:
        st.info(f"News/sentiment unavailable: {type(e).__name__}: {e}")
        return
    engine = "FinBERT" if sent.backend == "finbert" else \
        "lexicon (FinBERT unavailable)" if sent.backend == "lexicon" else "none"
    st.markdown(f"<div class='step-note'>Bond-oriented sentiment (positive = bullish FI / "
                f"yields lower). FinBERT supplies tone; a rate-direction layer sets the bond "
                f"sign so “yields surge” reads bearish. Engine: <b>{engine}</b> · "
                f"{sent.windows['1M'].n} headlines cached.</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.3])
    with left:
        cols = st.columns(3)
        for c, k in zip(cols, ("1D", "1W", "1M")):
            w = sent.windows[k]
            c.metric(f"{k} sentiment", f"{w.score:+.0f}",
                     f"{w.n_bull}▲ / {w.n_bear}▼  ·  n={w.n}")
        if len(sent.series) > 1:
            st.plotly_chart(charts.sentiment_series_fig(sent.series, mode),
                            width="stretch", key="sentseries")
    with right:
        st.markdown("**Latest headlines**")
        if len(sent.feed):
            components.sentiment_feed(sent.feed, mode)
        else:
            st.caption("No headlines fetched — feeds may be blocked from this network.")


# --------------------------------------------------------------------------- #
# Extra positioning indicators
# --------------------------------------------------------------------------- #
def extra_indicators(mode: str) -> None:
    st.markdown("#### Extra positioning indicators")
    st.caption("Standalone signals beside the composite (foldable into the PCA later).")
    try:
        inds = data.load_extra_indicators()
    except Exception as e:
        st.info(f"Extra indicators unavailable: {type(e).__name__}: {e}")
        return
    cols = st.columns(len(inds))
    for c, ind in zip(cols, inds):
        with c:
            components.indicator_tile(ind, mode)


# --------------------------------------------------------------------------- #
# Step-by-step methodology
# --------------------------------------------------------------------------- #
def _note(text: str) -> None:
    st.markdown(f"<div class='step-note'>{text}</div>", unsafe_allow_html=True)


def steps(res, mode: str, k_entry: float) -> None:
    st.markdown("#### The score, step by step")
    t = st.tabs(["1 · Raw", "2 · DV01", "3 · Aggregate", "4 · Normalize",
                 "5 · PCA", "6 · Composite", "7 · Predictive", "8 · Signal"])

    with t[0]:
        _note("Raw CFTC Traders-in-Financial-Futures net positions (contracts) by trader "
              "type × tenor, plus NY Fed primary-dealer cash. This is the source material.")
        latest_date = res.fut_dv01["date"].max()
        raw = (res.fut_dv01[res.fut_dv01["date"] == latest_date]
               .pivot_table(index="category", columns="instrument",
                            values="net_contracts", aggfunc="sum")
               .reindex(index=list(CATEGORY_LABELS))
               .rename(index=CATEGORY_LABELS)[charts.INSTRUMENT_ORDER])
        st.caption(f"Net contracts · {str(latest_date)[:10]}")
        st.dataframe(raw.style.format("{:,.0f}"), width='stretch')

    with t[1]:
        _note("Contracts aren't additive across tenors — convert each to <b>DV01</b> "
              "($ per 1bp) so a 2Y and a Bond position are in the same risk unit.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.net_dv01_bars(res, mode), width='stretch',
                            key="dv01bars")
        with c2:
            st.plotly_chart(charts.positioning_heatmap(res, mode), width='stretch',
                            key="posheat")

    with t[2]:
        _note("Sum DV01 across tenors within each trader type → per-category net duration "
              "positioning. Leveraged funds (fast money) and asset managers (real money) "
              "typically sit on opposite sides.")
        st.plotly_chart(charts.category_duration_fig(res, mode), width='stretch',
                        key="catdur")

    with t[3]:
        _note(f"Causal rolling z-score / percentile ({Z_LOOKBACK_LONG}w) puts every series "
              "on one comparable, point-in-time scale. No look-ahead.")
        st.plotly_chart(charts.feature_heatmap(res, mode), width='stretch',
                        key="featheat")

    with t[4]:
        _note("PCA across the standardized features extracts the common factors: "
              "<b>PC1 = level positioning</b> (the composite), <b>PC2 = curve positioning</b>. "
              "This is the factor-weighting that removes double-counting of correlated series.")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(charts.loadings_fig(res, mode), width='stretch',
                            key="load")
        with c2:
            st.plotly_chart(charts.variance_fig(res, mode), width='stretch',
                            key="var")

    with t[5]:
        _note("The composite crowding index = sign-aligned PC1, re-standardized. "
              "Dashed bands = signal entry; shading = contrarian stance in force.")
        st.plotly_chart(charts.composite_fig(res, mode, k_entry), width='stretch',
                        key="comp")

    with t[6]:
        m = res.predictive.metrics
        _note("⚠️ <b>Weak edge — context only.</b> A calibrated logistic maps positioning to "
              f"P(selloff, next 4w). Walk-forward AUC {m['auc']:.2f}, Brier {m['brier']:.2f} "
              f"(base rate {m['base_rate']:.2f}). Positioning is a strong <i>descriptive</i> "
              "gauge but only a modest short-horizon predictor — do not trade this number alone.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.forward_prob_fig(res, mode), width='stretch',
                            key="fwd")
        with c2:
            st.plotly_chart(charts.calibration_fig(res, mode), width='stretch',
                            key="calib")
        coef = pd.Series(res.predictive.coef).sort_values()
        st.caption("Logistic coefficients (feature → log-odds of selloff)")
        st.dataframe(coef.round(3).to_frame("coef").T, width='stretch')

    with t[7]:
        _note("The continuous composite becomes a discrete long/short/flat stance via a "
              "hysteresis band, then sizes into a DV01 tilt. Contrarian: fade the crowd.")
        st_ = res.stance.dropna()
        st.line_chart(st_.tail(260), height=220)
        st.metric("Current stance", res.latest["stance_text"])


# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = sidebar()
    theme.inject_css(cfg["mode"])
    try:
        with st.spinner("Fetching CFTC / NY Fed / FRED and scoring…"):
            res = data.load_score(cfg["start"], cfg["train_end"], cfg["horizon"],
                                  cfg["k_entry"], cfg["k_exit"], force=cfg["force"])
    except Exception as e:  # surface data/scoring errors cleanly on the desk
        st.error(f"Could not build the score: {type(e).__name__}: {e}")
        st.stop()

    theme.header(res.latest["as_of"], data.freshness(res))
    headline(res, cfg["mode"])
    st.divider()
    board(res, cfg["mode"], cfg["pos_thr"], cfg["sig_thr"])
    st.divider()
    extra_indicators(cfg["mode"])
    st.divider()
    news_sentiment(cfg, cfg["mode"])
    st.divider()
    steps(res, cfg["mode"], cfg["k_entry"])


if __name__ == "__main__":
    main()
else:
    main()
