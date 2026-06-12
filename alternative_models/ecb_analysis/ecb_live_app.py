"""
ECB Communication Intelligence — 5-Tab Streamlit App

Live refresh via streamlit-autorefresh (JavaScript setInterval → full rerun every 3 s).
This is reliable in Codespace proxy environments where @st.fragment(run_every=N) is not.

Architecture:
  - Background threads (audio_pipe, market_reader) ONLY write to files / thread-safe dicts.
  - All session state reads/writes happen in the main Streamlit thread on each rerun.
  - _pull_feed() reads new bytes from the feed file on every rerun (fast, <1 ms).
  - Expensive computations (corpus, QR, t-SNE) are cached in session state.

Run:
    cd /workspaces/MODELS_BIG/alternative_models/ecb_analysis
    streamlit run ecb_live_app.py --server.port 8502
"""

from __future__ import annotations
import json
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexicon import score_segment, to_barometer, top_terms, label as tone_label
from transcript_loader import load_historical, FALLBACK_CORPUS
from quantile_model import (
    score_corpus, fit_quantile_regressions, build_fan_chart,
    summary_table, top_shift_drivers,
)
from tsne_bund import run_tsne_pipeline, cluster_summary
from scenario import (
    build_heatmap, named_scenario_table, TOPIC_FRAMINGS,
    compute_composite_bps, compute_barometer_from_selections,
)
from bayesian_barometer import BayesianBarometerState, build_gauge_trace
from market_reader import get_current_price, manual_update, autostart, fetch_ecb_bund_history
from audio_pipe import start_pipeline, stop_pipeline, simulate_feed, FEED_FILE
from audio_pipe import is_running as audio_is_running
import news_reader

BIG_ORANGE = "#FF8200"
BIG_GREY   = "#3C3C3C"

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECB Communication Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh: JavaScript setInterval triggers a full rerun every 3 seconds.
# Returns the cumulative tick count (int) so we can detect live ticks in the UI.
# Only active when a live session is in progress (sidebar toggle).
# ─────────────────────────────────────────────────────────────────────────────
_autorefresh_active = st.session_state.get("autorefresh_on", False)
refresh_count = st_autorefresh(
    interval=3000,       # ms between reruns
    limit=None,          # run indefinitely
    debounce=False,
    key="ecb_autorefresh",
) if _autorefresh_active else 0

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{
      font-family: "Helvetica Neue", Arial, sans-serif;
      font-size: 0.9rem; font-weight: 600;
      padding: 6px 16px; border-radius: 4px 4px 0 0;
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {BIG_ORANGE} !important; color: white !important;
  }}
  div[data-testid="metric-container"] {{
      background: #f9f9f9; border-left: 3px solid {BIG_ORANGE};
      padding: 8px 12px; border-radius: 4px;
  }}
  .alert-box {{
      background: #fff3cd; border-left: 4px solid {BIG_ORANGE};
      padding: 10px 14px; border-radius: 4px; margin: 6px 0;
  }}
  .tick-badge {{
      background: #27ae60; color: white; padding: 2px 8px;
      border-radius: 10px; font-size: 0.78rem; font-weight: 700;
  }}
  .idle-badge {{
      background: #7f8c8d; color: white; padding: 2px 8px;
      border-radius: 10px; font-size: 0.78rem;
  }}
  .hawkish-badge {{ background:#c0392b; color:white; padding:2px 8px; border-radius:12px; font-size:0.78rem }}
  .dovish-badge  {{ background:#2980b9; color:white; padding:2px 8px; border-radius:12px; font-size:0.78rem }}
  .neutral-badge {{ background:#7f8c8d; color:white; padding:2px 8px; border-radius:12px; font-size:0.78rem }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults  (only set on first run)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "barometer":        BayesianBarometerState(),
    "live_segments":    [],
    "feed_pos":         0,        # byte offset in FEED_FILE already consumed
    "historical_df":    None,
    "historical_qr":    None,
    "tsne_df":          None,
    "tsne_fig":         None,
    "market_started":   False,
    "autorefresh_on":   False,
    "last_segment_ts":  None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# _pull_feed: read new bytes from the feed file and update barometer
# Called on EVERY rerun (cheap — just a stat() + seek/read of new bytes)
# ─────────────────────────────────────────────────────────────────────────────
def _pull_feed() -> int:
    if not FEED_FILE.exists():
        return 0
    current_size = FEED_FILE.stat().st_size
    if current_size <= st.session_state.feed_pos:
        return 0

    with open(FEED_FILE, "r", encoding="utf-8") as f:
        f.seek(st.session_state.feed_pos)
        new_content = f.read()
    st.session_state.feed_pos = current_size

    ingested = 0
    for line in new_content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
        except json.JSONDecodeError:
            text = line
        if not text or len(text) < 10:
            continue

        raw = score_segment(text)
        bar = to_barometer(raw)
        idx = len(st.session_state.live_segments)
        result = st.session_state.barometer.update(idx, bar)
        st.session_state.live_segments.append({
            "meeting_date": "2026-06-11",
            "segment_index": idx,
            "speaker":       "Lagarde",
            "text":          text,
            "barometer":     bar,
            "raw_score":     raw,
            "result":        result,
        })
        st.session_state.last_segment_ts = datetime.now(timezone.utc).isoformat()
        ingested += 1
    return ingested


# ── Pull feed on every rerun ─────────────────────────────────────────────────
new_segments = _pull_feed()


# ─────────────────────────────────────────────────────────────────────────────
# Load historical corpus (once, cached in session state)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.historical_df is None:
    with st.spinner("Loading historical ECB corpus…"):
        segs    = load_historical()
        df_hist = score_corpus(segs)
        qr_hist = fit_quantile_regressions(df_hist)
        st.session_state.historical_df = df_hist
        st.session_state.historical_qr = qr_hist

hist_df: pd.DataFrame = st.session_state.historical_df
hist_qr: dict         = st.session_state.historical_qr


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("../Logo-BIG.png", use_container_width=True)
    except Exception:
        pass
    st.markdown(
        f"<div style='color:{BIG_ORANGE};font-weight:700;font-size:1.1rem'>"
        "ECB Intelligence</div>",
        unsafe_allow_html=True,
    )
    st.caption("June 11, 2026 | Press conference: ~14:45 CET")
    st.divider()

    # ── Live refresh toggle ──────────────────────────────────────────────────
    st.markdown("**Auto-Refresh**")
    refresh_on = st.toggle(
        "Live mode (refresh every 3 s)",
        value=st.session_state.autorefresh_on,
        key="toggle_refresh",
    )
    if refresh_on != st.session_state.autorefresh_on:
        st.session_state.autorefresh_on = refresh_on
        st.rerun()

    if st.session_state.autorefresh_on:
        st.markdown(
            f"<span class='tick-badge'>🔄 LIVE — tick #{refresh_count}</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span class='idle-badge'>⏸ Paused</span>", unsafe_allow_html=True)
        if st.button("Single refresh", use_container_width=True):
            st.rerun()

    st.divider()

    # ── Market feed ──────────────────────────────────────────────────────────
    st.markdown("**Market Data**")
    td_key = st.text_input(
        "TwelveData API key (optional)", type="password",
        placeholder="free at twelvedata.com",
    )
    if st.button("Start Market Feed", use_container_width=True):
        if not st.session_state.market_started:
            autostart(td_key or None)
            st.session_state.market_started = True
            st.success("Market feed started.")

    st.divider()

    # ── Live data source ─────────────────────────────────────────────────────
    st.markdown("**Live Text Source**")
    source_choice = st.radio(
        "Data source:",
        ["FinancialJuice (web scrape)", "Live Audio (ffmpeg + Whisper)", "Simulate (test mode)"],
        index=0,
        key="source_choice",
    )

    # Source status badge
    if news_reader.is_running():
        src_label = news_reader.active_source()
        st.markdown(
            f"<span class='tick-badge'>🟢 {src_label}</span>",
            unsafe_allow_html=True,
        )
    elif audio_is_running():
        st.markdown("<span class='tick-badge'>🎙️ Audio pipeline running</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='idle-badge'>⏹ No source active</span>", unsafe_allow_html=True)

    # Stream URL only shown for audio option
    stream_url = ""
    if source_choice == "Live Audio (ffmpeg + Whisper)":
        stream_url = st.text_input(
            "ECB stream URL (HLS .m3u8 or RTSP — not a YouTube page URL)",
            placeholder="https://live.ecb.europa.eu/…  or  yt-dlp -g <youtube_url>",
            key="stream_url_input",
        )
        st.caption(
            "To get the HLS URL from YouTube:  \n"
            "`yt-dlp -g https://youtube.com/live/… | head -1`"
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start Feed", use_container_width=True, key="btn_start_feed"):
            # Reset state
            FEED_FILE.unlink(missing_ok=True)
            st.session_state.feed_pos        = 0
            st.session_state.barometer       = BayesianBarometerState()
            st.session_state.live_segments   = []
            st.session_state.last_segment_ts = None
            news_reader.stop_news_feed()
            stop_pipeline()

            if source_choice == "FinancialJuice (web scrape)":
                news_reader.start_news_feed()
            elif source_choice == "Live Audio (ffmpeg + Whisper)":
                if stream_url:
                    start_pipeline(stream_url)
                else:
                    st.warning("Paste stream HLS URL first.")
            else:  # Simulate
                texts = [s["text"] for s in FALLBACK_CORPUS]
                threading.Thread(
                    target=simulate_feed, args=(texts, 2.0), daemon=True
                ).start()

            st.session_state.autorefresh_on = True
            st.rerun()
    with c2:
        if st.button("■ Stop", use_container_width=True, key="btn_stop_feed"):
            news_reader.stop_news_feed()
            stop_pipeline()
            st.session_state.autorefresh_on = False
            st.rerun()

    st.divider()
    if st.button("🔄 Reload Historical Corpus", use_container_width=True):
        with st.spinner("Fetching…"):
            segs    = load_historical()
            df      = score_corpus(segs)
            qr      = fit_quantile_regressions(df)
            st.session_state.historical_df = df
            st.session_state.historical_qr = qr
            st.session_state.tsne_df       = None
        st.success(f"Loaded {len(df)} segments.")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Barometer",
    "📈 Quantile Timeline",
    "🗺️ t-SNE Map",
    "🎛️ Scenarios",
    "📊 Market Data",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE BAROMETER
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"## <span style='color:{BIG_ORANGE}'>Live Hawkishness Barometer</span>",
        unsafe_allow_html=True,
    )

    bar_state  = st.session_state.barometer
    live_segs  = st.session_state.live_segments
    current_bar = bar_state.posterior_mean
    ci          = bar_state.credible_interval_95

    # Status line
    ts_str = (st.session_state.last_segment_ts or "—")[-8:]
    if st.session_state.autorefresh_on:
        st.markdown(
            f"<span class='tick-badge'>🔄 LIVE — tick #{refresh_count} | "
            f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
            f" | +{new_segments} new this tick</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span class='idle-badge'>⏸ Auto-refresh OFF — enable in sidebar</span>",
            unsafe_allow_html=True,
        )

    # Gauge
    gauge_fig = go.Figure(build_gauge_trace(current_bar, ci))
    gauge_fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=20))
    st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Posterior Mean",     f"{current_bar:.1f}",
              delta=f"{current_bar - 50:+.1f} vs neutral")
    m2.metric("95% CI",             f"[{ci[0]}, {ci[1]}]")
    m3.metric("Segments processed", len(live_segs))
    m4.metric("Tone",               tone_label(bar_state.posterior_mean - 50))

    # Change-point alerts
    if bar_state.alerts:
        st.markdown("### 🚨 Change-Point Alerts")
        for alert in bar_state.alerts[-3:]:
            emoji = "📈" if alert["direction"] == "more hawkish" else "📉"
            st.markdown(
                f'<div class="alert-box">{emoji} <b>{alert["message"]}</b>'
                f'<br>CUSUM statistic: {alert["cusum"]:.1f}</div>',
                unsafe_allow_html=True,
            )

    # Posterior time-series sparkline
    if len(live_segs) >= 2:
        xs   = [s["segment_index"] for s in live_segs]
        obs  = [s["barometer"]     for s in live_segs]
        # Replay posterior means
        _rep = BayesianBarometerState()
        post = [_rep.update(i, y)["posterior_mean"] for i, y in enumerate(obs)]

        spark = go.Figure()
        spark.add_trace(go.Scatter(
            x=xs, y=obs, mode="markers",
            marker=dict(color="#bbb", size=5), name="Segment score",
        ))
        spark.add_trace(go.Scatter(
            x=xs, y=post, mode="lines",
            line=dict(color=BIG_ORANGE, width=2), name="Posterior mean",
        ))
        for level, color, label_text in [
            (65, "red",  "Hawkish"),
            (50, "grey", "Neutral"),
            (35, "blue", "Dovish"),
        ]:
            spark.add_hline(
                y=level, line=dict(color=color, width=1, dash="dot"),
                annotation_text=label_text, annotation_position="right",
            )
        spark.update_layout(
            height=200, margin=dict(l=40, r=60, t=10, b=30),
            yaxis=dict(range=[0, 100]),
            template="plotly_white", showlegend=True,
            legend=dict(orientation="h", y=1.15),
            font=dict(family="Helvetica Neue, Arial", size=10),
        )
        st.plotly_chart(spark, use_container_width=True, key="sparkline")

    # Recent segments
    st.markdown("### Recent Speech Segments")
    if live_segs:
        for seg in reversed(live_segs[-4:]):
            bv = seg["barometer"]
            badge = (
                '<span class="hawkish-badge">Hawkish</span>' if bv >= 65 else
                '<span class="dovish-badge">Dovish</span>'   if bv <= 35 else
                '<span class="neutral-badge">Neutral</span>'
            )
            drivers  = top_terms(seg["text"], n=3)
            drv_str  = " · ".join(
                f"{t} ({'+' if w > 0 else ''}{w:.1f})" for t, w in drivers
            )
            st.markdown(
                f"**#{seg['segment_index']}** {badge} — Barometer: **{bv:.1f}**<br>"
                f"*\"{seg['text'][:120]}…\"*<br>"
                f"<small>Top drivers: {drv_str}</small>",
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info(
            "No live segments yet.  \n"
            "Click **🧪 Simulate feed** in the sidebar — it will auto-enable "
            "live mode and you will see the barometer update every ~3 seconds."
        )

    # Manual text paste
    with st.expander("📝 Manual input — paste a Lagarde quote"):
        manual_text = st.text_area("Paste quote:", height=80, key="manual_quote")
        if st.button("Score & update", key="btn_manual"):
            txt = manual_text.strip()
            if txt:
                raw = score_segment(txt)
                bv  = to_barometer(raw)
                idx = len(live_segs)
                res = bar_state.update(idx, bv)
                st.session_state.live_segments.append({
                    "meeting_date": "2026-06-11",
                    "segment_index": idx, "speaker": "Lagarde",
                    "text": txt, "barometer": bv, "raw_score": raw, "result": res,
                })
                st.session_state.last_segment_ts = datetime.now(timezone.utc).isoformat()
                st.success(f"Updated: {bv:.1f} ({tone_label(raw)})")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QUANTILE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"## <span style='color:{BIG_ORANGE}'>Quantile Regression: Tone Over Time</span>",
        unsafe_allow_html=True,
    )
    st.caption("Q10/Q25/Q50/Q75/Q90 hawkishness trajectories across speech segments.")

    meeting_opts     = ["pooled"] + sorted(hist_df["meeting_date"].unique().tolist())
    selected_meeting = st.selectbox("Select meeting:", meeting_opts, index=0)
    fig_fan = build_fan_chart(hist_df, hist_qr, selected_meeting)
    st.plotly_chart(fig_fan, use_container_width=True, key="fan_hist")

    # Live overlay (shown once >= 4 live segments)
    live_segs = st.session_state.live_segments
    if len(live_segs) >= 4:
        live_df = pd.DataFrame([{
            "meeting_date":  s["meeting_date"],
            "segment_index": s["segment_index"],
            "text":          s["text"],
            "barometer":     s["barometer"],
            "raw_score":     s["raw_score"],
            "bund_reaction": "",
            "speaker":       "Lagarde",
            "word_count":    len(s["text"].split()),
        } for s in live_segs])
        live_qr = fit_quantile_regressions(live_df)
        if "2026-06-11" in live_qr:
            st.markdown("### 🔴 Live: June 11, 2026")
            fig_live = build_fan_chart(live_df, live_qr, "2026-06-11")
            st.plotly_chart(fig_live, use_container_width=True, key="fan_live")

    # Shift attribution table
    st.markdown("### Top Tone Shift Drivers (Historical)")
    shifts = top_shift_drivers(hist_df)
    if shifts:
        rows = [{
            "Meeting":    s["meeting_date"],
            "Seg":        s["segment_index"],
            "Δ Barometer": f"{s['delta']:+.1f}",
            "Direction":  s["direction"],
            "Drivers":    ", ".join(t for t, _ in s["drivers"]),
            "Snippet":    s["text_snippet"][:80] + "…",
        } for s in shifts[:8]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Meeting Summary Statistics"):
        st.dataframe(summary_table(hist_df), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — t-SNE MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f"## <span style='color:{BIG_ORANGE}'>t-SNE: Phrases × FGBL Bund Reaction</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Semantic embedding of Lagarde phrases coloured by expected Bund direction. "
        "Hover for full phrase. K-means cluster labels (k=5)."
    )

    if st.session_state.tsne_df is None:
        with st.spinner("Running t-SNE (TF-IDF → PCA → TSNE)…"):
            tsne_df, fig_tsne = run_tsne_pipeline(hist_df, n_clusters=5, seed=42)
            st.session_state.tsne_df  = tsne_df
            st.session_state.tsne_fig = fig_tsne

    st.plotly_chart(st.session_state.tsne_fig, use_container_width=True, key="tsne_plot")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### Cluster Summary")
        clust = cluster_summary(st.session_state.tsne_df)
        st.dataframe(
            clust.drop(columns=["Representative Phrase"]),
            use_container_width=True, hide_index=True,
        )
    with cb:
        st.markdown("### Representative Phrases")
        for _, row in clust.iterrows():
            c = (
                "#c0392b" if row["Dominant Reaction"] in ("lower", "neutral_lower")
                else "#2980b9" if row["Dominant Reaction"] in ("higher", "sharply higher")
                else "#7f8c8d"
            )
            st.markdown(
                f"**{row['Cluster']}** — "
                f"<span style='background:{c};color:white;padding:1px 6px;"
                f"border-radius:10px;font-size:0.75rem'>{row['Dominant Reaction']}</span><br>"
                f"<small>{row['Representative Phrase']}</small>",
                unsafe_allow_html=True,
            )

    if st.button("♻️ Re-run t-SNE", key="btn_tsne"):
        st.session_state.tsne_df = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        f"## <span style='color:{BIG_ORANGE}'>Scenario Analysis: Topic × Framing → FGBL</span>",
        unsafe_allow_html=True,
    )
    st.caption("Select a framing for each topic to derive a composite Bund futures estimate.")

    col_sel, col_res = st.columns([1.6, 1])
    user_sel: dict[str, str] = {}

    with col_sel:
        st.markdown("### Configure Scenario")
        for topic, framings in TOPIC_FRAMINGS.items():
            user_sel[topic] = st.selectbox(
                f"**{topic}**", framings,
                index=min(1, len(framings) - 1),
                key=f"sc_{topic}",
            )

    with col_res:
        st.markdown("### Composite FGBL Estimate")
        bps = compute_composite_bps(user_sel)
        bar = compute_barometer_from_selections(user_sel)
        direction = ("📈 Rally" if bps > 10 else "📉 Sell-off" if bps < -10 else "➡️ Neutral")
        st.metric("Expected FGBL Move", f"{bps:+.0f} bps", delta=direction)
        st.metric("Barometer",          f"{bar:.1f} / 100")
        st.metric("Tone",               tone_label(bar - 50))
        st.markdown("### Named Scenarios")
        named = named_scenario_table()
        st.dataframe(
            named[["Scenario", "Composite FGBL (bps)", "Barometer", "FGBL Interpretation"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("### Scenario Heatmap")
    fig_heat = build_heatmap(user_sel)
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARKET DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        f"## <span style='color:{BIG_ORANGE}'>Bund Futures Market Data</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Live price: TwelveData WS → EODHD WS → Stooq 5 s poll (cascade). "
        "Auto-refreshes with the global 3 s ticker when Live mode is on."
    )

    price_data = get_current_price()
    price_val  = price_data.get("price")
    change_bps = price_data.get("change_bps")
    source     = price_data.get("source", "none")
    ts         = price_data.get("timestamp") or "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FGBL Last",    f"{price_val:.2f}" if price_val else "—")
    c2.metric("Change (bps)", f"{change_bps:+.1f}" if change_bps is not None else "—")
    c3.metric("Source",       source)
    c4.metric("Updated (UTC)", ts[-8:] if len(ts) >= 8 else "—")

    # Barometer vs price overlay
    live_segs = st.session_state.live_segments
    if live_segs and price_val:
        bar_ts = [s["barometer"] for s in live_segs]
        xs     = list(range(len(live_segs)))
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(
            x=xs, y=bar_ts, mode="lines+markers",
            line=dict(color=BIG_ORANGE, width=2), marker=dict(size=4),
            name="Barometer", yaxis="y1",
        ))
        fig_ov.update_layout(
            template="plotly_white",
            yaxis=dict(title="Barometer [0–100]", range=[0, 100]),
            legend=dict(orientation="h"),
            margin=dict(l=50, r=50, t=30, b=40), height=240,
            font=dict(family="Helvetica Neue, Arial", size=10),
        )
        st.plotly_chart(fig_ov, use_container_width=True, key="mkt_overlay")

    # Manual price entry
    with st.expander("Manual price entry (Bloomberg open next to this app)"):
        cp, cb2 = st.columns([3, 1])
        with cp:
            manual_price = st.number_input(
                "FGBL price:", min_value=100.0, max_value=160.0,
                step=0.01, value=float(price_val) if price_val else 133.00,
                key="manual_price",
            )
        with cb2:
            if st.button("Update", use_container_width=True, key="btn_price"):
                manual_update(manual_price)
                st.success(f"Price set: {manual_price:.2f}")
                st.rerun()

    # Historical Bund yield (ECB SDW)
    st.markdown("### German 10Y Bund Yield — ECB SDW (6 months)")
    with st.spinner("Fetching ECB yield data…"):
        yield_hist = fetch_ecb_bund_history(n_days=90)
    if yield_hist:
        ydf = pd.DataFrame(yield_hist)
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(
            x=ydf["date"], y=ydf["yield_pct"],
            mode="lines+markers",
            line=dict(color=BIG_ORANGE, width=2), marker=dict(size=3),
            name="DE 10Y (%)",
        ))
        fig_y.add_vline(
            x="2026-06-11",
            line=dict(color="red", dash="dash", width=1.5),
            annotation_text="ECB Jun 11", annotation_position="top left",
        )
        fig_y.update_layout(
            template="plotly_white",
            xaxis_title="Date", yaxis_title="Yield (%)",
            font=dict(family="Helvetica Neue, Arial", size=11),
            margin=dict(l=50, r=20, t=30, b=40), height=300,
        )
        st.plotly_chart(fig_y, use_container_width=True, key="ecb_yield")
    else:
        st.warning("Could not fetch ECB yield data.")
