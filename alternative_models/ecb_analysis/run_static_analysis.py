"""
Offline runner: loads historical corpus, runs all analysis modules,
and saves publication-quality figures for embedding in the LaTeX report.

Run:
    cd /workspaces/MODELS_BIG/alternative_models/ecb_analysis
    python run_static_analysis.py
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.io as pio

# Make sure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcript_loader import load_historical
from lexicon import score_segment, to_barometer, top_terms
from quantile_model import score_corpus, fit_quantile_regressions, build_fan_chart, summary_table, top_shift_drivers
from tsne_bund import build_tsne_frame, run_tsne_pipeline, cluster_summary
from scenario import build_heatmap, named_scenario_table, SCENARIO_MATRIX, TOPIC_WEIGHTS
from bayesian_barometer import BayesianBarometerState, posterior_time_series, build_barometer_from_history

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

PLOTLY_SCALE = 2.0
PLOTLY_WIDTH = 900
PLOTLY_HEIGHT = 480

BIG_ORANGE = "#FF8200"
BIG_GREY = "#3C3C3C"


def _save_plotly(fig, name: str, width: int = PLOTLY_WIDTH, height: int = PLOTLY_HEIGHT):
    path = FIG_DIR / f"{name}.png"
    try:
        fig.write_image(str(path), width=width, height=height, scale=PLOTLY_SCALE)
        print(f"  Saved: {path}")
    except Exception as exc:
        print(f"  WARNING: Could not save {name}.png via kaleido: {exc}")
        # Fallback: save HTML
        html_path = FIG_DIR / f"{name}.html"
        fig.write_html(str(html_path))
        print(f"  Fallback HTML saved: {html_path}")
    return path


def run_quantile_analysis(df: pd.DataFrame):
    print("\n[1] Quantile regression fan charts...")
    qr = fit_quantile_regressions(df)

    # Pooled fan chart
    fig_pooled = build_fan_chart(df, qr, "pooled")
    _save_plotly(fig_pooled, "quantile_fan_chart_pooled", width=PLOTLY_WIDTH, height=PLOTLY_HEIGHT)

    # Per-meeting fan charts (pick last 2 for report)
    meetings = sorted(df["meeting_date"].unique())
    for m in meetings[-2:]:
        fig_m = build_fan_chart(df, qr, m)
        tag = m.replace("-", "")
        _save_plotly(fig_m, f"quantile_fan_chart_{tag}", width=PLOTLY_WIDTH, height=PLOTLY_HEIGHT)

    # Summary table
    tbl = summary_table(df)
    print("\n  Summary Table:")
    print(tbl.to_string(index=False))

    # Shift attribution
    shifts = top_shift_drivers(df)
    print(f"\n  Top {len(shifts)} tone shifts detected.")
    for sh in shifts[:3]:
        print(f"    [{sh['meeting_date']}] seg {sh['segment_index']}: "
              f"Δ={sh['delta']:+.1f} ({sh['direction']}) | "
              f"drivers: {[t for t, _ in sh['drivers']]}")

    return qr, tbl, shifts


def run_tsne_analysis(df: pd.DataFrame):
    print("\n[2] t-SNE Bund map...")
    tsne_df, fig_tsne = run_tsne_pipeline(df, n_clusters=5, seed=42)
    _save_plotly(fig_tsne, "tsne_bund_map", width=1000, height=560)
    clust = cluster_summary(tsne_df)
    print("\n  Cluster Summary:")
    print(clust[["Cluster", "N Segments", "Mean Barometer", "Dominant Reaction"]].to_string(index=False))
    return tsne_df, clust


def run_scenario_analysis():
    print("\n[3] Scenario analysis heatmap...")
    fig_heat = build_heatmap()
    _save_plotly(fig_heat, "scenario_heatmap", width=820, height=440)
    tbl = named_scenario_table()
    print("\n  Named Scenarios:")
    print(tbl[["Scenario", "Composite FGBL (bps)", "Barometer", "FGBL Interpretation"]].to_string(index=False))
    return tbl


def run_barometer_history(df: pd.DataFrame):
    print("\n[4] Bayesian barometer history...")
    # Score corpus by meeting order
    df_sorted = df.sort_values(["meeting_date", "segment_index"]).reset_index(drop=True)
    scores = df_sorted["barometer"].tolist()

    state = BayesianBarometerState()
    series = []
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        result = state.update(i, row["barometer"])
        series.append({
            "i": i,
            "observation": row["barometer"],
            "posterior_mean": result["posterior_mean"],
            "ci_lo": result["ci_95"][0],
            "ci_hi": result["ci_95"][1],
            "meeting_date": row["meeting_date"],
            "alert": result.get("alert"),
        })

    # Matplotlib static figure for LaTeX
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("white")

    xs = [s["i"] for s in series]
    obs = [s["observation"] for s in series]
    post = [s["posterior_mean"] for s in series]
    ci_lo = [s["ci_lo"] for s in series]
    ci_hi = [s["ci_hi"] for s in series]

    ax1 = axes[0]
    ax1.fill_between(xs, ci_lo, ci_hi, alpha=0.15, color=BIG_ORANGE, label="95% CI")
    ax1.plot(xs, obs, "o", color="#888", ms=4, alpha=0.6, label="Segment score")
    ax1.plot(xs, post, "-", color=BIG_ORANGE, lw=2, label="Posterior mean")
    ax1.axhline(65, color="red", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(35, color="blue", lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(50, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Barometer (0=Dovish, 100=Hawkish)", fontsize=9, color=BIG_GREY)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Bayesian Barometer: Lagarde Tone Posterior (Historical Meetings)", fontsize=11, color=BIG_GREY)
    ax1.tick_params(labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Meeting boundary markers
    prev_date = None
    for s in series:
        if s["meeting_date"] != prev_date:
            ax1.axvline(s["i"], color="#ccc", lw=1.5, ls="-")
            ax1.text(s["i"] + 0.3, 95, s["meeting_date"], fontsize=6, color="#999", rotation=90, va="top")
            prev_date = s["meeting_date"]

    # CUSUM subplot
    ax2 = axes[1]
    cusum_pos = [0.0]
    cusum_neg = [0.0]
    k = 3.0
    for s in series:
        cp = max(0.0, cusum_pos[-1] + (s["observation"] - s["posterior_mean"]) - k)
        cn = max(0.0, cusum_neg[-1] - (s["observation"] - s["posterior_mean"]) - k)
        cusum_pos.append(cp)
        cusum_neg.append(cn)
    ax2.plot(range(len(cusum_pos)), cusum_pos, color="red", lw=1.2, label="CUSUM+ (hawkish shift)")
    ax2.plot(range(len(cusum_neg)), cusum_neg, color="blue", lw=1.2, label="CUSUM− (dovish shift)")
    ax2.axhline(15, color="grey", lw=0.8, ls="--")
    ax2.set_ylabel("CUSUM", fontsize=8, color=BIG_GREY)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.tick_params(labelsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlabel("Cumulative Segment Index", fontsize=9, color=BIG_GREY)

    plt.tight_layout()
    out = FIG_DIR / "barometer_history.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")
    return series


def run_word_attribution(df: pd.DataFrame, shifts: list[dict]):
    print("\n[5] Word attribution chart...")
    # Top 15 hawkish/dovish terms by total contribution across corpus
    from collections import defaultdict
    from lexicon import LEXICON
    contrib: dict[str, float] = defaultdict(float)
    for _, row in df.iterrows():
        for term, weight in top_terms(row["text"], n=10):
            contrib[term] += weight

    sorted_terms = sorted(contrib.items(), key=lambda x: x[1])
    # Show top 8 dovish and top 8 hawkish
    dovish = [t for t in sorted_terms if t[1] < 0][:8]
    hawkish = list(reversed([t for t in sorted_terms if t[1] > 0][-8:]))
    combined = dovish + hawkish

    labels = [t[0] for t in combined]
    values = [t[1] for t in combined]
    colors = [BIG_ORANGE if v > 0 else "#2980b9" for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("Cumulative Lexicon Contribution (corpus-wide)", fontsize=9, color=BIG_GREY)
    ax.set_title("Top Hawkish/Dovish Terms by Corpus-Wide Impact", fontsize=11, color=BIG_GREY)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    hawk_patch = mpatches.Patch(color=BIG_ORANGE, label="Hawkish")
    dove_patch = mpatches.Patch(color="#2980b9", label="Dovish")
    ax.legend(handles=[hawk_patch, dove_patch], fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "word_attribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 60)
    print("ECB Communication Analysis — Static Figure Generator")
    print("=" * 60)

    print("\n[0] Loading historical corpus...")
    segments = load_historical()
    print(f"  Loaded {len(segments)} segments across {len(set(s['meeting_date'] for s in segments))} meetings.")

    df = score_corpus(segments)
    print(f"  Barometer range: [{df['barometer'].min():.1f}, {df['barometer'].max():.1f}]")
    print(f"  Overall mean: {df['barometer'].mean():.1f} | median: {df['barometer'].median():.1f}")

    qr, tbl, shifts = run_quantile_analysis(df)
    run_tsne_analysis(df)
    run_scenario_analysis()
    run_barometer_history(df)
    run_word_attribution(df, shifts)

    # Save scored corpus for the Streamlit app
    df.to_parquet(FIG_DIR / "corpus_scored.parquet", index=False)
    print(f"\n  Corpus saved to {FIG_DIR}/corpus_scored.parquet")

    print("\n" + "=" * 60)
    print("All figures saved to:", FIG_DIR)
    print("Run 'streamlit run ecb_live_app.py' to launch the live app.")
    print("Run 'pdflatex ECB_June_2026_NLP_report.tex' to compile the paper.")
    print("=" * 60)


if __name__ == "__main__":
    main()
