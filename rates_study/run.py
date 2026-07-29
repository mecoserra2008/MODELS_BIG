"""Driver: build the positioning<->yields study, the US10Y ARIMA forecaster, the figures,
the results JSON, and (optionally) the Banco BIG LaTeX one-pager PDF.

    python -m rates_study.run [--force] [--report]

Outputs land in results_rates_study/ (house style: fig_*.png @dpi=200 + *_results.json + PDF).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import arima as arima_mod
from . import data as data_mod
from . import study as study_mod

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_rates_study"
RESULTS.mkdir(parents=True, exist_ok=True)

ORANGE = "#FF8200"
GREY = "#3C3C3C"
RED = "#C00000"
GREEN = "#2E7D32"
BLUE = "#1F4E79"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_composite_vs_yield(sd, path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    c = sd.composite.dropna()
    ax.plot(c.index, c.values, color=BLUE, lw=1.3, label="Positioning composite z (LHS)")
    ax.axhline(1.5, color=RED, ls="--", lw=0.7); ax.axhline(-1.5, color=RED, ls="--", lw=0.7)
    ax.fill_between(c.index, c.values, 0, where=c.values > 1.5, color=RED, alpha=0.12)
    ax.fill_between(c.index, c.values, 0, where=c.values < -1.5, color=GREEN, alpha=0.12)
    ax.set_ylabel("composite z", color=BLUE); ax.axhline(0, color=GREY, lw=0.5)
    ax2 = ax.twinx()
    y = sd.y10_w.reindex(c.index)
    ax2.plot(y.index, y.values, color=ORANGE, lw=1.5, label="US10Y (RHS, %)")
    ax2.set_ylabel("US10Y (%)", color=ORANGE)
    ax.set_title("CFTC positioning composite vs US 10Y yield")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_lead_lag(study, path):
    ll = pd.DataFrame(study["lead_lag"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [ORANGE if h > 0 else GREY for h in ll["h_weeks"]]
    ax.bar(ll["h_weeks"], ll["corr"], color=colors, width=0.9)
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=RED, lw=0.8, ls="--")
    ax.set_xlabel("horizon h (weeks): >0 = future yield change (positioning leading)")
    ax.set_ylabel("corr(composite, Δy)")
    ax.set_title("Lead–lag: composite vs yield change")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_conditional(study, path):
    cond = study["conditional"]
    Hs = sorted(cond, key=int)
    p = [cond[h]["crowded_long"]["p_selloff"] for h in Hs]
    unc = [cond[h]["uncond_p_up"] for h in Hs]
    ci = [cond[h]["crowded_long"].get("p_selloff_ci", (np.nan, np.nan)) for h in Hs]
    lo = [pp - c[0] for pp, c in zip(p, ci)]; hi = [c[1] - pp for pp, c in zip(p, ci)]
    x = np.arange(len(Hs))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, p, 0.55, color=RED, alpha=0.85, yerr=[lo, hi], capsize=5,
           label="P(selloff | crowded long) ±95% CI")
    ax.plot(x, unc, "o--", color=GREY, label="unconditional P(yield up)")
    ax.axhline(0.5, color=GREY, lw=0.7, ls=":")
    for i, h in enumerate(Hs):
        bp = cond[h]["crowded_long"]["mean_bp"]
        ax.annotate(f"{bp:+.0f}bp\n(n={cond[h]['crowded_long']['n']})",
                    (i, p[i]), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}w" for h in Hs]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(yield up over horizon)")
    ax.set_title("When positioning is crowded long, do yields rise? (with CIs)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_scatter(sd, path, H=8):
    c = sd.composite_pub; y = sd.y10_w_pub
    d = pd.concat([c.rename("z"), ((y.shift(-H) - y) * 100).rename("fwd")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.scatter(d["z"], d["fwd"], s=10, color=BLUE, alpha=0.4)
    b, a = np.polyfit(d["z"], d["fwd"], 1)
    xs = np.linspace(d["z"].min(), d["z"].max(), 50)
    ax.plot(xs, a + b * xs, color=ORANGE, lw=2, label=f"fit: {b:+.1f} bp per z")
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(1.5, color=RED, lw=0.7, ls="--")
    ax.set_xlabel("positioning composite z"); ax.set_ylabel(f"forward {H}w Δ US10Y (bp)")
    ax.set_title(f"Composite vs forward {H}w yield change")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_arima_oos(ar, path):
    wf = ar["walk_forward_daily"]["series"]
    actual, pred = wf["actual"], wf["arima"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(actual.index, actual.values, color=GREY, lw=1.2, label="US10Y actual")
    ax.plot(pred.index, pred.values, color=ORANGE, lw=1.0, alpha=0.9,
            label="ARIMA 1-step OOS")
    m = ar["walk_forward_daily"]["metrics"]
    ax.set_title(f"US10Y ARIMA{tuple(ar['order'])} one-step OOS — "
                 f"RMSE {m['arima']['rmse']:.3f} vs RW {m['rw']['rmse']:.3f} "
                 f"(ratio {ar['walk_forward_daily']['skill_vs_rw_rmse']:.3f})")
    ax.set_ylabel("%"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_arima_live(sd, ar, path, lookback=180):
    lv = ar["live"]
    hist = sd.dgs10_daily.dropna().iloc[-lookback:]
    mean, ci80, ci95 = lv["mean"], lv["ci80"], lv["ci95"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist.index, hist.values, color=GREY, lw=1.3, label="US10Y actual")
    ax.plot(mean.index, mean.values, color=ORANGE, lw=1.8, label="ARIMA forecast")
    ax.fill_between(ci95.index, ci95.iloc[:, 0], ci95.iloc[:, 1], color=ORANGE, alpha=0.12,
                    label="95% CI")
    ax.fill_between(ci80.index, ci80.iloc[:, 0], ci80.iloc[:, 1], color=ORANGE, alpha=0.22,
                    label="80% CI")
    ax.set_title(f"US10Y — {lv['steps']}-day ahead ARIMA forecast (near-random-walk)")
    ax.set_ylabel("%"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# JSON (drop non-serializable Series)
# --------------------------------------------------------------------------- #
def _arima_json(ar: dict) -> dict:
    a = dict(ar)
    wf = dict(a["walk_forward_daily"]); wf.pop("series", None)
    a["walk_forward_daily"] = wf
    lv = dict(a["live"])
    for kk in ("mean", "ci80", "ci95"):
        lv.pop(kk, None)
    a["live"] = lv
    return a


# --------------------------------------------------------------------------- #
# One-pager (Banco BIG LaTeX) — numbers injected live
# --------------------------------------------------------------------------- #
def _tex(study: dict, ar: dict, sd) -> str:
    c = study["conditional"]; reg = study["stats"]["forward_regression"]
    g = study["stats"]["granger_p_by_lag"]
    wf = ar["walk_forward_daily"]; m = wf["metrics"]; br = ar["positioning_bridge_weekly"]
    lv = ar["live"]

    def ci(h, key="p_selloff"):
        cc = c[h]["crowded_long"].get(key + "_ci")
        return f"[{cc[0]:.2f}, {cc[1]:.2f}]" if cc else "n/a"

    gmin = min(v for v in g.values()) if isinstance(g, dict) and g and "error" not in g else float("nan")
    r13 = reg.get("13", {}); r26 = reg.get("26", {})
    latest_z = study["latest"]["composite_z"]; as_of = study["latest"]["as_of"]

    return rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=1.4cm]{{geometry}}
\usepackage{{helvet}}\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage{{graphicx,booktabs,xcolor,fancyhdr,multicol,float}}
\definecolor{{bigorange}}{{RGB}}{{255,130,0}}
\definecolor{{biggrey}}{{RGB}}{{60,60,60}}
\pagestyle{{fancy}}\fancyhf{{}}
\lhead{{\textbf{{\color{{bigorange}}Banco BIG}} \color{{biggrey}}| Rates Research}}
\rhead{{\color{{biggrey}}\today}}
\renewcommand{{\headrulewidth}}{{1pt}}
\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}
\begin{{document}}
\begin{{center}}
{{\Large\textbf{{\color{{biggrey}}US Treasury Positioning \& the 10Y Yield}}}}\\[2pt]
{{\color{{bigorange}}\rule{{\textwidth}}{{2pt}}}}\\[2pt]
{{\small A CFTC-positioning crowding study + a dynamic US10Y ARIMA forecaster — free-data, reproducible.}}
\end{{center}}

\textbf{{\color{{bigorange}}1. What positioning measures \& where we are}}\\
We convert CFTC Traders-in-Financial-Futures net positions (TU/FV/TY/TN/US/UB) into DV01, aggregate by
trader type, z-score, and reduce to a PCA composite crowding index (+ = crowded net \emph{{long}} duration).
As of \textbf{{{as_of}}} the composite is \textbf{{{latest_z:+.2f}}} — a multi-year \emph{{crowded-long}}
extreme, driven by asset managers (long the long end) with dealers mirror-short.

\begin{{figure}}[H]\centering\includegraphics[width=0.98\textwidth]{{fig_composite_vs_yield.png}}\end{{figure}}

\textbf{{\color{{bigorange}}2. Does crowded positioning precede higher yields? (the honest answer)}}\\
Directionally \emph{{yes, weakly}}: at crowded-long extremes ($z>1.5$) the 10Y rose over the next
4/8/13/26w with probability {c['4']['crowded_long']['p_selloff']:.2f}/{c['8']['crowded_long']['p_selloff']:.2f}/{c['13']['crowded_long']['p_selloff']:.2f}/{c['26']['crowded_long']['p_selloff']:.2f}
(vs unconditional {c['4']['uncond_p_up']:.2f}/{c['8']['uncond_p_up']:.2f}/{c['13']['uncond_p_up']:.2f}/{c['26']['uncond_p_up']:.2f}),
mean $\Delta y$ {c['8']['crowded_long']['mean_bp']:+.0f}bp (8w) rising to {c['26']['crowded_long']['mean_bp']:+.0f}bp (26w).
\textbf{{But it is not statistically robust}}: 95\% bootstrap CIs include 0.5 at every horizon
(e.g.\ 13w {ci('13')}), Granger causality is insignificant (min p $\approx$ {gmin:.2f}), and forward-return
regressions are weak (13w $t={r13.get('t_stat',float('nan')):+.2f}$, $R^2={r13.get('r2',0):.3f}$;
26w $t={r26.get('t_stat',float('nan')):+.2f}$). The effect is \textbf{{asymmetric}} — fading crowded
\emph{{longs}} works, fading crowded \emph{{shorts}} does not — and rests on few independent episodes.

\begin{{multicols}}{{2}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_conditional.png}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_scatter.png}}\end{{figure}}
\end{{multicols}}

\textbf{{\color{{bigorange}}3. Dynamic US10Y forecast — simple ARIMA{tuple(ar['order'])}}}\\
DGS10 is I(1) (ADF level p={ar['adf']['level_p']:.2f}, diff p$<$0.01). A recursive walk-forward
ARIMA{tuple(ar['order'])} gives one-step OOS RMSE \textbf{{{m['arima']['rmse']:.3f}}} vs a random walk
{m['rw']['rmse']:.3f} (ratio {wf['skill_vs_rw_rmse']:.3f}; Diebold–Mariano p={wf['dm_arima_vs_rw']['p_value']:.2f})
— i.e.\ \textbf{{statistically indistinguishable from a random walk}}, as expected for yields. Adding the
positioning composite as an exogenous regressor (ARIMAX, weekly) does \textbf{{not}} help out-of-sample
(RMSE ratio {br['rmse_ratio_x_over_plain']:.3f}, DM p={br['dm_exog_vs_noexog']['p_value']:.2f}). Latest
level {lv['last_obs']:.2f}\%; {lv['steps']}-day point forecast {lv['point_1m']:.2f}\% (essentially flat).

\begin{{figure}}[H]\centering\includegraphics[width=0.98\textwidth]{{fig_arima_live.png}}\end{{figure}}

\textbf{{\color{{bigorange}}Bottom line}}\\
Positioning is a strong \emph{{descriptive}} crowding gauge and currently flags a stretched long-duration
market whose modest historical tendency is toward higher yields over 1--6 months — but the predictive
edge is weak and not statistically significant, and neither ARIMA nor positioning beats a random walk for
point-forecasting the 10Y. Use positioning for context and risk framing, not as a standalone yield forecast.

\vspace{{4pt}}{{\footnotesize\color{{biggrey}}Sources: CFTC TFF (gpe5-46if), FRED DGS10/DGS2. Causal
(publication-lagged) composite; block-bootstrap CIs; expanding-window OOS. Reproduce:
\texttt{{python -m rates\_study.run --report}}.}}
\end{{document}}
"""


def build_report(study: dict, ar: dict, sd):
    tex = _tex(study, ar, sd)
    texfile = RESULTS / "onepager.tex"
    texfile.write_text(tex)
    ok = False
    try:
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                "onepager.tex"], cwd=RESULTS, capture_output=True, text=True)
        ok = (RESULTS / "onepager.pdf").exists() and r.returncode == 0
        if not ok:
            print("pdflatex tail:\n", r.stdout[-1500:])
    except FileNotFoundError:
        print("pdflatex not found — wrote onepager.tex only")
    print(("Built " if ok else "Wrote (PDF build failed) ") + str(texfile))
    return ok


def main(force=False, report=False):
    print("Loading data (positioning composite + FRED yields) ...")
    sd = data_mod.load(force=force)
    print("Running positioning<->yields study ...")
    study = study_mod.run_study(sd)
    print("Fitting US10Y ARIMA + positioning bridge (walk-forward) ...")
    ar = arima_mod.run_arima(sd)

    fig_composite_vs_yield(sd, RESULTS / "fig_composite_vs_yield.png")
    fig_lead_lag(study, RESULTS / "fig_lead_lag.png")
    fig_conditional(study, RESULTS / "fig_conditional.png")
    fig_scatter(sd, RESULTS / "fig_scatter.png")
    fig_arima_oos(ar, RESULTS / "fig_arima_oos.png")
    fig_arima_live(sd, ar, RESULTS / "fig_arima_live.png")

    (RESULTS / "rates_study_results.json").write_text(
        json.dumps(study, indent=2, default=str))
    (RESULTS / "us10y_arima_results.json").write_text(
        json.dumps(_arima_json(ar), indent=2, default=str))
    print(f"Wrote results + figures to {RESULTS}")

    if report:
        build_report(study, ar, sd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="force re-download of all data")
    ap.add_argument("--report", action="store_true", help="build the LaTeX one-pager PDF")
    args = ap.parse_args()
    main(force=args.force, report=args.report)
