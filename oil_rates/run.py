"""Driver: run the oil<->rates conditional study, write figures + JSON + Banco BIG one-pager.

    python -m oil_rates.run [--force] [--report]

Outputs -> results_oil_rates/ (house style: fig_*.png @dpi=200, oil_rates_results.json, onepager.pdf).
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

from . import data as data_mod
from . import study as study_mod

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_oil_rates"
RESULTS.mkdir(parents=True, exist_ok=True)

ORANGE = "#FF8200"; GREY = "#3C3C3C"; RED = "#C00000"; GREEN = "#2E7D32"; BLUE = "#1F4E79"


# --------------------------------------------------------------------------- #
def fig_beta_timeline(res, path):
    rb = res["unconditional"]["rolling_beta"].dropna()
    tv = res["unconditional"]["tvp_beta"].reindex(rb.index)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rb.index, rb.values, color=BLUE, lw=1.0, alpha=0.8, label="rolling 126d β")
    ax.plot(tv.index, tv.values, color=ORANGE, lw=1.6, label="Kalman TVP β")
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axhline(res["unconditional"]["daily_wti"]["beta"], color=RED, ls="--", lw=0.8,
               label=f"full-sample β={res['unconditional']['daily_wti']['beta']:+.2f}")
    ax.set_ylabel("bp of Δ10Y per 1% oil move")
    ax.set_title("Oil→rates sensitivity is small and time-varying")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_decomposition(res, path):
    dec = res["decomposition"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[1, 2])
    legs = ["dy10", "dreal", "dbei"]
    labels = ["nominal", "real (TIPS)", "breakeven"]
    betas = [dec[l]["beta"] for l in legs]; ts = [dec[l]["t"] for l in legs]
    cols = [GREY, GREEN, RED]
    axes[0].bar(labels, betas, color=cols, alpha=0.85)
    for i, (b, t) in enumerate(zip(betas, ts)):
        axes[0].annotate(f"{b:+.2f}\n(t={t:.1f})", (i, b), ha="center", va="bottom", fontsize=8)
    axes[0].axhline(0, color=GREY, lw=0.7)
    axes[0].set_ylabel("bp per 1% oil"); axes[0].set_title("Full-sample β by leg (2003+)")
    roll = dec["rolling"].dropna()
    for l, lab, c in zip(legs, labels, cols):
        axes[1].plot(roll.index, roll[l], color=c, lw=1.1, label=lab)
    axes[1].axhline(0, color=GREY, lw=0.7)
    axes[1].set_title("Rolling 126d β: the breakeven channel carries it")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_markov(res, od, path):
    mk = res["markov"]
    p = mk["prob_low_series"].dropna()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(p.index, 0, p.values, color=RED, alpha=0.25,
                    label=f"P(low-β state)  [β={mk['beta_low']:+.2f}, σ={mk['sigma_low']:.1f}bp]")
    ax.plot(p.index, p.values, color=RED, lw=0.6)
    ax.set_ylim(0, 1); ax.set_ylabel("probability")
    ax.set_title(f"Low-sensitivity regime (high-vol/stress): {mk['share_low_state']:.0%} of days — "
                 f"calm-state β={mk['beta_high']:+.2f} (σ={mk['sigma_high']:.1f}bp)")
    ax2 = ax.twinx()
    wti = od.daily["wti_level"].reindex(p.index)
    ax2.plot(wti.index, wti.values, color=GREY, lw=0.8, alpha=0.7)
    ax2.set_ylabel("WTI ($)", color=GREY)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_conditions(res, path):
    mk = res["markov"]["condition_profile"]; it = res["interactions"]
    conds = [c for c in mk if c in it]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # ratio of means low/high state (log scale-ish display as % difference)
    diffs = [100 * (mk[c]["low_beta_mean"] / mk[c]["high_beta_mean"] - 1)
             if mk[c]["high_beta_mean"] not in (0, None) else 0 for c in conds]
    axes[0].barh(conds, diffs, color=[RED if d > 0 else GREEN for d in diffs], alpha=0.8)
    axes[0].axvline(0, color=GREY, lw=0.7)
    axes[0].set_title("Low-β state conditions vs calm state (% difference)")
    axes[0].set_xlabel("% higher in the low-sensitivity state")
    ts = [it[c]["t_interaction"] for c in conds]
    axes[1].barh(conds, ts, color=[RED if abs(t) > 1.96 else GREY for t in ts], alpha=0.8)
    axes[1].axvline(0, color=GREY, lw=0.7)
    for s in (-1.96, 1.96):
        axes[1].axvline(s, color=RED, lw=0.7, ls="--")
    axes[1].set_title("Interaction t-stats: oil × condition → Δ10Y")
    axes[1].set_xlabel("HAC t-stat (|t|>1.96 significant)")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_influence(res, od, path):
    d = od.daily[["oil_ret", "dy10"]].dropna()
    inf = res["influence"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["oil_ret"], d["dy10"], s=6, color=BLUE, alpha=0.25)
    xs = np.linspace(d["oil_ret"].min(), d["oil_ret"].max(), 50)
    ax.plot(xs, inf["beta_ols"] * xs, color=GREY, lw=1.5,
            label=f"OLS β={inf['beta_ols']:+.2f}")
    ax.plot(xs, inf["beta_excl_top1pct_cooks"] * xs, color=ORANGE, lw=1.5, ls="--",
            label=f"excl. top-1% Cook's D β={inf['beta_excl_top1pct_cooks']:+.2f}")
    for r in inf["top_influential"][:6]:
        ax.scatter([r["oil_ret"]], [r["dy10"]], s=45, color=RED, zorder=5)
        ax.annotate(r["date"], (r["oil_ret"], r["dy10"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=RED)
    ax.set_xlabel("WTI log-return (%)"); ax.set_ylabel("Δ US10Y (bp)")
    ax.set_title("The tail dates that skew the fit (supply/panic days)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_quantile(res, path):
    q = res["quantiles"]
    taus = [float(t) for t in q]; betas = [q[t]["beta"] for t in q]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(taus, betas, "o-", color=ORANGE, lw=2)
    ax.axhline(res["unconditional"]["daily_wti"]["beta"], color=GREY, ls="--", lw=0.8,
               label="OLS β")
    ax.set_xlabel("quantile τ of Δ10Y"); ax.set_ylabel("β (bp per 1% oil)")
    ax.set_title("Quantile regression: β across the outcome distribution")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_shock_type(res, path):
    st = res["shock_type"]
    legs = ["dy10", "dreal", "dbei"]
    labels = ["nominal", "real (TIPS)", "breakeven"]
    x = np.arange(len(legs)); w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, suffix, title in (
            (axes[0], "", f"All classified days ({st['sample']['start'][:4]}+)"),
            (axes[1], "_big", "Big-oil days only (|oil| > 2%)")):
        dem = st["demand" + suffix]; sup = st["supply" + suffix]
        for off, blk, col, lab in ((-w / 2, dem, BLUE, "demand days"),
                                   (+w / 2, sup, ORANGE, "supply days")):
            betas = [blk[l]["beta"] for l in legs]
            ax.bar(x + off, betas, w, color=col, alpha=0.85, label=lab)
            for xi, l in zip(x + off, legs):
                b, t = blk[l]["beta"], blk[l]["t"]
                ax.annotate(f"{b:+.2f}\n(t={t:.1f})", (xi, b), ha="center",
                            va="bottom" if b >= 0 else "top", fontsize=8)
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(f"{title}   n={dem['dy10']['n']}/{sup['dy10']['n']}", fontsize=10)
        ax.margins(y=0.30)
    axes[0].set_ylabel("bp per 1% oil")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Sign-identified shocks: only demand-day oil moves carry a significant rate beta",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
def _jsonable(res: dict) -> dict:
    out = json.loads(json.dumps(res, default=lambda o: None))  # strip Series/DataFrames
    # re-add compact versions of key tables
    out["var"]["irf_oil_to_bei"] = res["var"]["irf_oil_to_bei"]["irf"].round(4).tolist() \
        if isinstance(res["var"]["irf_oil_to_bei"], pd.DataFrame) else None
    out["var"]["irf_oil_to_real"] = res["var"]["irf_oil_to_real"]["irf"].round(4).tolist() \
        if isinstance(res["var"]["irf_oil_to_real"], pd.DataFrame) else None
    return out


def _tex(res: dict, od) -> str:
    u = res["unconditional"]["daily_wti"]; um = res["unconditional"]["monthly_wti"]
    ub = res["unconditional"]["daily_brent"]
    dec = res["decomposition"]; big = dec["big_oil_days"]
    mk = res["markov"]; it = res["interactions"]; inf = res["influence"]
    g_bei = next((r for r in res["var"]["granger_oil_to_rates"] if r["caused"] == "dbei"), {})
    g_real = next((r for r in res["var"]["granger_oil_to_rates"] if r["caused"] == "dreal"), {})
    top = inf["top_influential"][0]
    prof = mk["condition_profile"]
    st = res["shock_type"]; std = st["demand"]; sts = st["supply"]; rec = st["recent"]

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
{{\Large\textbf{{\color{{biggrey}}Why Don't Rates Move With Oil?}}}}\\[2pt]
{{\color{{bigorange}}\rule{{\textwidth}}{{2pt}}}}\\[2pt]
{{\small A conditional-sensitivity study: US10Y vs WTI, 1986--2026 daily (FRED, free data).}}
\end{{center}}

\textbf{{\color{{bigorange}}1. The puzzle, quantified}}\\
Both series are I(1); everything below runs on oil log-returns vs bp changes. The full-sample daily
sensitivity is \textbf{{{u['beta']:+.2f}bp per 1\% oil}} (HAC $t$={u['t']:.1f}) — statistically real but
economically tiny: $R^2$ = {u['r2']*100:.1f}\%, so a 5\% oil move maps to $\approx${5*u['beta']:.1f}bp on the 10Y.
Monthly: {um['beta']:+.2f}bp ($t$={um['t']:.1f}); Brent robustness: {ub['beta']:+.2f}bp ($t$={ub['t']:.1f}).

\begin{{figure}}[H]\centering\includegraphics[width=0.98\textwidth]{{fig_beta_timeline.png}}\end{{figure}}

\textbf{{\color{{bigorange}}2. The mechanism: two channels that offset}}\\
Nominal = real + breakeven. Oil's effect runs \emph{{entirely}} through the inflation leg:
oil$\to$breakeven $\beta$={dec['dbei']['beta']:+.2f} ($t$={dec['dbei']['t']:.1f}, $R^2$={dec['dbei']['r2']*100:.1f}\%);
oil$\to$real $\beta$={dec['dreal']['beta']:+.2f} ($t$={dec['dreal']['t']:.1f}, $\approx$0). Granger confirms:
oil$\to$BEI p={g_bei.get('p_value', float('nan')):.3f}, oil$\to$real p={g_real.get('p_value', float('nan')):.2f}.
On big-oil days ($|$ret$|>$3\%, n={big['n']}) the real and breakeven legs move in \textbf{{opposite
directions {big['share_offsetting']*100:.0f}\% of the time}} (corr {big['corr_real_bei']:+.2f}) and the real leg is
\emph{{larger}} on average ($|\Delta$real$|$={big['mean_abs_dreal']:.1f}bp vs $|\Delta$bei$|$={big['mean_abs_dbei']:.1f}bp)
— the growth/risk channel cancels the inflation channel exactly when oil moves most.

\begin{{figure}}[H]\centering\includegraphics[width=0.98\textwidth]{{fig_decomposition.png}}\end{{figure}}

\textbf{{\color{{bigorange}}3. The conditions for low sensitivity (regimes)}}\\
A 2-state Markov-switching regression finds a \textbf{{low-sensitivity state}}
($\beta$={mk['beta_low']:+.2f}, {mk['share_low_state']*100:.0f}\% of days) that is the \emph{{high-stress}} state:
VIX {prof['vix']['low_beta_mean']:.0f} vs {prof['vix']['high_beta_mean']:.0f}, oil vol
{prof['oil_vol']['low_beta_mean']:.1f} vs {prof['oil_vol']['high_beta_mean']:.1f}\%/d, rate vol
{prof['rate_vol']['low_beta_mean']:.1f} vs {prof['rate_vol']['high_beta_mean']:.1f}bp/d (all p$<$0.01);
the calm state carries $\beta$={mk['beta_high']:+.2f}. The only significant interaction is
\textbf{{oil volatility}} ($t$={it['oil_vol']['t_interaction']:.1f}): per-unit sensitivity \emph{{halves}}
as oil vol doubles — the response \textbf{{saturates}}, so exactly when oil moves most (supply shocks,
risk-off), each 1\% carries the least rate signal. In bp terms the two states respond similarly
($\beta\times$typical move $\approx$ {mk['beta_low']*prof['oil_vol']['low_beta_mean']:.1f} vs
{mk['beta_high']*prof['oil_vol']['high_beta_mean']:.1f}bp/day).

\begin{{multicols}}{{2}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_markov.png}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_conditions.png}}\end{{figure}}
\end{{multicols}}

\textbf{{\color{{bigorange}}4. The datapoints that skew the distribution}}\\
The influential tail is a handful of supply/panic days — {top['date']} (oil {top['oil_ret']:+.0f}\%,
rates {top['dy10']:+.0f}bp), the Mar-2020 COVID cluster, 17-Jan-1991 (Gulf War), Nov-2008 — where rates
decoupled or moved \emph{{against}} oil. They \textbf{{drag the unconditional $\beta$ down}}: OLS
{inf['beta_ols']:+.2f} $\to$ {inf['beta_excl_top1pct_cooks']:+.2f} excluding the top-1\% Cook's-D days
(robust Huber {inf['beta_rlm_huber']:+.2f}). Conditional on $|$oil$|>$5\% (n={inf['dist_given_big_oil']['n']}),
the mean rate move is {inf['dist_given_big_oil']['mean_bp']:+.1f}bp — effectively zero.

\begin{{multicols}}{{2}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_influence.png}}\end{{figure}}
\begin{{figure}}[H]\centering\includegraphics[width=\linewidth]{{fig_quantile.png}}\end{{figure}}
\end{{multicols}}

\textbf{{\color{{bigorange}}5. Demand vs supply shocks: sign identification}}\\
Classifying each day by same-day S\&P~500 co-movement (oil and equities together = \textbf{{demand}}
shock, opposite = \textbf{{supply}} shock; {st['sample']['n']} days
{st['sample']['start'][:4]}--{st['sample']['end'][:4]}, {st['share_demand']*100:.0f}\% demand /
{st['share_supply']*100:.0f}\% supply — the free equity history caps the sample at $\sim$10y):
on \textbf{{demand days}} oil$\to$10Y $\beta$={std['dy10']['beta']:+.2f} ($t$={std['dy10']['t']:.1f}),
carried by the breakeven leg ($\beta$={std['dbei']['beta']:+.2f}, $t$={std['dbei']['t']:.1f}; real
$\beta$={std['dreal']['beta']:+.2f}, $t$={std['dreal']['t']:.1f}). On \textbf{{supply days}} \emph{{no}}
leg is significant: nominal $\beta$={sts['dy10']['beta']:+.2f} ($t$={sts['dy10']['t']:.1f}), breakeven
$\beta$={sts['dbei']['beta']:+.2f} ($t$={sts['dbei']['t']:.1f}). The ordering matches the demand-shock
hypothesis and holds on big-oil days ($|$oil$|>$2\%: demand $t$={res['shock_type']['demand_big']['dy10']['t']:.1f}
vs supply $t$={res['shock_type']['supply_big']['dy10']['t']:.1f}) — but honestly stated, the demand--supply
gap in $\beta$ is \emph{{modest}} ({std['dy10']['beta']:+.2f} vs {sts['dy10']['beta']:+.2f}bp per 1\%),
not dramatic: the sharper distinction is significance, not size. Most recent {rec['window_days']}
classified days ({rec['start']}--{rec['end']}): {rec['n_demand']} demand / {rec['n_supply']} supply
$\to$ a \textbf{{{rec['read']}}} tape, so the current oil impulse should be read with the (weaker)
supply-day betas in mind.

\begin{{figure}}[H]\centering\includegraphics[width=0.98\textwidth]{{fig_shock_type.png}}\end{{figure}}

\textbf{{\color{{bigorange}}Bottom line}}\\
Rates ``don't move with oil'' because (i) the pass-through runs only through breakevens and is small
($\approx$0.3bp per 1\%), (ii) in stress/supply-shock regimes the real-rate (growth) leg moves opposite
to the inflation leg and cancels it — and that is precisely when oil moves most, (iii) per-unit
sensitivity saturates with oil volatility, and (iv) the unconditional estimate is further dragged down by
a few identifiable panic days. Oil matters for \emph{{breakeven}} positioning; for nominal duration it is
a second-order input unless the move is demand-led in a calm regime.

\vspace{{4pt}}{{\footnotesize\color{{biggrey}}Sources: FRED (DCOILWTICO, DCOILBRENTEU, DGS10, DFII10,
T10YIE, VIXCLS, DTWEXBGS, DFF, SP500). WTI negative-price day 2020-04-20 excluded from returns.
HAC/Newey-West errors; Markov 2-state ML; Cook's-distance influence; Kilian-style sign identification
for demand/supply days. Reproduce: \texttt{{python -m oil\_rates.run --report}}.}}
\end{{document}}
"""


def build_report(res, od):
    (RESULTS / "onepager.tex").write_text(_tex(res, od))
    r = None
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "onepager.tex"], cwd=RESULTS, capture_output=True, text=True)
    ok = (RESULTS / "onepager.pdf").exists() and r.returncode == 0
    if not ok:
        print("pdflatex tail:\n", r.stdout[-1500:])
    print(("Built " if ok else "FAILED ") + str(RESULTS / "onepager.pdf"))


def main(force=False, report=False):
    print("Building data (FRED) ...")
    od = data_mod.build(force=force)
    print("Running study ...")
    res = study_mod.run_all(od)

    fig_beta_timeline(res, RESULTS / "fig_beta_timeline.png")
    fig_decomposition(res, RESULTS / "fig_decomposition.png")
    fig_markov(res, od, RESULTS / "fig_markov.png")
    fig_conditions(res, RESULTS / "fig_conditions.png")
    fig_influence(res, od, RESULTS / "fig_influence.png")
    fig_quantile(res, RESULTS / "fig_quantile.png")
    fig_shock_type(res, RESULTS / "fig_shock_type.png")

    (RESULTS / "oil_rates_results.json").write_text(
        json.dumps(_jsonable(res), indent=2, default=str))
    print(f"Wrote results + figures to {RESULTS}")
    if report:
        build_report(res, od)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    main(force=args.force, report=args.report)
