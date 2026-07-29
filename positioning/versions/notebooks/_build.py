"""Reproducible builder for the richer-deviations research notebooks.

    python -m positioning.versions.notebooks._build      # regenerates the .ipynb files

Each notebook is authored against the FROZEN versions-package contract (base.load_inputs,
compare.build_all, richness.score/scorecard/frontier, VersionResult, params) so it executes headless
via `jupyter nbconvert --execute`. Cells are defensive (a failing sub-metric prints, never aborts).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Common preamble prepended to every notebook (imports, style, data load, helpers).
PREAMBLE = r'''
import sys, os
# Make the notebook self-sufficient: walk up from cwd to the repo root (dir containing `positioning`).
_root = os.getcwd()
while _root != os.path.dirname(_root) and not os.path.isdir(os.path.join(_root, "positioning")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import warnings; warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    def display(x): print(x)

plt.rcParams.update({
    "figure.figsize": (11, 4.2), "figure.dpi": 110, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})
ORANGE="#FF8200"; GREY="#3C3C3C"; BLUE="#1F4E79"; RED="#C00000"; GREEN="#2E7D32"; PURPLE="#7030A0"; TEAL="#0F9D9D"

from positioning.versions import base, compare, richness, dsp, transforms
from positioning.versions import params as P
from positioning.versions.base import responsiveness

def show(df, n=None):
    """Render a DataFrame/dict, truncated."""
    try:
        display(df if n is None else df.head(n))
    except Exception:
        print(df)

def zline(ax, bounded=False):
    """Draw crowd bands for a composite axis."""
    if bounded:
        ax.axhline(80, color=RED, ls="--", lw=0.7); ax.axhline(20, color=GREEN, ls="--", lw=0.7)
    else:
        for k in (1.5, -1.5): ax.axhline(k, color=GREY, ls="--", lw=0.6)
        ax.axhline(0, color=GREY, lw=0.4)

def crossings(z, level=0.0):
    """Count sign changes of (z-level): a proxy for 'deviations per unit time'."""
    s = z.dropna().astype(float) - level
    return int((np.sign(s).diff().abs() > 0).sum())

print("loaded versions package — building shared inputs (cache-first)…")
inp = base.load_inputs()
y10 = inp.y10_w
print(f"inputs: {len(inp.report_dates)} weekly dates {inp.report_dates.min().date()}..{inp.report_dates.max().date()}")
'''


def nb(*cells) -> nbf.NotebookNode:
    n = new_notebook()
    n.cells = list(cells)
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
    return n


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def preamble_cell():
    return code(PREAMBLE)


# --------------------------------------------------------------------------- #
def notebook_00():
    return nb(
        md(r"""
# 00 · Richness backbone — the objective, the rubric, and where V0–V5 sit

**Goal (user):** *richer signals — more **informative** deviations (threshold crossings / turning
points / extremes) per unit time — without degenerating into noise.*

This notebook builds the **measurement backbone**. It scores the existing versions V0–V5 on a
richness rubric and a **strict out-of-sample gate**, establishing the current frontier that the new
signals (V6 roofing, V7 breadth, V8 change-point, V9 blend) must beat.

**The rubric** (`positioning.versions.richness`), synthesised from the research:
- *Richness*: crossings/yr at 0/±1/±2σ, turning-points/yr vs the **Bienaymé** random benchmark
  (`E[p]=⅔(n−2)`), **Rice's-formula** crossing budget.
- *Structure*: OU half-life, spectral entropy, effective sample size `n_eff`.
- *Information*: Spearman **IC** vs forward Δ10Y + Newey–West t; mutual information.
- *Strict gate*: the **incremental** deviations (ones a candidate has that V0 lacks) must carry IC
  that (a) is >0, (b) beats an **AR(1)/IAAFT surrogate** (p<0.05), (c) survives **CPCV purge+embargo**,
  (d) clears **t>3** / Deflated-Sharpe. Richness that fails this is labelled *manufactured noise*.

*Refs: Rice 1944; Bienaymé/Kendall turning-point test; López de Prado (Deflated Sharpe, PBO, CPCV);
Grinold–Kahn (IR=IC·√breadth).*
"""),
        preamble_cell(),
        md("## Build every version on the same inputs"),
        code(r"""
inp, results = compare.build_all()
keys = ["v0","v1","v2","v3","v4","v5"]
for k in keys:
    r = results[k]; c = r.composite.dropna()
    print(f"{k:3s}  n={len(c):3d}  latest={(c.iloc[-1] if len(c) else float('nan')):+8.3f}  {r.name}")
"""),
        md("## The richness scorecard (V0 = baseline for the incremental-IC gate)"),
        code(r"""
sc = richness.scorecard({k: results[k] for k in keys}, inp.y10_w, baseline_key="v0")
show(sc.round(3))
"""),
        md("""
### Reading the scorecard
- **crossings/yr, turns/yr** ↑ = richer (more deviations). Compare against **V0** (very smooth/slow).
- **bienayme_z** near/above ~0 means turns are no more than random — richness alone is cheap.
- **incr_ic / surrogate_p / cpcv_ic / gate_pass** = the honesty columns. A version can be far richer
  than V0 yet **fail** the gate — that is the signal we care about.
"""),
        md("## Composites through time — visualising 'deviations per unit time'"),
        code(r"""
fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
for ax, k in zip(axes.ravel(), keys):
    r = results[k]; c = r.composite.dropna()
    col = ORANGE if k=="v0" else BLUE
    ax.plot(c.index, c.values, color=col, lw=1.1)
    zline(ax, r.bounded)
    nx = crossings(richness.to_z(c, r.bounded))
    ax.set_title(f"{k.upper()} · {r.name[:34]}  ·  0-crossings={nx}")
fig.suptitle("V0 (orange) is slow & deviation-poor; the alternatives cross far more often", y=1.01)
fig.tight_layout(); display(fig); plt.close(fig)
"""),
        md("## Efficient frontier — turns/yr vs forward informativeness"),
        code(r"""
try:
    fr = richness.frontier(sc)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in fr.iterrows():
        passed = bool(row.get("gate_pass", False))
        ax.scatter(row["turns_per_yr"], row["hit_rate"],
                   s=140, color=(GREEN if passed else RED), edgecolor="k", zorder=3)
        ax.annotate(str(row["version"]).upper(), (row["turns_per_yr"], row["hit_rate"]),
                    xytext=(6,4), textcoords="offset points", fontsize=9)
    ax.axhline(0.5, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("turning points / yr  (richness →)"); ax.set_ylabel("forward hit-rate  (informativeness →)")
    ax.set_title("green = passes strict OOS gate · red = richness not (yet) informative")
    display(fig); plt.close(fig)
    show(fr.round(3))
except Exception as e:
    print("frontier:", type(e).__name__, e)
"""),
        md("""
## Takeaway
The alternatives are all **richer** than V0 (more crossings/turns per year) — that was already known
from the methodology audit. The open question this project answers: **can we add deviations that are
also *informative* out-of-sample?** The next notebooks build V6/V7/V8, score each addition against
this exact rubric, and assemble only the survivors into V9.
"""),
    )


def notebook_v(num, ver, title, angle_md, variants_code, extra_md=""):
    """Template for a single-version notebook (V6/V7/V8)."""
    cells = [
        md(f"# {num} · {title}\n\n{angle_md}"),
        preamble_cell(),
        md(f"## Build {ver.upper()} (default) and score it against V0"),
        code(fr"""
inp, results = compare.build_all()
r = results["{ver}"]; c = r.composite.dropna()
print(r.name); print(r.description)
print(f"n finite={{len(c)}}  latest={{(c.iloc[-1] if len(c) else float('nan')):+.4f}}  0-crossings={{crossings(richness.to_z(c, r.bounded))}}")
print("meta keys:", list(r.meta)[:20])
"""),
        code(fr"""
# Score the default build vs V0 (strict gate)
s = richness.score(r.composite, inp.y10_w, bounded=r.bounded, baseline=results["v0"].composite, name="{ver}")
def _flat(d, pre=""):
    for k, v in (d or {{}}).items():
        if isinstance(v, dict): _flat(v, pre+str(k)+".")
        else: print(f"  {{pre+str(k):38s}} {{v}}")
for sec in ("richness","structure","gate"):
    print(f"[{{sec}}]"); _flat(s.get(sec))
"""),
        md(f"## {ver.upper()} vs V0 through time"),
        code(fr"""
v0 = results["v0"].composite.dropna()
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.plot(c.index, richness.to_z(c, r.bounded), color=BLUE, lw=1.1, label="{ver.upper()} (z)")
ax.plot(v0.index, richness.to_z(v0, False), color=ORANGE, lw=1.3, label="V0 (z)")
zline(ax); ax.legend(loc="upper left")
ax.set_title(f"{{r.name[:48]}} — 0-crossings {{crossings(richness.to_z(c, r.bounded))}} vs V0 {{crossings(richness.to_z(v0, False))}}")
display(fig); plt.close(fig)
"""),
        md("## Variants"),
        code(variants_code),
    ]
    if extra_md:
        cells.append(md(extra_md))
    return nb(*cells)


def notebook_04_transforms():
    return nb(
        md(r"""
# 04 · Upstream normalization transforms (adaptive-normalization angle)

The net-DV01 **level** is near-integrated; a 3y z-score normalises away the very trend that *is* the
signal. Three transforms (`positioning.versions.transforms`) make it stationary-but-memory-bearing,
comparable through time, and robust to CFTC spikes:
- **fracdiff** — minimal-`d` fractional differencing (López de Prado): stationary while keeping >~long memory.
- **volscale** — divide by trailing EWMA vol (RiskMetrics): a "2σ crowd" means the same in 2008/2020/2022.
- **robust** — median/MAD z: one dislocation can't desensitise the detector.
"""),
        preamble_cell(),
        md("## The buy-side net-DV01 signal, raw vs each transform"),
        code(r"""
p = P.merge("v6", None)                       # default categories/signal_mode
sig = P.resolve_signal(inp, p).dropna()
from statsmodels.tsa.stattools import adfuller
ffd = transforms.min_ffd_d(sig)
print(f"raw ADF p={adfuller(sig.values, maxlag=1, autolag=None)[1]:.3f}  ->  "
      f"min-d={ffd['d']}  ADF p={ffd['adf_p']:.3f}  corr(raw)={ffd['corr']:.2f}")

series = {
    "raw": sig,
    f"fracdiff d={ffd['d']}": ffd["series"].dropna(),
    "volscale": transforms.ewma_vol_scale(sig).dropna(),
    "robust MAD-z": transforms.robust_mad_z(sig).dropna(),
}
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for ax, (name, s) in zip(axes, series.items()):
    ax.plot(s.index, s.values, color=BLUE, lw=1.0)
    ax.axhline(s.mean(), color=GREY, lw=0.5)
    ax.set_title(f"{name}   ·   0-crossings(demeaned)={crossings(s - s.mean())}")
fig.suptitle("Raw level barely crosses its mean; the transforms restore frequent, comparable deviations", y=1.005)
fig.tight_layout(); display(fig); plt.close(fig)
"""),
        md("## Does each transform's extra crossing come with information? (IC vs forward Δ10Y)"),
        code(r"""
rows = []
for name, s in series.items():
    z = richness.to_z(s, False)
    icm = richness.ic_metrics(z, inp.y10_w, horizons=(8,), bounded=False)
    st  = richness.structure_metrics(z)
    rows.append({"transform": name, "0-cross": crossings(z),
                 "half_life_wk": st.get("ou_half_life_wk"),
                 "spec_entropy": st.get("spectral_entropy"),
                 "IC_8w": icm.get("8", {}).get("ic"), "IC_t": icm.get("8", {}).get("t")})
show(pd.DataFrame(rows).round(3))
"""),
        md(r"""
### Reading it
`fracdiff` should add crossings while *keeping* IC (memory-bearing), whereas a transform that only
adds crossings with IC→0 is injecting noise (rising spectral entropy, sub-week half-life). These
transforms are exposed as the shared `norm_mode` knob on V6/V7/V8 — the notebooks above build variants
with `{'norm_mode': 'fracdiff'}` etc.
"""),
    )


def notebook_05_final():
    return nb(
        md(r"""
# 05 · V9 — the final richer-deviations composite (strict-gate blend)

V9 builds V6 (roofing), V7 (breadth), V8 (change-point), runs each through the **strict OOS gate**
vs the V0 baseline, and blends **only the survivors** (rank / robust-z average). `gate='report_only'`
blends all and shows the scorecard for comparison. V0's 3y-z is kept as the slow *context* band.
"""),
        preamble_cell(),
        md("## Build V9 and see which components survived the gate"),
        code(r"""
inp, results = compare.build_all()
r9 = results["v9"]; c9 = r9.composite.dropna()
print(r9.name); print(r9.description)
print("survivors / gate detail (meta):")
for k, v in r9.meta.items():
    print(f"  {k}: {str(v)[:100]}")
"""),
        md("## Full scorecard across V0–V9"),
        code(r"""
allkeys = [k for k in ["v0","v1","v2","v3","v4","v5","v6","v7","v8","v9"] if k in results]
sc = richness.scorecard({k: results[k] for k in allkeys}, inp.y10_w, baseline_key="v0")
show(sc.round(3))
"""),
        md("## V9 vs V0 — richer, and (by construction) only where informative"),
        code(r"""
v0 = results["v0"].composite.dropna()
fig, ax = plt.subplots(figsize=(12.5, 4.6))
ax.plot(c9.index, richness.to_z(c9, r9.bounded), color=PURPLE, lw=1.3, label="V9 final (z)")
ax.plot(v0.index, richness.to_z(v0, False), color=ORANGE, lw=1.4, label="V0 context (z)")
zline(ax); ax.legend(loc="upper left")
ax.set_title(f"V9 0-crossings={crossings(richness.to_z(c9, r9.bounded))} vs V0={crossings(richness.to_z(v0, False))}")
display(fig); plt.close(fig)
"""),
        md("## Final efficient frontier"),
        code(r"""
try:
    fr = richness.frontier(sc)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for _, row in fr.iterrows():
        passed = bool(row.get("gate_pass", False))
        big = str(row["version"]) in ("v9","v0")
        ax.scatter(row["turns_per_yr"], row["hit_rate"], s=(220 if big else 130),
                   color=(GREEN if passed else RED), edgecolor="k", zorder=3)
        ax.annotate(str(row["version"]).upper(), (row["turns_per_yr"], row["hit_rate"]),
                    xytext=(6,4), textcoords="offset points", fontsize=9)
    ax.axhline(0.5, color=GREY, ls="--", lw=0.8)
    ax.set_xlabel("turning points / yr  (richness →)"); ax.set_ylabel("forward hit-rate")
    ax.set_title("V9 aims for the upper-right: many deviations that stay informative")
    display(fig); plt.close(fig)
    show(fr.round(3))
except Exception as e:
    print("frontier:", type(e).__name__, e)
"""),
        md(r"""
## Recommendation & honesty note
The strict gate is deliberately conservative on a short (~8y), autocorrelated weekly sample — several
candidates are expected to fail, and that failure is reported, not hidden. V9 is the blend of whatever
components' **incremental** deviations survived. Where nothing clears the gate at t>3, the honest read
is that the *richer* signals improve **timeliness/turn-density** (a real, usable property for a desk
monitor) even where the marginal forward-return edge is not statistically established on this sample —
and the roofing oscillator / breadth-diffusion remain the best fast **context** reads alongside V0.
"""),
    )


def notebook_06_v10():
    return nb(
        md(r"""
# 06 · V10 — Regime-adaptive Kalman dynamic-factor (Kalman *instead of* PCA)

Deepens **V0** and **V5**: replace V0's `StandardScaler + PCA → PC1 → re-3y-z` with a **Kalman
dynamic factor** over the *same* 7-KPI panel. The filtered latent factor is a "far more informed,
dynamic average" of the KPIs (a dynamic PC1 with proper covariance); the standardized **innovation**
(multivariate NIS) is the **deviation from that informed average**. A **break-of-structure filter**
(CUSUM of recursive residuals + V8 BOCPD) monitors process stability; on a break the model **retrains**
(re-standardizes to the new regime) and **resamples** the new-regime distribution to build **robust
quantile bands with bootstrap CIs** — so "crowded/extreme" is calibrated to the regime's actual
(skewed, fat-tailed) distribution, not a fixed ±1.5σ.

Given the prior result (richer deviations don't predict forward yields here), V10 is judged on
**state-estimation quality** — band **calibration/coverage**, break detection, and **adaptivity vs V0**
— with the strict forward-return gate reported only as an honest secondary.

*Refs: dynamic factor models (Stock–Watson); Kalman NIS consistency; Brown–Durbin–Evans CUSUM;
Peaks-Over-Threshold / GPD; bootstrap CIs.*
"""),
        preamble_cell(),
        code(r"""
from positioning.versions import regime
from positioning.core import score as score_mod
inp, results = compare.build_all()
r10 = results["v10"]; meta = r10.meta
dev = r10.composite.dropna()      # headline: deviation from the informed average (NIS)
fac = r10.curve.dropna()          # filtered latent factor = dynamic PC1 (informed average)
v0  = results["v0"].composite.dropna()
print("DFM converged:", meta.get("dfm_converged"), "| fit:", meta.get("fit_mode_used"), meta.get("fit_scope"))
print("factor vs V0 corr:", meta.get("corr_with_v0"), "| deviation_std:", round(meta.get("deviation_std", float("nan")),2))
print("first_break:", meta.get("first_break"), "| retrained_at:", meta.get("retrained_at"))
"""),
        md("## Deepen V0 — the 7-KPI panel the PCA (and now the Kalman) consumes"),
        code(r"""
fp = score_mod.run_score(inp.fut_raw, inp.cash_raw, inp.yields_raw).feature_panel
print("feature_panel:", fp.shape, "->", list(fp.columns))
show(fp.dropna().tail(4).round(2))
"""),
        md("## 1 · Kalman factor (dynamic PC1) vs V0 PCA composite — is it faster / less lagged?"),
        code(r"""
al = pd.concat([fac.rename("f"), v0.rename("v0")], axis=1).dropna()
lags = range(-12, 13)
cc = [al["f"].shift(k).corr(al["v0"]) for k in lags]
best = list(lags)[int(np.nanargmax(np.abs(cc)))]
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.plot(fac.index, (fac - fac.mean())/ (fac.std() or 1), color=BLUE, lw=1.2, label="V10 Kalman factor (z)")
ax.plot(v0.index, richness.to_z(v0, False), color=ORANGE, lw=1.3, label="V0 PCA composite (z)")
zline(ax); ax.legend(loc="upper left")
ax.set_title(f"Kalman factor vs V0 — corr={al['f'].corr(al['v0']):+.2f}, best lead/lag={best:+d}w (>0 ⇒ factor leads V0)")
display(fig); plt.close(fig)
"""),
        md("## 2 · Deviation from the informed average (multivariate NIS) — deepening V5"),
        code(r"""
r5 = results["v5"].composite.dropna()      # V5 = univariate Kalman deviation
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dev.index, dev.values, color=PURPLE, lw=1.1, label="V10 deviation (multivariate NIS)")
ax.plot(r5.index, richness.to_z(r5, False), color=TEAL, lw=1.0, alpha=0.8, label="V5 deviation (univariate)")
zline(ax); ax.legend(loc="upper left")
ax.set_title("Deviation from the informed average: V10 pools all 7 KPIs (NIS) vs V5's single series")
display(fig); plt.close(fig)
"""),
        md("## 3 · Break-of-structure filter — process stability (rare by design; sanity: 2020/2022)"),
        code(r"""
bp = meta.get("break_prob"); cusum = meta.get("cusum", {}) or {}
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
if hasattr(bp, "dropna"):
    b = bp.dropna(); axes[0].plot(b.index, b.values, color=RED, lw=1.0); axes[0].set_ylim(0, 1)
    top = [str(d.date()) for d in b.iloc[8:].sort_values().index[-5:]]   # skip the run-in edge
    print("top break-prob dates (excl. run-in):", top)
axes[0].set_title(f"BOCPD break probability — first_break={meta.get('first_break')}")
cu = cusum.get("cusum"); hi = cusum.get("boundary_hi"); lo = cusum.get("boundary_lo")
if hasattr(cu, "dropna"):
    axes[1].plot(cu.index, cu.values, color=GREY, lw=1.0, label="CUSUM")
    axes[1].plot(hi.index, hi.values, color=RED, ls="--", lw=0.8)
    axes[1].plot(lo.index, lo.values, color=RED, ls="--", lw=0.8)
    axes[1].legend(loc="upper left")
axes[1].set_title(f"CUSUM of recursive residuals (Brown–Durbin–Evans) — first breach {cusum.get('first_breach')}")
fig.tight_layout(); display(fig); plt.close(fig)
"""),
        md("## 4 · Regime-resampled ROBUST bands (headline) + calibration/coverage"),
        code(r"""
rw = (meta.get("params", {}) or {}).get("regime_window", 104)
rb = meta.get("rolling_bands")
if not isinstance(rb, pd.DataFrame):
    rb = regime.rolling_resampled_bands(dev, window=rw)
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.plot(dev.index, dev.values, color=GREY, lw=0.9, label="deviation")
for col, color, lab in [("q0.95", ORANGE, "5/95"), ("q0.05", ORANGE, None),
                         ("q0.99", RED, "1/99"), ("q0.01", RED, None)]:
    if col in rb.columns:
        ax.plot(rb.index, rb[col].values, color=color, lw=0.9, ls="--", label=lab)
ax.legend(loc="upper left")
ax.set_title("Deviation with rolling regime-resampled quantile bands (GPD-tailed) — bands adapt to the regime")
display(fig); plt.close(fig)

bands = meta.get("resampled_bands", {}) or {}
show(pd.DataFrame([{"quantile": q, "point": v.get("point"), "ci_lo": v.get("lo"), "ci_hi": v.get("hi")}
                   for q, v in bands.items() if isinstance(v, dict)]).round(3))
cov = meta.get("coverage", {}) or {}
show(pd.DataFrame([{"quantile": q, "breach_rate": v.get("breach_rate"), "nominal": v.get("nominal"),
                    "well_calibrated": v.get("well_calibrated")}
                   for q, v in cov.items() if isinstance(v, dict)]).round(3))
print("Calibration: breach_rate should sit near nominal (≈5% beyond the 95% band, ≈1% beyond the 99%).")
"""),
        md("## 5 · Adaptivity vs V0, and the honest forward-return gate"),
        code(r"""
facz = richness.to_z(fac, False)      # standardize the latent factor so ±1.5 flags are comparable to V0
resp10 = responsiveness(facz, False); resp0 = responsiveness(v0, False)
print("factor first crowd-flag:", resp10.get("first_flag"), "| V0 first flag:", resp0.get("first_flag"),
      "| factor weeks-since-flag:", resp10.get("weeks_since_flag"), "vs V0", resp0.get("weeks_since_flag"),
      "(more weeks = flagged earlier = leads)")
s = richness.score(dev, inp.y10_w, bounded=False, baseline=v0, name="v10")
g = s.get("gate") or {}
print("richness gate (honest secondary): pass=%s incr_ic=%s surrogate_p=%s cpcv_ic=%s t=%s"
      % (g.get("pass"), g.get("incremental_ic"), g.get("surrogate_p"), g.get("cpcv_ic"), g.get("t")))
"""),
        md("## Scorecard — V10 alongside V0 / V5 / V8"),
        code(r"""
sc = richness.scorecard({k: results[k] for k in ["v0", "v5", "v8", "v10"]}, inp.y10_w, baseline_key="v0")
show(sc.round(3))
"""),
        md(r"""
## Takeaway
V10 answers the brief: a **Kalman dynamic factor replaces PCA** as a *more informed, dynamic average*
of the KPIs; the **NIS innovation** is the deviation from it; a **break filter** flags the rare
structural shifts; and on a break the model **retrains + resamples** the new regime to produce **robust,
regime-calibrated bands with confidence intervals** rather than a static ±1.5σ. Success is read from
**calibration/coverage + adaptivity vs V0**, not forward-return prediction (reported honestly as a
secondary). The resampled bands are the durable, usable output: crowding thresholds that stay
well-calibrated as the positioning regime changes.
"""),
    )


# --------------------------------------------------------------------------- #
def main():
    v6_variants = r"""
cfgs = [{}, {"output":"phase"}, {"output":"fisher"}, {"cell_level":True}, {"norm_mode":"fracdiff"}]
rows = []
for cfg in cfgs:
    rr = compare.BUILDERS["v6"](inp, cfg)
    cc = rr.composite.dropna()
    s = richness.score(rr.composite, inp.y10_w, bounded=rr.bounded, baseline=results["v0"].composite, name="v6")
    g = s.get("gate") or {}
    rows.append({"cfg": str(cfg)[:34], "n": len(cc), "latest": (cc.iloc[-1] if len(cc) else np.nan),
                 "0-cross": crossings(richness.to_z(cc, rr.bounded)),
                 "incr_ic": (g.get("incremental_ic")), "surr_p": g.get("surrogate_p"),
                 "cpcv_ic": g.get("cpcv_ic"), "t": g.get("t"), "pass": g.get("pass")})
show(pd.DataFrame(rows).round(3))
try:
    cf = compare.BUILDERS["v6"](inp, {}).meta
    print("CF cross-check corr(cf_cycle, oscillator):", cf.get("cf_corr"))
except Exception as e:
    print("cf:", e)
"""
    v7_variants = r"""
cfgs = [{}, {"combine":"rank"}, {"combine":"detoned_pca"}, {"combine":"ica"}, {"orthogonalize":False}]
rows = []
for cfg in cfgs:
    rr = compare.BUILDERS["v7"](inp, cfg)
    cc = rr.composite.dropna()
    s = richness.score(rr.composite, inp.y10_w, bounded=rr.bounded, baseline=results["v0"].composite, name="v7")
    g = s.get("gate") or {}
    rows.append({"cfg": str(cfg)[:30], "n": len(cc), "latest": (cc.iloc[-1] if len(cc) else np.nan),
                 "0-cross": crossings(richness.to_z(cc, rr.bounded)),
                 "BR_eff": rr.meta.get("BR_eff"), "N": rr.meta.get("N"), "rho": rr.meta.get("rho"),
                 "incr_ic": g.get("incremental_ic"), "surr_p": g.get("surrogate_p"), "pass": g.get("pass")})
show(pd.DataFrame(rows).round(3))
print("BR_eff vs N — a real finding: the buy-side cells are OFFSETTING (asset managers long duration "
      "vs leveraged funds short), so the mean cell-z correlation rho is near-zero/negative and "
      "effective breadth is HIGH, not collapsed. The naive 'positions co-move so breadth << N' story "
      "does not hold on this panel — favourable for a breadth composite.")
"""
    v8_variants = r"""
cfgs = [{}, {"output":"changepoint_prob"}, {"output":"regime_prob"}, {"bocpd_obs":"gaussian"}, {"hmm_states":3}]
rows = []
for cfg in cfgs:
    rr = compare.BUILDERS["v8"](inp, cfg)
    cc = rr.composite.dropna()
    rows.append({"cfg": str(cfg)[:30], "n": len(cc), "latest": (cc.iloc[-1] if len(cc) else np.nan),
                 "meta": {k: round(v,3) if isinstance(v,(int,float)) else str(v)[:24]
                          for k,v in list(rr.meta.items())[:5]}})
show(pd.DataFrame(rows))
# BOCPD change-point signal over time. Readout = recent run-length mass P(r_t<=4), NOT P(r_t=0)
# (which is identically the hazard prior and carries no information). First ~8 weeks are the BOCPD
# warm-up edge (the very first point always looks like a 'change') — skip them. Sanity: 2020 COVID.
cp  = compare.BUILDERS["v8"](inp, {"output":"changepoint_prob"}).composite.dropna().iloc[8:]           # t (default)
cpg = compare.BUILDERS["v8"](inp, {"output":"changepoint_prob","bocpd_obs":"gaussian"}).composite.dropna().iloc[8:]
if len(cp):
    fig, ax = plt.subplots(figsize=(12,3.6))
    ax.plot(cp.index,  cp.values,  color=RED,  lw=1.1, label="t obs (default, robust)")
    ax.plot(cpg.index, cpg.values, color=GREY, lw=0.9, label="gaussian obs")
    ax.set_ylim(0,1); ax.legend(loc="upper right")
    ax.set_title("V8 BOCPD run-length-mass P(r_t<=4) — fires at genuine breaks (COVID-2020); 2022 grind muted")
    display(fig); plt.close(fig)
    print("top change-point dates (t obs):", [str(d.date()) for d in cp.sort_values().index[-6:]])
"""

    notebooks = {
        "00_richness_backbone.ipynb": notebook_00(),
        "01_v6_roofing.ipynb": notebook_v(
            "01", "v6", "V6 Roofing oscillator — DSP cycle extraction",
            "**Angle:** level-normalisers never force a zero mean, so the signal hangs at extremes. A "
            "**roofing filter** (2-pole high-pass → SuperSmoother) makes a zero-mean oscillator whose "
            "every sign-change is a candidate turn; **Hilbert phase** leads the turn; **Fisher** sharpens "
            "extremes. *Refs: Ehlers, Predictive Indicators / Cybernetic Analysis.*",
            v6_variants,
            "### Note\n`phase`/`fisher` outputs and `cell_level=True` (roof each cell then aggregate) and "
            "`norm_mode='fracdiff'` are all selectable. Watch the **gate** column: more crossings only "
            "'count' if their incremental IC survives."),
        "02_v7_breadth.ipynb": notebook_v(
            "02", "v7", "V7 Breadth / orthogonalized composite — more independent bets",
            "**Angle:** Grinold IR=TC·IC·√BR — breadth comes only from *independent* bets, and the ~18 "
            "category×tenor cells co-move so effective breadth `BR_eff=N/[1+(N−1)ρ]` collapses. Fix: "
            "**Gram–Schmidt** (level→flow⊥→accel⊥), a **breadth-diffusion** index, a **dispersion** index, "
            "and **detoning/ICA**. *Refs: Grinold–Kahn; Clarke–de Silva–Thorley; López de Prado detoning.*",
            v7_variants,
            "### Note\n`BR_eff ≪ N` quantifies how little genuinely-independent information the raw "
            "cross-section carries — orthogonalisation and detoning are the levers that add real breadth."),
        "03_v8_changepoint.ipynb": notebook_v(
            "03", "v8", "V8 Change-point / surprise overlay — discrete events",
            "**Angle:** add discrete *events*, not variance. **BOCPD** run-length P(changepoint), **NIS "
            "surprise** off the Kalman we already run, **HMM** filtered regime probability. *Refs: "
            "Adams–MacKay 2007 (BOCPD); Kalman NIS consistency; Hamilton 1989 (Markov-switching).*",
            v8_variants,
            "### Note\nThese are **event** signals (mostly non-negative intensity), signed by the source's "
            "deviation so + = crowded-long event. Sanity: BOCPD should light up around 2020 and 2022."),
        "04_transforms.ipynb": notebook_04_transforms(),
        "05_v9_final_composite.ipynb": notebook_05_final(),
        "06_v10_kalman_factor.ipynb": notebook_06_v10(),
    }
    # Optional argv filter: `python -m ..._build 06` writes only matching notebooks (so already-executed
    # notebooks aren't reset to blank). No args -> write all.
    only = sys.argv[1:]
    for fname, notebook in notebooks.items():
        if only and not any(tok in fname for tok in only):
            continue
        path = HERE / fname
        nbf.write(notebook, str(path))
        print(f"wrote {path.relative_to(HERE.parents[3])}  ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    import sys
    main()
