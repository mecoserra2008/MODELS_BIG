"""V6 Roofing oscillator (DSP cycle extraction).

Roof the buy-side net-DV01 level (or each category×tenor cell) into a zero-mean band-pass
oscillator so every sign-change is a candidate turn — the direct fix for the trend-contaminated,
~13-month-slow 3y-z. Selectable output: oscillator | phase (Hilbert Sine/LeadSine + amplitude) |
fisher (sharpened extremes). Optional Christiano–Fitzgerald band-pass cross-check in meta.

Reuses positioning.versions.dsp (roofing/hilbert_phase/fisher) and transforms.apply_norm.
Built from the RAW aggregate/per-cell net-DV01 panel (2018→2026), not the warmup-truncated
production composite, so downstream OOS scoring sees the full sample. POSITIVE = above-trend
long build = crowded net long.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsp
from . import params as P
from . import transforms
from .base import VersionInputs, VersionResult


def _last(s: pd.Series) -> float:
    """Latest finite value of a series, or NaN."""
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _zero_crossings(s: pd.Series) -> int:
    """Count sign changes of a (dropna'd) series — each is a candidate turn."""
    sg = np.sign(s.dropna())
    sg = sg[sg != 0.0]
    return int((sg.diff().abs() > 0).sum())


def _cell_oscillator(inp: VersionInputs, p: dict) -> pd.Series:
    """Roof EACH buy-side cell's net-DV01, then DV01-weight (mean |net-DV01|) the oscillators."""
    cats = p.get("categories") or list(P.CATS)
    cd = inp.contract_dv01
    cols = [c for c in cd.columns if c[0] in cats]
    if not cols:
        return pd.Series(dtype=float, index=cd.index, name="roofing")
    oscs, weights = {}, {}
    for c in cols:
        cell = cd[c].astype(float)
        cell_n = transforms.apply_norm(cell, p["norm_mode"])
        oscs[c] = dsp.roofing(cell_n, int(p["hp_cutoff"]), int(p["ss_period"]))
        weights[c] = float(cell.abs().mean())         # DV01 weight from RAW cell level
    osc_df = pd.DataFrame(oscs)
    w = pd.Series(weights)
    tot = w.sum()
    w = (w / tot) if tot > 0 else pd.Series(1.0 / len(w), index=w.index)
    return osc_df.mul(w, axis=1).sum(axis=1, min_count=1).rename("roofing")


def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v6", params)

    # 1. Source: aggregate buy-side net-DV01 level, upstream-normalized.
    sig = P.resolve_signal(inp, p).astype(float)
    sig = transforms.apply_norm(sig, p["norm_mode"]).rename("signal")

    # 2. Zero-mean band-pass oscillator (roof aggregate, or roof-then-aggregate cells).
    if p.get("cell_level", False):
        osc = _cell_oscillator(inp, p)
    else:
        osc = dsp.roofing(sig, int(p["hp_cutoff"]), int(p["ss_period"]))
    osc = osc.rename("roofing")

    # 3. Headline output.
    out = p.get("output", "oscillator")
    sine_latest = fisher_latest = float("nan")
    ph_meta: dict = {}
    if out == "phase":
        ph = dsp.hilbert_phase(osc, int(p["phase_period"]))
        composite = ph["sine"].rename("v6")           # −1..1; sign = cycle direction (+ = building)
        sine_latest = _last(ph["sine"])
        # Sine/LeadSine crossover: Sine crossing ABOVE LeadSine = turning up (buy).
        diff = (ph["sine"] - ph["leadsine"]).dropna()
        sgn = np.sign(diff)
        cross = sgn.diff().fillna(0.0)
        last_cross = cross[cross != 0.0]
        ph_meta = {
            "sine_latest": sine_latest,
            "leadsine_latest": _last(ph["leadsine"]),
            "amp_latest": _last(ph["amp"]),
            "sine_gt_leadsine": bool(diff.iloc[-1] > 0) if len(diff) else None,
            "last_crossover_date": (str(last_cross.index[-1].date()) if len(last_cross) else None),
            "last_crossover_dir": ("up" if len(last_cross) and last_cross.iloc[-1] > 0
                                   else "down" if len(last_cross) else None),
        }
    elif out == "fisher":
        composite = dsp.fisher(osc, int(p["fisher_window"])).rename("v6")
        fisher_latest = _last(composite)
    else:  # oscillator (default)
        composite = osc.rename("v6")

    # 4. Christiano–Fitzgerald band-pass cross-check (peer-reviewed, full-sample) in meta.
    cf_latest = cf_corr = float("nan")
    if p.get("cf_crosscheck", False):
        try:
            from statsmodels.tsa.filters.cf_filter import cffilter
            cf_cycle, _cf_trend = cffilter(sig.dropna(), low=int(p["ss_period"]),
                                           high=int(p["hp_cutoff"]), drift=True)
            cf_cycle = pd.Series(np.asarray(cf_cycle).ravel(), index=sig.dropna().index, name="cf")
            cf_latest = _last(cf_cycle)
            aln = pd.concat([cf_cycle, osc.rename("osc")], axis=1).dropna()
            cf_corr = float(aln["cf"].corr(aln["osc"])) if len(aln) > 2 else float("nan")
        except Exception as exc:                       # never raise inside build
            cf_latest, cf_corr = float("nan"), float("nan")
            ph_meta.setdefault("cf_error", str(exc))

    zc = _zero_crossings(osc)
    note = (
        f"Ehlers roofing (2-pole HP@{p['hp_cutoff']}w -> SuperSmoother@{p['ss_period']}w) turns "
        f"the trend-contaminated net-DV01 level into a zero-mean oscillator ({zc} zero-crossings "
        f"= candidate turns). output={out}; POSITIVE = above-trend long build (crowded long), "
        f"a sign-flip up = fresh long build starting."
    )
    meta = {
        "output": out,
        "cell_level": bool(p.get("cell_level", False)),
        "norm_mode": p["norm_mode"],
        "osc_latest": _last(osc),
        "sine_latest": sine_latest,
        "fisher_latest": fisher_latest,
        "hp_cutoff": int(p["hp_cutoff"]),
        "ss_period": int(p["ss_period"]),
        "zero_crossings": zc,
        "cf_cycle_latest": cf_latest,
        "cf_corr": cf_corr,
        "n_finite": int(composite.notna().sum()),
        "note": note,
        **ph_meta,
    }

    return VersionResult(
        name="V6 Roofing oscillator (Ehlers band-pass)",
        description="Roofing filter (2-pole high-pass -> SuperSmoother) turns the trend-contaminated "
                    "buy-side net-DV01 level into a causal zero-mean oscillator; output = "
                    "oscillator / Hilbert phase (Sine/LeadSine) / Fisher; CF band-pass cross-check.",
        composite=composite,
        bounded=False,
        curve=osc,
        meta=meta,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    from .base import load_inputs

    inp = load_inputs()
    cases = [
        ("defaults", None),
        ("phase", {"output": "phase"}),
        ("fisher", {"output": "fisher"}),
        ("cell_level", {"cell_level": True}),
        ("fracdiff", {"norm_mode": "fracdiff"}),
    ]
    for label, ov in cases:
        res = build(inp, ov)
        comp = res.composite
        cf = res.meta.get("cf_corr")
        cf_s = f"{cf:+.3f}" if isinstance(cf, float) and np.isfinite(cf) else "  n/a"
        print(f"[{label:10s}] n_finite={int(comp.notna().sum()):4d}  latest={_last(comp):+.4f}  "
              f"curve_zero_crossings={_zero_crossings(res.curve):3d}  cf_corr={cf_s}")
