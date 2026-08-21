"""V9 Final richer-deviations composite — strict-gate blend of the survivors.

Builds V6 (roofing), V7 (breadth), V8 (change-point) on the same inputs, standardizes each to a
comparable z, runs each through the STRICT richness gate vs the V0 baseline, and blends ONLY the
components whose *incremental* deviations pass (rank / robust-z / equal average). `gate='report_only'`
blends all components and reports the gate outcomes without filtering. V0's slow 3y-z is preserved as
the context band by consumers (not folded in here).

Honesty: on a short (~8y), autocorrelated weekly sample the strict gate is conservative and may admit
few or zero components. When zero pass, V9 still returns a usable blend of all components but flags
`gate_certified=False` in meta and the note — a richer *timeliness* read, not a gate-certified alpha.

Imports the component builders directly (not via compare) to avoid a circular import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P
from . import richness, transforms
from . import v6_roofing, v7_breadth, v8_changepoint
from .base import VersionInputs, VersionResult

_BUILDERS = {"v6": v6_roofing.build, "v7": v7_breadth.build, "v8": v8_changepoint.build}


def _trailing_rank(s: pd.Series, window: int = 156) -> pd.Series:
    """Causal trailing percentile rank in [-1, 1] (last point's rank within the trailing window)."""
    def _r(w):
        return (np.argsort(np.argsort(w))[-1] + 1) / len(w)
    mp = max(window // 3, 26)
    return (2.0 * s.rolling(window, min_periods=mp).apply(_r, raw=True) - 1.0)


def _blend(zcols: dict[str, pd.Series], method: str) -> pd.Series:
    """Combine component z-series (already POSITIVE=crowded-long) into one composite."""
    if not zcols:
        return pd.Series(dtype=float, name="v9")
    df = pd.DataFrame(zcols)
    if method == "robust_z":
        df = df.apply(lambda c: transforms.robust_mad_z(c.dropna()).reindex(df.index))
    elif method == "rank":
        df = df.apply(lambda c: _trailing_rank(c.dropna()).reindex(df.index))
    # "equal" (and post-transform) -> row mean, skipna
    return df.mean(axis=1, skipna=True).rename("v9")


def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v9", params)
    comps = list(p.get("components") or ["v6", "v7", "v8"])
    horizon = int(p.get("horizon", 8))
    mode = p.get("gate", "strict")
    baseline = inp.prod_composite

    built, zcols, gate_info, survivors = {}, {}, {}, []
    for k in comps:
        try:
            r = _BUILDERS[k](inp, None)            # default config for each component
        except Exception as exc:                    # never let one component break V9
            gate_info[k] = {"pass": False, "reasons": [f"build failed: {exc!r}"]}
            continue
        built[k] = r
        z = richness.to_z(r.composite, r.bounded)   # comparable, sign = crowded-long
        if mode == "report_only":
            zcols[k] = z
            try:
                gate_info[k] = richness.gate(r.composite, baseline, inp.y10_w,
                                             horizon=horizon, bounded=r.bounded)
            except Exception as exc:
                gate_info[k] = {"pass": None, "reasons": [f"gate error: {exc!r}"]}
        else:  # strict
            try:
                g = richness.gate(r.composite, baseline, inp.y10_w, horizon=horizon, bounded=r.bounded)
            except Exception as exc:
                g = {"pass": False, "reasons": [f"gate error: {exc!r}"]}
            gate_info[k] = g
            if g.get("pass"):
                zcols[k] = z
                survivors.append(k)

    gate_certified = bool(survivors) or (mode == "report_only")
    if not zcols:
        # strict gate admitted nothing -> fall back to all components as a context-only blend.
        for k, r in built.items():
            zcols[k] = richness.to_z(r.composite, r.bounded)

    composite = _blend(zcols, p.get("blend", "rank"))

    n_fin = int(composite.notna().sum())
    latest = float(composite.dropna().iloc[-1]) if n_fin else float("nan")
    note = (
        f"Strict-gate blend of {comps}. survivors={survivors or 'NONE'}"
        + ("" if survivors or mode == "report_only" else
           " — no component cleared the strict OOS gate on this ~8y weekly sample; V9 is a "
           "context-only richness blend (gate_certified=False), improving timeliness/turn-density "
           "rather than an established forward-return edge.")
        + f" blend={p.get('blend')}, gate={mode}, horizon={horizon}w. POSITIVE = crowded net long."
    )
    meta = {
        "components": comps,
        "survivors": survivors,
        "gate_mode": mode,
        "gate_certified": gate_certified,
        "blend": p.get("blend"),
        "horizon": horizon,
        "gate": {k: {"pass": gate_info[k].get("pass"),
                     "incremental_ic": gate_info[k].get("incremental_ic"),
                     "surrogate_p": gate_info[k].get("surrogate_p"),
                     "cpcv_ic": gate_info[k].get("cpcv_ic"),
                     "t": gate_info[k].get("t"),
                     "reasons": gate_info[k].get("reasons")} for k in gate_info},
        "latest": latest,
        "n_finite": n_fin,
        "note": note,
    }
    return VersionResult(
        name="V9 Final richer-deviations composite",
        description="Strict-OOS-gated blend of the surviving V6/V7/V8 components (rank/robust-z/equal); "
                    "keeps V0 as the slow context band. Reports which components' incremental "
                    "deviations passed the gate; honest when none do.",
        composite=composite,
        bounded=False,
        meta=meta,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from .base import load_inputs

    _inp = load_inputs()
    for _mode in ("strict", "report_only"):
        _r = build(_inp, {"gate": _mode})
        print(f"[{_mode:11s}] n_finite={_r.meta['n_finite']:4d}  latest={_r.meta['latest']:+.4f}  "
              f"survivors={_r.meta['survivors']}  certified={_r.meta['gate_certified']}")
    print("note:", _r.meta["note"][:160])
