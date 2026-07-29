"""V0 — the live production composite, as a version (baseline).

With default params this returns the exact production composite already on `inp.prod_composite`
(rolling-3y-z features -> PCA fit 2006-19 -> rolling-3y-z of PC1). If `train_end`/`k_entry` are
overridden, it re-runs `score.run_score` on the raw frames so the baseline is tunable too.
Backend agent may extend; the default path must stay byte-identical to production.
"""
from __future__ import annotations

from positioning.core import score as score_mod
from . import base
from . import params as P


def build(inp: base.VersionInputs, params: dict | None = None) -> base.VersionResult:
    p = P.merge("v0", params)
    default = (params is None) or (
        p["train_end"] == "2019-12-31" and abs(float(p["k_entry"]) - 1.5) < 1e-9)
    if default or inp.fut_raw is None:
        comp = inp.prod_composite.rename("v0")
    else:
        res = score_mod.run_score(inp.fut_raw, inp.cash_raw, inp.yields_raw,
                                  train_end=p["train_end"], k_entry=float(p["k_entry"]))
        comp = res.composite.rename("v0")
    return base.VersionResult(
        name="V0 Production (2x 3y-z + PCA)",
        description="Live composite: rolling-3y-z features -> PCA (fit 2006-19) -> rolling-3y-z of PC1.",
        composite=comp, bounded=False,
        meta={"train_end": p["train_end"], "k_entry": p["k_entry"], "note": "production baseline"})


if __name__ == "__main__":
    inp = base.load_inputs()
    r = build(inp)
    print(r.name, "| latest", round(float(r.composite.dropna().iloc[-1]), 3),
          "|", base.responsiveness(r.composite, False))
