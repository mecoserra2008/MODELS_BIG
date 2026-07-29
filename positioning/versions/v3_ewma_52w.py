"""V3 — EWMA / 52w single-pass composite (the simplest responsiveness fix).

Design goal: the *fast, low-lag* positioning composite. Every other version leans on
multi-window stacks, PCA, or double normalization, all of which add lag. This one is
deliberately the SIMPLEST: a single-pass z-score of the aggregate buy-side net-DV01
level, recency-weighted with an EWMA, blended with a short flow z.

Signals (all built from `inp.buyside_level()`, the aggregate lev+AM net-DV01 stock):
  * z52  = rolling_z(level, 52)        single-pass 52-week level z (NO PCA, NO double norm)
  * zew  = ewma_z(level, hl=26)        EWMA z, half-life 26 weeks — keeps all history but
                                       decays it, so recent weeks dominate -> low lag
  * flow = rolling_z(level.diff(13), 52)   13-week flow (change in stock) z

PRIMARY composite is a level+flow blend of the EWMA z:
    composite = 0.7 * zew + 0.3 * flow
  * 0.7 on the EWMA level z  -> "how crowded is the current stock vs. recent history"
  * 0.3 on the 13w flow z    -> a lead tilt so the composite turns before the level does

Weights favour the level (crowding is fundamentally a stock concept) while letting flow
provide early responsiveness. Unbounded, roughly N(0,1). POSITIVE = crowded net LONG.
"""
from __future__ import annotations

import pandas as pd

from positioning.versions import base
from positioning.versions import params as P
from positioning.versions.base import (
    ewma_z,
    load_inputs,
    responsiveness,
    rolling_z,
)

# Default blend weights (documented above): level (EWMA z) dominant, flow provides the lead
# tilt. Live weights come from params (w_ewma / w_flow), normalized to sum to 1.
W_ZEW = 0.7
W_FLOW = 0.3


def build(inp: base.VersionInputs, params: dict | None = None) -> base.VersionResult:
    p = P.merge("v3", params)
    level = P.resolve_signal(inp, p)   # buy-side net-DV01 signal (categories/signal_mode applied)

    hl = int(p["ewma_hl"])
    z_w = int(p["z_window"])
    flow_w = int(p["flow_window"])
    w_ewma, w_flow = float(p["w_ewma"]), float(p["w_flow"])
    tot = w_ewma + w_flow
    if tot > 0 and abs(tot - 1.0) > 1e-12:
        w_ewma, w_flow = w_ewma / tot, w_flow / tot

    z52 = rolling_z(level, z_w)                 # single-pass level z (no PCA / no double-norm)
    zew = ewma_z(level, hl)                      # EWMA z — recency-weighted
    flow = rolling_z(level.diff(flow_w), z_w)    # flow (change in stock) z

    composite = (w_ewma * zew + w_flow * flow).rename("v3_ewma_52w")

    def _latest(s: pd.Series):
        s = s.dropna()
        return None if s.empty else float(s.iloc[-1])

    meta = {
        "z52_latest": _latest(z52),
        "zew_latest": _latest(zew),
        "flow_latest": _latest(flow),
        "weights": {"zew": w_ewma, "flow": w_flow},
        "windows": {"ewma_hl": hl, "z_window": z_w, "flow_window": flow_w},
        "z52_first_flag": responsiveness(z52, False)["first_flag"],
        "zew_first_flag": responsiveness(zew, False)["first_flag"],
    }

    return base.VersionResult(
        name="V3 EWMA/52w single-pass",
        description="Simplest low-lag composite: 0.7*EWMA-z(level, hl=26) + 0.3*flow-z(13w). "
                    "Single-pass, no PCA, no double normalization.",
        composite=composite,
        bounded=False,
        meta=meta,
    )


if __name__ == "__main__":
    inp = load_inputs()
    res = build(inp)
    comp = res.composite.dropna()

    print(res.name)
    print("description:", res.description)
    print("latest composite:", None if comp.empty else float(comp.iloc[-1]))
    print("meta:", res.meta)
    print("responsiveness:", responsiveness(res.composite, False))
