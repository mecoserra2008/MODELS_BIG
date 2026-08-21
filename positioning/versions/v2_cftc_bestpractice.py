"""V2 — CFTC / sell-side / asset-manager BEST-PRACTICE positioning composite.

This version reproduces the family of readings that the cited industry survey
attributes to CFTC commentary, sell-side rates strategists and asset-manager
positioning desks. Everything is computed from the aggregate buy-side (lev+AM)
net DV01 level (`inp.buyside_level()`, $mm/bp).

Formulas
--------
* COT Index — min-max stochastic (Williams / Barchart / COT-Unchained):
      COT_w(t) = 100 * (x_t - min_{w} x) / (max_{w} x - min_{w} x)
  reads 0 at the most-net-SHORT and 100 at the most-net-LONG point of the
  trailing window `w`. Classic windows are 52w (1y) and 156w (3y); the primary
  composite blends the two. Bounded to [0,100]; crowd flag at >80.

* z-score standard (COTInsight): trailing 52-week z-score of the net level.

* %-of-open-interest hedging pressure (CFTC %OI / De Roon-Nijman-Veld
  hedging-pressure / US Treasury TBAC): scale the net level by total open
  interest, then z-score — makes the reading invariant to the secular growth
  of the futures market (market-size invariant).

* Flow / change premium (Kang-Rouwenhorst-Tang): z-score of the 13-week change
  in the net level — captures the SPEED of accumulation (flow) rather than the
  STOCK (level).

Composite
---------
PRIMARY = bounded COT-index blend = mean(COT_52, COT_156), range [0,100],
POSITIVE / HIGH = crowded net LONG duration (rising net long pushes x toward the
window max, so the min-max index rises — verified below). crowd flag at >80.

In `meta` we also expose an unbounded 'street_z' = documented weighted average
of the three z-family readings with weights 0.5 / 0.3 / 0.2
(z_52 / oi_z / flow_z), i.e. street_z = 0.5*z_52 + 0.3*oi_z + 0.2*flow_z.

OFR note
--------
`inp.buyside_level()` is ALREADY DV01 / 10y-equivalent weighted (aggregate
lev+AM net DV01, $mm/bp). This COT index is therefore a DV01-weighted COT index,
unlike a classic raw-contract COT index built on contract counts.

Sign convention: POSITIVE = crowded net LONG duration (bounded, [0,100]).
"""
from __future__ import annotations

import pandas as pd

from positioning.versions import base
from positioning.versions import params as P

# Documented default weighting for the unbounded street z-blend (sums to 1.0). The live
# weights come from params (w_z / w_oi / w_flow) and are normalized to sum to 1.
STREET_WEIGHTS = {"z_52": 0.5, "oi_z": 0.3, "flow_z": 0.2}


def _latest(s: pd.Series) -> float:
    """Last finite value of a series (NaN if empty)."""
    d = s.dropna()
    return float(d.iloc[-1]) if len(d) else float("nan")


def _first_flag(s: pd.Series, bounded: bool) -> str | None:
    """When the sub-series first flagged 'crowded' in the current episode."""
    return base.responsiveness(s, bounded)["first_flag"]


def build(inp: base.VersionInputs, params: dict | None = None) -> base.VersionResult:
    p = P.merge("v2", params)
    level = P.resolve_signal(inp, p)   # buy-side net-DV01 signal (categories/signal_mode applied)

    cot_short_w = int(p["cot_short"])
    cot_long_w = int(p["cot_long"])
    z_w = int(p["z_window"])
    flow_w = int(p["flow_window"])
    crowd_thr = float(p["crowd_thr"])

    # street z-blend weights (normalize to sum 1 if the caller's don't).
    w_z, w_oi, w_flow = float(p["w_z"]), float(p["w_oi"]), float(p["w_flow"])
    tot = w_z + w_oi + w_flow
    if tot > 0 and abs(tot - 1.0) > 1e-12:
        w_z, w_oi, w_flow = w_z / tot, w_oi / tot, w_flow / tot

    # --- Classic COT Index, min-max stochastic (bounded 0-100) ------------- #
    # high = most net long in the window -> POSITIVE = crowded long.
    cot_52 = base.cot_index(level, cot_short_w).rename("cot_52")
    cot_156 = base.cot_index(level, cot_long_w).rename("cot_156")
    # clip guards only float rounding at the window extremes (e.g. 100.0000000001).
    cot_blend = (
        pd.concat([cot_52, cot_156], axis=1).mean(axis=1).clip(0.0, 100.0).rename("cot_blend")
    )

    # --- Unbounded street z-family ---------------------------------------- #
    z_52 = base.rolling_z(level, z_w).rename("z_52")                        # COTInsight 52w z
    oi_z = base.rolling_z(base.pct_of_oi(level, inp.oi), z_w).rename("oi_z")  # %OI hedging pressure
    flow_z = base.rolling_z(level.diff(flow_w), z_w).rename("flow_z")       # flow/change z

    # Documented weighted average (weights normalized to sum to 1.0).
    street_z = (w_z * z_52 + w_oi * oi_z + w_flow * flow_z).rename("street_z")

    meta = {
        # latest of each parallel reading
        "cot_52_latest": _latest(cot_52),
        "cot_156_latest": _latest(cot_156),
        "z_52_latest": _latest(z_52),
        "oi_z_latest": _latest(oi_z),
        "flow_z_latest": _latest(flow_z),
        # unbounded street z-blend: both the full series and its latest value
        "street_z": street_z,
        "street_z_latest": _latest(street_z),
        "weights": {"z_52": w_z, "oi_z": w_oi, "flow_z": w_flow},
        "crowd_thr": crowd_thr,
        "windows": {"cot_short": cot_short_w, "cot_long": cot_long_w,
                    "z_window": z_w, "flow_window": flow_w},
        # first-crowd-flag date per sub-series (COT indices bounded>80; z-family |z|>1.5)
        "sub_first_flags": {
            "cot_52": _first_flag(cot_52, True),
            "cot_156": _first_flag(cot_156, True),
            "cot_blend": _first_flag(cot_blend, True),
            "z_52": _first_flag(z_52, False),
            "oi_z": _first_flag(oi_z, False),
            "flow_z": _first_flag(flow_z, False),
            "street_z": _first_flag(street_z, False),
        },
        "note": (
            "inp.buyside_level() is already DV01 / 10y-equivalent weighted "
            "(aggregate lev+AM net DV01, $mm/bp) per the OFR convention, so this "
            "IS a DV01-weighted COT index — not a raw-contract COT index."
        ),
    }

    return base.VersionResult(
        name="V2 CFTC/Street best-practice (COT-index + %OI + flow)",
        description=(
            "Classic COT Index min-max blend (52w & 156w) as the bounded [0,100] "
            "primary reading; unbounded street z-blend (52w z / %OI z / 13w flow z, "
            "weighted 0.5/0.3/0.2) exposed in meta."
        ),
        composite=cot_blend,
        bounded=True,
        meta=meta,
    )


if __name__ == "__main__":
    inp = base.load_inputs()
    res = build(inp)
    comp = res.composite.dropna()

    print(res.name)
    print(f"latest COT-index composite (0-100): {comp.iloc[-1]:.2f}")
    print("responsiveness:", base.responsiveness(res.composite, res.bounded))
    print(f"street_z latest: {res.meta['street_z_latest']:.3f}")
    print(
        "sub-readings latest:",
        {
            "cot_52": round(res.meta["cot_52_latest"], 2),
            "cot_156": round(res.meta["cot_156_latest"], 2),
            "z_52": round(res.meta["z_52_latest"], 3),
            "oi_z": round(res.meta["oi_z_latest"], 3),
            "flow_z": round(res.meta["flow_z_latest"], 3),
        },
    )
    print("first-crowd-flag (composite):",
          base.responsiveness(res.composite, res.bounded)["first_flag"])
