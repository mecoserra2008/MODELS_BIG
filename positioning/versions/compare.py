"""Join layer: run every composite version on the same inputs and compare them.

    python -m positioning.versions.compare     # -> results_positioning_method/ figures + JSON

Registry + `get_composite(version)` selector. Production composite (V0) is included as the
baseline reference; it is NOT modified anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import base
from .base import VersionResult, responsiveness
from . import (v1_kaiser_pca, v2_cftc_bestpractice, v3_ewma_52w, v4_curve, v5_kalman,
               v6_roofing, v7_breadth, v8_changepoint, v9_composite, v10_kalman_factor)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_positioning_method"
OUT.mkdir(parents=True, exist_ok=True)
ORANGE = "#FF8200"; GREY = "#3C3C3C"; RED = "#C00000"; BLUE = "#1F4E79"; GREEN = "#2E7D32"
PURPLE = "#7030A0"; TEAL = "#0F9D9D"

BUILDERS = {"v1": v1_kaiser_pca.build, "v2": v2_cftc_bestpractice.build,
            "v3": v3_ewma_52w.build, "v4": v4_curve.build, "v5": v5_kalman.build,
            "v6": v6_roofing.build, "v7": v7_breadth.build, "v8": v8_changepoint.build,
            "v9": v9_composite.build, "v10": v10_kalman_factor.build}


def _v0(inp) -> VersionResult:
    return VersionResult(name="V0 Production (2x 3y-z + PCA)",
                         description="Current live composite (rolling 3y-z of features -> PCA -> "
                                     "rolling 3y-z of PC1).",
                         composite=inp.prod_composite.rename("v0"), bounded=False,
                         meta={"note": "baseline; unchanged"})


def build_all(force: bool = False):
    inp = base.load_inputs(force=force)
    results = {"v0": _v0(inp)}
    for k, fn in BUILDERS.items():
        try:
            results[k] = fn(inp)
        except Exception as e:  # pragma: no cover
            print(f"  {k} FAILED: {type(e).__name__}: {e}")
    return inp, results


def get_composite(version: str = "v0", force: bool = False) -> pd.Series:
    """Selector API: return one version's composite series."""
    inp = base.load_inputs(force=force)
    if version == "v0":
        return inp.prod_composite
    return BUILDERS[version](inp).composite


# --------------------------------------------------------------------------- #
def _fig(inp, results):
    v0 = results["v0"].composite.dropna()
    # -- panel 1: unbounded z-family (prod, V3) on left; V1 (fixed-scale) on right --
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11))
    ax = axes[0]
    ax.plot(v0.index, v0.values, color=GREY, lw=1.6, label=results["v0"].name)
    v3 = results["v3"].composite.dropna()
    ax.plot(v3.index, v3.values, color=ORANGE, lw=1.2, label=results["v3"].name)
    for k in (1.5, -1.5):
        ax.axhline(k, color=GREY, ls="--", lw=0.6)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.set_ylabel("z (crowd $\\pm$1.5)")
    axr = ax.twinx()
    v1 = results["v1"].composite.dropna()
    axr.plot(v1.index, v1.values, color=PURPLE, lw=1.0, alpha=0.8)
    axr.set_ylabel("V1 Kaiser PC1 (fixed pre-2020 scale)", color=PURPLE)
    ax.set_title("Duration-crowding composites: production (slow) vs single-pass / Kaiser (fast)")
    ax.legend(loc="upper left", fontsize=8)
    ax.plot([], [], color=PURPLE, lw=1.0, label=results["v1"].name)  # legend proxy
    ax.legend(loc="upper left", fontsize=8)

    # -- panel 2: bounded V2 (COT-index) --
    ax = axes[1]
    v2 = results["v2"].composite.dropna()
    ax.plot(v2.index, v2.values, color=BLUE, lw=1.3, label=results["v2"].name)
    st = results["v2"].meta.get("street_z")
    ax.axhline(80, color=RED, ls="--", lw=0.7, label="crowd >80")
    ax.axhline(20, color=GREEN, ls="--", lw=0.7)
    ax.set_ylim(0, 100); ax.set_ylabel("0-100"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("V2 CFTC/Street best-practice: DV01-weighted COT-index (bounded)")

    # -- panel 3: curve (V4) book tilt vs momentum --
    ax = axes[2]
    book = inp.buyside_front_long().dropna()
    ax.plot(book.index, book.values, color=GREY, lw=1.1, label="raw book tilt ($mm/bp)")
    ax.axhline(0, color=GREY, lw=0.5); ax.set_ylabel("$mm/bp", color=GREY)
    axr = ax.twinx()
    mom = results["v4"].curve.dropna()
    axr.plot(mom.index, mom.values, color=RED, lw=1.2)
    for k in (1.5, -1.5):
        axr.axhline(k, color=RED, ls="--", lw=0.5)
    axr.set_ylabel("momentum z (+ = steepener)", color=RED)
    ax.set_title("V4 Curve: BOOK tilt (raw, $\\approx$flat) vs MOMENTUM (fast z, steepener-dir)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_versions_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(force: bool = False):
    inp, results = build_all(force)
    v0_wk = responsiveness(results["v0"].composite, False)["weeks_since_flag"]

    rows = []
    for k in ("v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"):
        r = results[k]
        resp = responsiveness(r.curve if k == "v4" else r.composite, r.bounded)
        lead = None if (resp["weeks_since_flag"] is None or v0_wk is None) \
            else resp["weeks_since_flag"] - v0_wk
        rows.append({"version": k, "name": r.name, "bounded": r.bounded,
                     "latest": None if resp["latest"] is None else round(resp["latest"], 3),
                     "first_flag": resp["first_flag"],
                     "lead_weeks_vs_prod": lead,
                     "pct_time_extreme": None if resp["pct_time_extreme"] is None
                     else round(resp["pct_time_extreme"], 3),
                     "note": r.meta.get("note", "")})
    tbl = pd.DataFrame(rows)

    _fig(inp, results)
    out = {"as_of": str(inp.report_dates.max().date()),
           "responsiveness": rows,
           "v1_meta": {k: results["v1"].meta.get(k) for k in
                       ("n_pcs_kaiser", "eigenvalues", "explained_variance", "features_used")},
           "v2_meta": {k: results["v2"].meta.get(k) for k in
                       ("cot_52_latest", "cot_156_latest", "z_52_latest", "oi_z_latest",
                        "flow_z_latest", "weights")},
           "v4_meta": {k: results["v4"].meta.get(k) for k in
                       ("book_tilt", "mom_latest", "pca_pc2_curve", "cot_front_long")}}
    (OUT / "versions_compare.json").write_text(json.dumps(out, indent=2, default=str))
    print(tbl.to_string(index=False))
    print(f"\nWrote fig_versions_overlay.png + versions_compare.json to {OUT}")
    return out


if __name__ == "__main__":
    main()
