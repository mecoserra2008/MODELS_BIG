"""
Conditional characteristic models: given that an issuer issues, what are the
odds the bond is callable / floating / has a coupon in [a, b] / long tenor / FX?

Issuer behaviour dominates these (agencies issue mostly callables; a treasury
issues fixed non-callable), so each characteristic uses empirical-Bayes
shrinkage: the issuer's own historical mix, shrunk toward its country|sector
group, shrunk toward the global mix.  Sparse issuers borrow strength; frequent
issuers keep their own signature.  Fully modular -> an unspecified characteristic
is simply left out of the product in predict.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

import config as C

K = 8.0   # empirical-Bayes prior strength (equivalent prior sample size)


def _shrink3(v_i, n_i, v_g, n_g, v_glob):
    """Two-level EB blend: issuer -> group -> global."""
    v_g_b = (n_g * v_g + K * v_glob) / (n_g + K) if n_g > 0 else v_glob
    return (n_i * v_i + K * v_g_b) / (n_i + K) if n_i > 0 else v_g_b


@dataclass
class CharModels:
    events: pd.DataFrame
    group_col: str = "pais_setor"
    # precomputed lookups
    issuer_group: dict = field(default_factory=dict)
    n_issuer: dict = field(default_factory=dict)
    n_group: dict = field(default_factory=dict)
    binary_rate: dict = field(default_factory=dict)   # col -> (issuer_rate, group_rate, global_rate)
    cat_rate: dict = field(default_factory=dict)       # col -> {value -> (issuer,group,global)}
    coupon_vals: dict = field(default_factory=dict)    # 'issuer'/'group'/'global' -> {key: np.array}
    tenor_vals: dict = field(default_factory=dict)

    # ---------- construction ----------
    @classmethod
    def fit(cls, events: pd.DataFrame, group_col: str = "pais_setor") -> "CharModels":
        ev = events.copy()
        if group_col not in ev.columns:
            ev[group_col] = ev["country"].fillna("(unknown)") + " | " + ev["sector"].fillna("(unknown)")
        m = cls(events=ev, group_col=group_col)
        m.issuer_group = ev.groupby("issuer")[group_col].agg(
            lambda s: s.mode().iloc[0] if len(s.mode()) else "(unknown)").to_dict()
        m.n_issuer = ev.groupby("issuer").size().to_dict()
        m.n_group = ev.groupby(group_col).size().to_dict()

        for col in ["callable", "puttable", "sinkable", "convertible", "fx_flag", "early_redeem"]:
            s = pd.to_numeric(ev[col], errors="coerce")
            m.binary_rate[col] = (
                s.groupby(ev["issuer"]).mean().to_dict(),
                s.groupby(ev[group_col]).mean().to_dict(),
                float(s.mean()),
            )
        for col in ["coupon_type", "duration_type"]:
            m.cat_rate[col] = {}
            for val in ev[col].dropna().unique():
                ind = (ev[col] == val).astype(float)
                m.cat_rate[col][val] = (
                    ind.groupby(ev["issuer"]).mean().to_dict(),
                    ind.groupby(ev[group_col]).mean().to_dict(),
                    float(ind.mean()),
                )
        # coupon values (fixed-coupon only) and tenor values, for interval queries
        fx = ev[(ev["coupon_type"] == "fixed") & ev["coupon_rate"].notna()]
        m.coupon_vals = {
            "issuer": {i: g["coupon_rate"].to_numpy() for i, g in fx.groupby("issuer")},
            "group": {i: g["coupon_rate"].to_numpy() for i, g in fx.groupby(group_col)},
            "global": fx["coupon_rate"].to_numpy(),
        }
        tv = ev[ev["tenor_days"].notna()]
        m.tenor_vals = {
            "issuer": {i: g["tenor_days"].to_numpy() for i, g in tv.groupby("issuer")},
            "group": {i: g["tenor_days"].to_numpy() for i, g in tv.groupby(group_col)},
            "global": tv["tenor_days"].to_numpy(),
        }
        return m

    # ---------- accessors ----------
    def _grp(self, issuer):
        return self.issuer_group.get(issuer, "(unknown)")

    def p_binary(self, issuer, col) -> float:
        ir, gr, gl = self.binary_rate[col]
        g = self._grp(issuer)
        return float(_shrink3(ir.get(issuer, gl), self.n_issuer.get(issuer, 0),
                              gr.get(g, gl), self.n_group.get(g, 0), gl))

    def p_category(self, issuer, col, value) -> float:
        if value not in self.cat_rate[col]:
            return 0.0
        ir, gr, gl = self.cat_rate[col][value]
        g = self._grp(issuer)
        return float(_shrink3(ir.get(issuer, gl), self.n_issuer.get(issuer, 0),
                              gr.get(g, gl), self.n_group.get(g, 0), gl))

    def _interval_frac(self, store, issuer, lo, hi) -> float:
        """EB-blended P(value in [lo, hi]) from issuer/group/global sample arrays."""
        g = self._grp(issuer)
        def frac(arr):
            return float(((arr >= lo) & (arr <= hi)).mean()) if arr is not None and len(arr) else np.nan
        gl = frac(store["global"])
        fg = frac(store["group"].get(g)); ng = self.n_group.get(g, 0)
        fi = frac(store["issuer"].get(issuer)); ni = self.n_issuer.get(issuer, 0)
        fg = gl if np.isnan(fg) else fg
        fi = fg if np.isnan(fi) else fi
        return float(_shrink3(fi, ni, fg, ng, gl))

    def p_coupon_interval(self, issuer, lo, hi) -> float:
        """P(fixed coupon rate in [lo, hi] | issuance) = P(fixed) * P(rate in [lo,hi] | fixed)."""
        p_fixed = self.p_category(issuer, "coupon_type", "fixed")
        return p_fixed * self._interval_frac(self.coupon_vals, issuer, lo, hi)

    def p_tenor_interval_days(self, issuer, lo_days, hi_days) -> float:
        return self._interval_frac(self.tenor_vals, issuer, lo_days, hi_days)


if __name__ == "__main__":
    from panel import build_panel
    _, events, _ = build_panel()
    cm = CharModels.fit(events)
    for iss in ["Federal Home Loan Bank System", "Government of Argentina",
                "The Goldman Sachs Group, Inc.", "United States Treasury"]:
        if iss not in cm.n_issuer:
            print(f"\n{iss}: (no in-window events)"); continue
        print(f"\n{iss}  (n={cm.n_issuer[iss]})")
        print(f"  P(callable)      = {cm.p_binary(iss,'callable'):.3f}")
        print(f"  P(fixed coupon)  = {cm.p_category(iss,'coupon_type','fixed'):.3f}")
        print(f"  P(floating)      = {cm.p_category(iss,'coupon_type','floating'):.3f}")
        print(f"  P(coupon 4-5%)   = {cm.p_coupon_interval(iss,4,5):.3f}")
        print(f"  P(tenor <3y)     = {cm.p_tenor_interval_days(iss,0,3*366):.3f}")
        print(f"  P(FX/foreign)    = {cm.p_binary(iss,'fx_flag'):.3f}")
