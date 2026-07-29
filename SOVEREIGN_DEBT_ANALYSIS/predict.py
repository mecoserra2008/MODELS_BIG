"""
Query engine: fuse the timing hazard and the conditional characteristic models
to answer "what is the probability that issuer X issues a bond of type C within
horizon H?".

Timing horizon semantics (weekly hazards rolled forward from the as-of date):
  S(k)               = prod_{j=1..k} (1 - h_j)         survival = P(no issuance in next k weeks)
  P(within k weeks)  = 1 - S(k)
  P(first in [a,b])  = S(a-1) - S(b)                   first issuance lands in that week window
  P(>=1 in [a,b])    = 1 - S(b)/S(a-1)                 recurrent-event variant

Characteristic conditioning multiplies the timing probability by
prod_c P(characteristic c | an issuance) from char_models (EB-shrunk, issuer
specific).  Unspecified characteristics are simply omitted (factor 1).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import config as C
from timing_model import TimingModel
from char_models import CharModels


@dataclass
class IssuanceModel:
    timing: TimingModel
    chars: CharModels
    issuer_static: pd.DataFrame          # indexed by issuer
    event_wk: dict                       # issuer -> sorted np.array of absolute W-MON week idx
    base_start: pd.Timestamp
    asof_wk: int

    # ---------- build ----------
    @classmethod
    def build(cls, timing, chars, issuer_static, events) -> "IssuanceModel":
        base = pd.Timestamp(C.WARMUP_START).to_period("W-MON")
        base_start = base.to_timestamp(how="start")
        asof = pd.Timestamp(C.AS_OF)
        asof_wk = int(asof.to_period("W-MON").ordinal - base.ordinal)
        ev = events.copy()
        ev["wk"] = (ev["issue_date"].dt.to_period("W-MON").astype("int64") - base.ordinal).astype(int)
        event_wk = {i: np.sort(g["wk"].to_numpy()) for i, g in ev.groupby("issuer")}
        st = issuer_static.set_index("issuer")
        return cls(timing=timing, chars=chars, issuer_static=st, event_wk=event_wk,
                   base_start=base_start, asof_wk=asof_wk)

    def issuers(self) -> list:
        return sorted(self.issuer_static.index.tolist())

    # ---------- timing ----------
    def _forward_frame(self, issuer: str, horizon: int) -> pd.DataFrame:
        st = self.issuer_static.loc[issuer]
        ew = self.event_wk.get(issuer, np.array([self.asof_wk]))
        last_evt = int(ew.max()) if len(ew) else self.asof_wk
        j = np.arange(1, horizon + 1)
        w = self.asof_wk + j
        wsl = np.minimum(w - last_evt, 260)
        trail = np.array([int(((ew >= wi - 52) & (ew <= wi - 1)).sum()) for wi in w])
        dates = self.base_start + pd.to_timedelta(w * 7, unit="D")
        df = pd.DataFrame({
            "weeks_since_last": wsl,
            "issues_trailing_52w": trail,
            "log_wsl": np.log1p(wsl),
            "log_trail": np.log1p(trail),
            "month": dates.month,
            "quarter_end": dates.month.isin([3, 6, 9, 12]).astype(int),
            "issuer_type": st["issuer_type"],
            "sector": st["sector"],
            "country": st["country"],
            "rating_filled": st["rating_filled"],
            "rating_missing": st["rating_missing"],
        })
        df["date"] = dates.values
        return df

    def forward_hazards(self, issuer: str, horizon: int) -> pd.DataFrame:
        df = self._forward_frame(issuer, horizon)
        df["hazard"] = self.timing.predict_hazard(df)
        df["survival"] = np.cumprod(1.0 - df["hazard"].values)   # S(k) at row k (k=1..horizon)
        df["week_ahead"] = np.arange(1, horizon + 1)
        return df

    def _survival_array(self, issuer: str, horizon: int) -> np.ndarray:
        """S(0..horizon): S[0]=1, S[k]=prod_{j<=k}(1-h_j)."""
        h = self.forward_hazards(issuer, horizon)["hazard"].values
        return np.concatenate([[1.0], np.cumprod(1.0 - h)])

    def timing_prob(self, issuer: str, horizon_spec) -> tuple[float, pd.DataFrame]:
        """horizon_spec = ('within', k) or ('interval', a, b) or ('recurrent', a, b)."""
        kind = horizon_spec[0]
        hmax = horizon_spec[-1]
        fh = self.forward_hazards(issuer, int(hmax))
        S = np.concatenate([[1.0], fh["survival"].values])       # S[0..hmax]
        if kind == "within":
            k = int(horizon_spec[1]); p = 1.0 - S[k]
        elif kind == "interval":
            a, b = int(horizon_spec[1]), int(horizon_spec[2]); p = S[a - 1] - S[b]
        elif kind == "recurrent":
            a, b = int(horizon_spec[1]), int(horizon_spec[2]); p = 1.0 - S[b] / S[a - 1]
        else:
            raise ValueError(f"unknown horizon kind {kind!r}")
        return float(np.clip(p, 0.0, 1.0)), fh

    # ---------- characteristics ----------
    def char_multiplier(self, issuer: str, characteristics: dict) -> tuple[float, list]:
        """Return (product of conditional probs, breakdown rows)."""
        mult, rows = 1.0, []
        if not characteristics:
            return 1.0, rows
        cm = self.chars
        for key, val in characteristics.items():
            if val is None:
                continue
            if key in ("callable", "puttable", "sinkable", "convertible", "fx"):
                col = "fx_flag" if key == "fx" else key
                p = cm.p_binary(issuer, col)
                p = p if val else (1.0 - p)
                label = f"{key}={'yes' if val else 'no'}"
            elif key == "coupon_type":
                p = cm.p_category(issuer, "coupon_type", val); label = f"coupon_type={val}"
            elif key == "coupon_interval":
                lo, hi = val; p = cm.p_coupon_interval(issuer, lo, hi); label = f"coupon in [{lo},{hi}]%"
            elif key == "tenor_bucket":
                idx = C.TENOR_BUCKET_LABELS.index(val)
                lo, hi = C.TENOR_BUCKETS_DAYS[idx]
                p = cm.p_tenor_interval_days(issuer, lo, hi); label = f"tenor {val}"
            elif key == "tenor_interval_days":
                lo, hi = val; p = cm.p_tenor_interval_days(issuer, lo, hi); label = f"tenor [{lo},{hi}]d"
            else:
                continue
            rows.append((label, float(p)))
            mult *= float(p)
        return mult, rows

    # ---------- combined ----------
    def probability(self, issuer: str, horizon_spec, characteristics: dict | None = None) -> dict:
        t_prob, fh = self.timing_prob(issuer, horizon_spec)
        c_mult, c_rows = self.char_multiplier(issuer, characteristics or {})
        return {
            "issuer": issuer,
            "horizon_spec": horizon_spec,
            "timing_prob": t_prob,
            "char_multiplier": c_mult,
            "char_breakdown": c_rows,
            "probability": float(np.clip(t_prob * c_mult, 0.0, 1.0)),
            "forward": fh,
        }


if __name__ == "__main__":
    import pickle
    from panel import build_panel
    from timing_model import fit_timing_model

    grid, events, iss = build_panel()
    tm = fit_timing_model(grid)
    cm = CharModels.fit(events)
    model = IssuanceModel.build(tm, cm, iss, events)

    for name in ["Federal Home Loan Bank System", "Government of Argentina",
                 "The Goldman Sachs Group, Inc."]:
        if name not in model.issuer_static.index:
            continue
        p1 = model.probability(name, ("within", 1))
        p8 = model.probability(name, ("within", 8))
        pc = model.probability(name, ("within", 8), {"callable": True, "coupon_interval": (4, 5)})
        pint = model.timing_prob(name, ("interval", 5, 10))[0]
        print(f"\n{name}")
        print(f"  P(issue next week)          = {p1['probability']:.3f}")
        print(f"  P(issue within 8 weeks)     = {p8['probability']:.3f}")
        print(f"  P(first issue in weeks 5-10)= {pint:.3f}")
        print(f"  P(callable & 4-5% coupon, 8w)= {pc['probability']:.4f}  "
              f"(timing {pc['timing_prob']:.3f} x {pc['char_multiplier']:.3f})")
