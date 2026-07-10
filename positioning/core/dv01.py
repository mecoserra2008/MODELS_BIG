"""Contract DV01 and DV01-weighting.

Positioning is reported in *contracts*, which are not additive across tenors: 100k of
2Y risk is a fraction of 100k of Ultra-Bond risk. To aggregate a trader's book into a
single duration-positioning number we convert every net position to DV01-equivalent
(USD change in value per 1bp) before summing.

Track A uses a documented par-bond approximation for the contract DV01:

    contract_DV01 ~= face_value * modified_duration(tenor, current_yield) * 1e-4

driven by the CTD-representative tenor in config.CONTRACTS and the live FRED yield.
Track B (Bloomberg) can drop in the exact CTD DV01 (RISK_MID / DUR_ADJ_MID on the CTD)
without touching anything downstream, because the scoring core only ever sees DV01.

Mirrors the intent of the repo's src/dv01.py (par-bond DV01, DV01-neutral sizing) but
is kept self-contained so the positioning package has no sys.path coupling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONTRACT_BY_KEY, ContractSpec


def _price_and_moddur(ytm: float, maturity: float, coupon: float | None,
                      freq: int = 2) -> tuple[float, float]:
    """Price (per 100 face) and modified duration of a level-coupon bond.

    ytm, coupon in decimal (e.g. 0.042). If coupon is None the bond is assumed at par
    (coupon = ytm -> price 100), which yields a clean CTD-like duration proxy.
    """
    if maturity <= 0:
        return 100.0, 0.0
    if coupon is None:
        coupon = ytm
    y = ytm / freq
    c = coupon / freq  # coupon per period (decimal)
    n = max(int(round(maturity * freq)), 1)
    t = np.arange(1, n + 1)
    cfs = np.full(n, c * 100.0)
    cfs[-1] += 100.0
    if abs(y) < 1e-12:
        disc = np.ones(n)
    else:
        disc = 1.0 / (1.0 + y) ** t
    pv = cfs * disc
    price = pv.sum()
    if price <= 0:
        return price, 0.0
    macaulay_periods = (t * pv).sum() / price
    macaulay_years = macaulay_periods / freq
    mod_dur = macaulay_years / (1.0 + y)
    return price, mod_dur


def contract_dv01(spec: ContractSpec, ytm_pct: float) -> float:
    """USD DV01 of one futures contract given the current yield (in percent).

    Falls back to a neutral mid-yield if the input is missing/NaN so a single stale
    yield print never nukes the whole panel.
    """
    if ytm_pct is None or not np.isfinite(ytm_pct):
        ytm_pct = 3.0
    price, mod_dur = _price_and_moddur(ytm_pct / 100.0, spec.tenor_years, coupon=None)
    return spec.face_value * mod_dur * (price / 100.0) * 1e-4


def contract_dv01_series(spec: ContractSpec, yields: pd.Series) -> pd.Series:
    """Time series of per-contract DV01 from a yield series (percent)."""
    return yields.apply(lambda y: contract_dv01(spec, y))


def add_net_dv01(futures: pd.DataFrame, yields_weekly: pd.DataFrame) -> pd.DataFrame:
    """Add a `net_dv01` column = net_contracts * per-contract DV01.

    futures        validated FuturesPanel (long), already aligned to weekly dates.
    yields_weekly  wide yields frame reindexed to the SAME weekly dates as futures,
                   forward-filled; columns are FRED series (DGS2, DGS5, ...).

    per-contract DV01 uses each instrument's config yield series and the yield value
    on that row's date. Returns a copy with `dv01_per_contract` and `net_dv01`.
    """
    out = futures.copy()
    y = yields_weekly.reindex(pd.to_datetime(out["date"].unique())).sort_index()

    def _dv01(row) -> float:
        spec = CONTRACT_BY_KEY[row["instrument"]]
        yv = y.at[pd.Timestamp(row["date"]), spec.fred_yield] \
            if spec.fred_yield in y.columns else np.nan
        return contract_dv01(spec, yv)

    out["dv01_per_contract"] = out.apply(_dv01, axis=1)
    out["net_dv01"] = out["net_contracts"] * out["dv01_per_contract"]
    return out
