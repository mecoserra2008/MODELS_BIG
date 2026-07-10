"""
DV01 translation layer for the 10s5s curve model.

Turns curve signals into duration-aware sizing:

- ``dv01_per_notional``     DV01 (price value of 1bp) of a par government bond.
- ``dv01_neutral_weights``  Notionals that make the 5Y and 10Y legs carry equal
                            and opposite DV01 (the pure curve trade).
- ``duration_allocation``   Given a *structural net-long* DV01 target and each
                            market's current curve signal, recommend where the
                            duration should sit (which leg carries it) and how
                            to shift it between legs/markets as flows arrive.

The desk context: there is a standing need to be net DV01 long, but flows land
at different times per market, so duration is parked in one leg now and moved to
another leg/market later. ``duration_allocation`` expresses that: the structural
long is always covered, while the *tilt* between the 5Y and 10Y leg follows the
signal (steepener -> weight the 5Y leg, flattener -> weight the 10Y leg).

Bond maths are the standard par-bond closed forms; DV01 here is per 1 unit of
notional (multiply by face value). Approximations are fine for hedge ratios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _price_and_duration(ytm: float, maturity: float, coupon: float,
                        freq: int = 2) -> tuple[float, float]:
    """
    Clean price (per 1 face) and modified duration of a fixed-coupon bond.

    ytm, coupon are decimals (e.g. 0.032 for 3.2%). Semiannual by default.
    """
    n = int(round(maturity * freq))
    if n <= 0:
        return 1.0, 0.0
    c = coupon / freq            # periodic coupon
    y = ytm / freq               # periodic yield
    t = np.arange(1, n + 1)
    disc = (1.0 + y) ** t
    cfs = np.full(n, c)
    cfs[-1] += 1.0               # redemption
    pv = cfs / disc
    price = pv.sum()
    # Macaulay duration in periods -> years, then modified duration
    mac_years = (t * pv).sum() / price / freq
    mod_dur = mac_years / (1.0 + y)
    return float(price), float(mod_dur)


def dv01_per_notional(ytm: float, maturity: float,
                      coupon: float | None = None, freq: int = 2) -> float:
    """
    DV01 (price change for a 1bp yield move) per 1 unit of notional.

    If ``coupon`` is None the bond is assumed to trade at par (coupon = ytm),
    which is the natural assumption for on-the-run benchmark govvies and is
    accurate enough for curve hedge ratios.

    ``ytm`` may be passed in percent (e.g. 3.2) or decimal (0.032); values with
    magnitude > 1 are treated as percent and converted.
    """
    if abs(ytm) > 1.0:
        ytm = ytm / 100.0
    if coupon is None:
        coupon = ytm
    elif abs(coupon) > 1.0:
        coupon = coupon / 100.0
    price, mod_dur = _price_and_duration(ytm, maturity, coupon, freq)
    # DV01 = modified duration * price * 1bp
    return mod_dur * price * 1e-4


@dataclass
class LegWeights:
    """DV01-neutral notionals for a 5s/10s curve trade."""

    notional_5y: float        # + = long the 5Y leg
    notional_10y: float       # + = long the 10Y leg
    dv01_5y: float            # DV01 per unit notional, 5Y
    dv01_10y: float           # DV01 per unit notional, 10Y
    hedge_ratio: float        # notional_5y / notional_10y for DV01 neutrality


def dv01_neutral_weights(
    y5: float,
    y10: float,
    target_dv01: float = 10_000.0,
    direction: int = 1,
    m5: float = 5.0,
    m10: float = 10.0,
) -> LegWeights:
    """
    Size a DV01-neutral 10s5s trade to ``target_dv01`` of curve exposure.

    ``direction``: +1 = steepener (long 5Y DV01 / short 10Y DV01),
                   -1 = flattener (short 5Y DV01 / long 10Y DV01).

    Each leg is sized to ``target_dv01`` of DV01 (equal and opposite), so the
    trade is first-order insensitive to parallel shifts and expresses only the
    slope. Returns notionals per 1 face unit of DV01 accounting.
    """
    dv01_5 = dv01_per_notional(y5, m5)
    dv01_10 = dv01_per_notional(y10, m10)

    # notional so that |notional * dv01_per_notional| == target_dv01
    n5 = direction * target_dv01 / dv01_5
    n10 = -direction * target_dv01 / dv01_10
    hedge = abs(n5 / n10) if n10 != 0 else float("nan")
    return LegWeights(
        notional_5y=float(n5),
        notional_10y=float(n10),
        dv01_5y=float(dv01_5),
        dv01_10y=float(dv01_10),
        hedge_ratio=float(hedge),
    )


def duration_allocation(
    structural_dv01_target: float,
    market_signals: dict[str, dict],
    curve_tilt_frac: float = 0.35,
) -> pd.DataFrame:
    """
    Translate a structural net-long DV01 target into per-market, per-leg DV01,
    tilting each market's duration into the leg its signal favours.

    Parameters
    ----------
    structural_dv01_target : float
        Desired net long DV01 across the book (sum over markets/legs).
    market_signals : dict
        ``{market: {"signal": int, "y5": float, "y10": float, "z": float}}``
        where signal is +1 steepener / -1 flattener / 0 flat, from the model.
    curve_tilt_frac : float
        Fraction of a market's structural DV01 that is *tilted* onto the
        signalled leg (the rest is split to stay broadly neutral within the
        market). 0 = no tilt (duration split evenly), 1 = all in one leg.

    Returns
    -------
    pd.DataFrame indexed by market with columns:
        signal, z, dv01_5y, dv01_10y, dv01_total, tilt_leg, note

    Interpretation for the desk: ``dv01_total`` per market sums to the
    structural target (each market carries an equal share of the net long),
    while ``dv01_5y`` / ``dv01_10y`` show *where that duration sits*. When a
    market flips signal, the delta between successive tables is the duration to
    shift from one leg to the other (or, comparing markets, from one curve to
    another as flows land).
    """
    markets = list(market_signals.keys())
    if not markets:
        return pd.DataFrame()

    per_market = structural_dv01_target / len(markets)
    base = per_market / 2.0                      # even split baseline
    tilt = per_market * curve_tilt_frac / 2.0    # amount moved across legs

    rows = []
    for mkt in markets:
        info = market_signals[mkt]
        sig = int(info.get("signal", 0))
        if sig > 0:              # steepener -> weight the 5Y leg
            dv5, dv10 = base + tilt, base - tilt
            leg = "5Y"
        elif sig < 0:            # flattener -> weight the 10Y leg
            dv5, dv10 = base - tilt, base + tilt
            leg = "10Y"
        else:                    # flat -> hold duration evenly
            dv5, dv10 = base, base
            leg = "even"
        rows.append(
            {
                "market": mkt,
                "signal": sig,
                "z": round(float(info.get("z", np.nan)), 2),
                "dv01_5y": round(dv5, 1),
                "dv01_10y": round(dv10, 1),
                "dv01_total": round(dv5 + dv10, 1),
                "tilt_leg": leg,
                "note": {1: "steepener", -1: "flattener", 0: "flat"}[sig],
            }
        )
    return pd.DataFrame(rows).set_index("market")
