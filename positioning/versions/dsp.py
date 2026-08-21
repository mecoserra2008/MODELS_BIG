"""DSP cycle-extraction primitives (Ehlers family) for the richer-deviations signals.

All filters are CAUSAL: the value at date t uses only data <= t (no look-ahead), so any
zero-crossing / turning point they produce can be traded and scored out-of-sample honestly.

  highpass_2pole   Ehlers 2-pole high-pass (12 dB/oct) — removes the trend so the residual
                   oscillates around zero. This is the fix for the trend-contaminated,
                   near-integrated net-DV01 level that the 3y double-z normalizes away.
  supersmoother    Ehlers 2-pole SuperSmoother low-pass — near-zero response at Nyquist with
                   far less lag than an EMA.
  roofing          highpass_2pole -> supersmoother = a band-passed, zero-mean oscillator.
  hilbert_phase    Instantaneous amplitude / phase / Sine / LeadSine of an oscillator. The
                   default `causal=True` derives the quadrature from the (90-degree-leading)
                   causal first difference; `causal=False` uses scipy's full-series Hilbert
                   (NON-causal — descriptive plots only, never a scored signal).
  fisher           Fisher transform (arctanh of a rolling min-max normalized series) — sharpens
                   rounded extremes into crisp threshold crossings.

Refs: Ehlers, "Predictive Indicators" / "Cybernetic Analysis for Stocks and Futures".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def _finite_run(x: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Return (dropna'd series, its float values) — filters run on the contiguous finite run."""
    s = x.dropna().astype(float)
    return s, s.to_numpy()


def supersmoother(x: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers 2-pole SuperSmoother low-pass (causal). `period` = critical period in bars(=weeks).

        a1 = exp(-1.414*pi/period);  b1 = 2*a1*cos(1.414*pi/period)
        c1 = 1-b1+a1^2;  Filt = c1*(x+x[-1])/2 + b1*Filt[-1] - a1^2*Filt[-2]
    """
    s, v = _finite_run(x)
    n = len(v)
    if n < 3:
        return pd.Series(np.nan, index=x.index, name="ss")
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2.0 * a1 * np.cos(1.414 * np.pi / period)
    c3 = -a1 * a1
    c2 = b1
    c1 = 1.0 - c2 - c3
    f = np.empty(n)
    f[0] = v[0]
    f[1] = v[1]
    for i in range(2, n):
        f[i] = c1 * (v[i] + v[i - 1]) / 2.0 + c2 * f[i - 1] + c3 * f[i - 2]
    return pd.Series(f, index=s.index, name="ss").reindex(x.index)


def highpass_2pole(x: pd.Series, period: int = 48) -> pd.Series:
    """Ehlers 2-pole high-pass (causal, 12 dB/oct). Removes cycles longer than `period` bars,
    giving a zero-mean residual — the core "make it oscillate" detrender.

        alpha1 = (cos(.707*2pi/period)+sin(.707*2pi/period)-1)/cos(.707*2pi/period)
        HP = (1-alpha1/2)^2*(x-2x[-1]+x[-2]) + 2(1-alpha1)HP[-1] - (1-alpha1)^2 HP[-2]
    """
    s, v = _finite_run(x)
    n = len(v)
    if n < 3:
        return pd.Series(np.nan, index=x.index, name="hp")
    c = np.cos(0.707 * 2.0 * np.pi / period)
    sn = np.sin(0.707 * 2.0 * np.pi / period)
    alpha1 = (c + sn - 1.0) / c
    k = (1.0 - alpha1 / 2.0) ** 2
    b = 2.0 * (1.0 - alpha1)
    d = (1.0 - alpha1) ** 2
    hp = np.zeros(n)
    for i in range(2, n):
        hp[i] = k * (v[i] - 2.0 * v[i - 1] + v[i - 2]) + b * hp[i - 1] - d * hp[i - 2]
    return pd.Series(hp, index=s.index, name="hp").reindex(x.index)


def roofing(x: pd.Series, hp_period: int = 48, ss_period: int = 10) -> pd.Series:
    """Roofing filter = 2-pole high-pass (trend kill) then SuperSmoother (noise kill).
    Output is a zero-mean band-pass oscillator over roughly [ss_period, hp_period] bars."""
    return supersmoother(highpass_2pole(x, hp_period), ss_period).rename("roofing")


def hilbert_phase(osc: pd.Series, period: int = 20, causal: bool = True) -> pd.DataFrame:
    """Instantaneous amplitude/phase + Sine/LeadSine of a (roofed) oscillator.

    causal=True (default, tradeable): for a narrowband r=A*cos(wt), dr/dt=-A*w*sin(wt), so the
    quadrature Q = -(1/scale)*Delta r leads by 90 degrees; phase=atan2(Q, r). Uses only the
    backward difference -> strictly causal. `period` sets the quadrature scale (nominal cycle).
    causal=False: scipy.signal.hilbert over the whole finite run (NON-causal; plots only).
    Returns columns: amp, phase, sine, leadsine.
    """
    s, v = _finite_run(osc)
    if len(v) < 3:
        return pd.DataFrame(index=osc.index, columns=["amp", "phase", "sine", "leadsine"])
    if causal:
        scale = 2.0 * np.sin(np.pi / period)
        q = -np.diff(v, prepend=v[0]) / (scale if scale else np.nan)
        i_comp = v
        amp = np.sqrt(i_comp ** 2 + q ** 2)
        phase = np.arctan2(q, i_comp)
    else:
        from scipy.signal import hilbert as _hilbert
        a = _hilbert(v)
        amp = np.abs(a)
        phase = np.angle(a)
    out = pd.DataFrame(
        {"amp": amp, "phase": phase,
         "sine": np.sin(phase), "leadsine": np.sin(phase + np.pi / 4.0)},
        index=s.index,
    )
    return out.reindex(osc.index)


def fisher(x: pd.Series, window: int = 52, clip: float = 0.999) -> pd.Series:
    """Fisher transform: rolling min-max normalize to [-1,1] then arctanh. Turns a near-uniform
    crowding measure into a heavy-tailed Gaussian -> sharp, decisive spikes at extremes."""
    s = x.astype(float)
    lo = s.rolling(window, min_periods=window // 2).min()
    hi = s.rolling(window, min_periods=window // 2).max()
    rng = (hi - lo).replace(0.0, np.nan)
    v = (2.0 * (s - lo) / rng - 1.0).clip(-clip, clip)
    return (0.5 * np.log((1.0 + v) / (1.0 - v))).rename("fisher")


if __name__ == "__main__":  # smoke test
    idx = pd.date_range("2018-01-07", periods=422, freq="W")
    t = np.arange(422)
    # trend + 30-week cycle + noise -> roofing should recover the cycle with a zero mean
    sig = pd.Series(0.02 * t + 3.0 * np.sin(2 * np.pi * t / 30) + np.random.default_rng(0).normal(0, 0.5, 422), index=idx)
    r = roofing(sig, hp_period=48, ss_period=10)
    ph = hilbert_phase(r, period=30)
    fz = fisher(r, 52)
    zc = int((np.sign(r.dropna()).diff().abs() > 0).sum())
    print(f"roofing: n={r.notna().sum()}  mean={r.mean():.4f} (~0)  zero-crossings={zc}  "
          f"(raw trend crossings would be ~0)")
    print(f"hilbert amp/phase finite: {ph['amp'].notna().sum()}  fisher finite: {fz.notna().sum()}")
    print("dsp.py OK" if abs(r.mean()) < 0.5 and zc > 10 else "dsp.py CHECK")
