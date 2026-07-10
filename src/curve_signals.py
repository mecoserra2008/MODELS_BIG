"""
Curve relative-value signal engine for the 10s5s slope.

Estimation primitives (all causal — value at date t uses only data <= t):

- ``kalman_tvp_regression``   Time-varying (alpha_t, beta_t) hedge ratio between
                              the 10Y and 5Y legs via a random-walk state-space
                              regression. Yields the empirical DV01-style hedge
                              ratio and a stationary residual series r_t.
- ``kalman_local_linear_trend`` Local-linear-trend Kalman (statsmodels
                              UnobservedComponents) on the raw 10s5s spread ->
                              dynamic fair value mu_t + slope.
- ``bollinger_bands`` / ``zscore``  Rolling trailing bands / standardized score.
- ``ou_half_life``            Ornstein-Uhlenbeck / AR(1) mean-reversion half-life.
- ``generate_signals``        Band-crossing steepener/flattener positions.

Nothing here fetches data or writes files; the driver (run_10s5s.py) wires
these together. Only numpy / pandas / statsmodels are used (all already
installed) so there is no new hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Kalman #1 — time-varying hedge ratio between the two legs
# ---------------------------------------------------------------------------

@dataclass
class TVPRegressionResult:
    """Output of the time-varying (alpha, beta) regression."""

    alpha: pd.Series          # time-varying intercept
    beta: pd.Series           # time-varying hedge ratio (d y10 / d y5)
    resid: pd.Series          # r_t = y10 - (alpha_t + beta_t * y5)  [stationary]
    obs_var: float            # estimated observation variance R
    state_var: float          # estimated state (random-walk) variance scale Q


def kalman_tvp_regression(
    y10: pd.Series,
    y5: pd.Series,
    delta: float = 1e-5,
    obs_var: float | None = None,
) -> TVPRegressionResult:
    """
    Kalman filter for the dynamic linear regression

        y10_t = alpha_t + beta_t * y5_t + eps_t,     eps_t ~ N(0, R)
        [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + w_t,  w_t ~ N(0, Q)

    The state-transition is a random walk (F = I). Q is parameterised as a
    Bruce-Hamilton "delta" ratio: Q = delta / (1 - delta) * R * I, a standard
    trick (see Montana/pairs-trading Kalman) that needs only one knob. This is
    the empirical, causal hedge ratio between the 10Y and 5Y legs — compare
    ``beta`` against the DV01-neutral ratio for a cross-check.

    Parameters
    ----------
    y10, y5 : pd.Series
        Aligned yield levels (percent). Must share an index.
    delta : float
        State-noise ratio in (0, 1). Larger -> beta adapts faster.
    obs_var : float or None
        Observation variance R. If None, estimated from a rolling OLS residual.

    Returns
    -------
    TVPRegressionResult
    """
    df = pd.concat([y10.rename("y10"), y5.rename("y5")], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("kalman_tvp_regression: need at least 30 aligned points")

    y = df["y10"].to_numpy(dtype=float)
    x = df["y5"].to_numpy(dtype=float)
    n = len(y)

    if obs_var is None:
        # crude R estimate from a static OLS fit
        A = np.vstack([np.ones(n), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        obs_var = float(np.var(y - A @ coef, ddof=2)) or 1e-6

    R = float(obs_var)
    Q = (delta / (1.0 - delta)) * R * np.eye(2)

    # state = [alpha, beta]
    state = np.zeros(2)
    # diffuse-ish prior
    P = np.eye(2) * 1e3

    alpha = np.empty(n)
    beta = np.empty(n)
    resid = np.empty(n)

    for t in range(n):
        # --- predict (random walk: state unchanged, covariance inflates) ---
        P = P + Q

        # --- observation ---
        H = np.array([1.0, x[t]])              # design row
        y_pred = H @ state
        e = y[t] - y_pred                       # innovation (causal residual)
        S = H @ P @ H + R                        # innovation variance
        K = (P @ H) / S                          # Kalman gain

        # --- update ---
        state = state + K * e
        P = P - np.outer(K, H) @ P

        alpha[t] = state[0]
        beta[t] = state[1]
        resid[t] = e

    idx = df.index
    return TVPRegressionResult(
        alpha=pd.Series(alpha, index=idx, name="alpha"),
        beta=pd.Series(beta, index=idx, name="beta"),
        resid=pd.Series(resid, index=idx, name="tvp_resid"),
        obs_var=R,
        state_var=float(Q[0, 0]),
    )


# ---------------------------------------------------------------------------
# Kalman #2 — local linear trend on the raw spread (dynamic fair value)
# ---------------------------------------------------------------------------

@dataclass
class LocalTrendResult:
    """Output of the local-linear-trend Kalman on the spread."""

    level: pd.Series          # mu_t — dynamic fair value of the spread
    trend: pd.Series          # slope of the level
    resid: pd.Series          # spread - mu_t
    innov_std: float          # sqrt of the fitted observation variance
    params: dict = field(default_factory=dict)


def kalman_local_linear_trend(
    spread: pd.Series,
    fixed_params: dict | None = None,
) -> LocalTrendResult:
    """
    Local-linear-trend state space model on the 10s5s spread:

        spread_t = mu_t + eps_t
        mu_t     = mu_{t-1} + nu_{t-1} + xi_t
        nu_t     = nu_{t-1} + zeta_t

    Uses statsmodels ``UnobservedComponents`` (Kalman filter under the hood).
    The **filtered** (not smoothed) level is returned so the fair value at t is
    causal — it never peeks at future observations.

    Parameters
    ----------
    spread : pd.Series
        The 10Y-5Y spread (percentage points or bp; units carry through).
    fixed_params : dict or None
        If provided (e.g. from a frozen fit), the variances are fixed and the
        filter is only run forward — this is the "live mode" path. Keys are the
        statsmodels param names in ``model.param_names``.

    Returns
    -------
    LocalTrendResult
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    s = spread.dropna().astype(float)
    if len(s) < 30:
        raise ValueError("kalman_local_linear_trend: need at least 30 points")

    model = UnobservedComponents(s, level="local linear trend")

    if fixed_params is not None:
        ordered = [fixed_params[name] for name in model.param_names]
        res = model.filter(np.asarray(ordered, dtype=float))
        fitted_params = dict(zip(model.param_names, ordered))
    else:
        res = model.fit(disp=False, maxiter=200)
        fitted_params = dict(zip(model.param_names, res.params))

    # filtered_state[0] = level (mu_t), filtered_state[1] = trend (nu_t)
    level = pd.Series(res.filtered_state[0], index=s.index, name="level")
    trend = pd.Series(res.filtered_state[1], index=s.index, name="trend")
    resid = (s - level).rename("lt_resid")

    obs_var = float(fitted_params.get("sigma2.irregular", np.nan))
    innov_std = float(np.sqrt(obs_var)) if obs_var == obs_var else float("nan")

    return LocalTrendResult(
        level=level,
        trend=trend,
        resid=resid,
        innov_std=innov_std,
        params=fitted_params,
    )


# ---------------------------------------------------------------------------
# Rolling bands & standardized score
# ---------------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series,
    window: int = 60,
    k: float = 2.0,
) -> pd.DataFrame:
    """Trailing rolling mean and +/- k*std Bollinger bands (causal)."""
    mid = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": mid + k * sd,
            "lower": mid - k * sd,
            "std": sd,
        }
    )


def zscore(
    series: pd.Series,
    window: int = 60,
    center: pd.Series | None = None,
) -> pd.Series:
    """
    Rolling z-score of ``series``. If ``center`` (e.g. a Kalman fair value) is
    given, the deviation is measured against it while the scale is the trailing
    rolling std of the series — i.e. z_t = (series_t - center_t) / std_t.
    """
    sd = series.rolling(window, min_periods=window).std(ddof=0)
    if center is None:
        center = series.rolling(window, min_periods=window).mean()
    z = (series - center) / sd.replace(0.0, np.nan)
    return z.rename("zscore")


# ---------------------------------------------------------------------------
# Mean-reversion diagnostics
# ---------------------------------------------------------------------------

def ou_half_life(resid: pd.Series) -> float:
    """
    Half-life of mean reversion from an AR(1) / discretised OU fit:

        d(resid)_t = lambda * resid_{t-1} + e_t   ->   half life = -ln(2)/lambda

    Returns np.inf if the series is not mean-reverting (lambda >= 0).
    """
    r = resid.dropna().astype(float)
    if len(r) < 30:
        return float("nan")
    lag = r.shift(1).dropna()
    dr = (r - r.shift(1)).dropna()
    lag, dr = lag.align(dr, join="inner")
    A = np.vstack([np.ones(len(lag)), lag.to_numpy()]).T
    coef, *_ = np.linalg.lstsq(A, dr.to_numpy(), rcond=None)
    lam = coef[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    z: pd.Series,
    k_entry: float = 2.0,
    k_exit: float = 0.5,
) -> pd.Series:
    """
    Band-crossing mean-reversion positions from a z-score series.

    Convention (spread = 10Y - 5Y):
      * z > +k_entry  -> spread rich / too steep -> FLATTENER  -> position -1
      * z < -k_entry  -> spread cheap / too flat -> STEEPENER  -> position +1
      * |z| < k_exit  -> revert to fair value    -> flat        -> position  0
      * otherwise      -> hold the previous position (hysteresis)

    Stateful but strictly causal: position at t depends only on z_{<=t} and the
    position carried in from t-1.
    """
    z = z.copy()
    pos = np.zeros(len(z))
    state = 0
    vals = z.to_numpy()
    for t in range(len(z)):
        zt = vals[t]
        if np.isnan(zt):
            pos[t] = state
            continue
        if state == 0:
            if zt >= k_entry:
                state = -1
            elif zt <= -k_entry:
                state = 1
        else:
            if abs(zt) <= k_exit:
                state = 0
            # allow direct flip if it blows through the opposite entry band
            elif state == -1 and zt <= -k_entry:
                state = 1
            elif state == 1 and zt >= k_entry:
                state = -1
        pos[t] = state
    return pd.Series(pos, index=z.index, name="position")
