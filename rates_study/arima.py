"""Dynamic US10Y forecaster: simple ARIMA (+ ARIMAX positioning bridge), honestly backtested.

* ARIMA(p,1,q) on the DGS10 level, order chosen by AIC over a small grid ("simple ARIMA").
* Dynamic (recursive) walk-forward: fit on train, then roll one step at a time appending the
  realized value (refit periodically), so every forecast is genuine out-of-sample.
* Baselines to beat: random walk (y_hat = last) and AR(1). Skill vs RW is judged with a
  Diebold-Mariano test on squared errors (yields are near-random-walk, so beating RW is hard;
  we report the truth either way).
* Positioning bridge: weekly SARIMAX with exog = causal (publication-lagged) positioning
  composite vs the same model without exog -> does positioning improve the OOS yield forecast?
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DAILY_ORDER_GRID = [(p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2)]
DEFAULT_TEST_DAYS = 504    # ~2y OOS
REFIT_EVERY = 21           # re-estimate params ~monthly during the roll


def adf_report(series: pd.Series) -> dict:
    s = series.dropna()
    lvl = adfuller(s, autolag="AIC")
    dif = adfuller(s.diff().dropna(), autolag="AIC")
    return {"level_p": float(lvl[1]), "diff_p": float(dif[1]),
            "conclusion": "I(1): level non-stationary, first-difference stationary"
            if lvl[1] > 0.05 and dif[1] < 0.05 else "check: unexpected ADF pattern"}


def select_order(series: pd.Series, grid=DAILY_ORDER_GRID) -> tuple:
    s = series.dropna()
    best, best_aic = (1, 1, 1), np.inf
    for order in grid:
        try:
            aic = ARIMA(s, order=order).fit().aic
            if np.isfinite(aic) and aic < best_aic:
                best, best_aic = order, aic
        except Exception:
            continue
    return best, float(best_aic)


def _metrics(y_true, y_pred, y_prev) -> dict:
    """RMSE/MAE + directional accuracy (sign of the predicted vs realized change)."""
    yt, yp, y0 = map(lambda a: np.asarray(a, float), (y_true, y_pred, y_prev))
    err = yp - yt
    dir_ok = np.sign(yp - y0) == np.sign(yt - y0)
    moved = (yt - y0) != 0
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "dir_acc": float(np.mean(dir_ok[moved])) if moved.any() else float("nan"),
            "n": int(len(yt))}


def _dm_test(e1, e2, h: int = 1) -> dict:
    """Diebold-Mariano (squared-error loss). d = e1^2 - e2^2; <0 => model1 better.
    Returns the DM stat and two-sided p. HAC variance with h-1 lags."""
    from scipy import stats
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm": float("nan"), "p_value": float("nan")}
    dbar = d.mean()
    gamma0 = d.var(ddof=0)
    var = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var += 2 * (1 - lag / h) * cov
    dm = dbar / np.sqrt(var / n) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(dm))) if np.isfinite(dm) else np.nan
    return {"dm": float(dm), "p_value": float(p)}


def walk_forward_daily(series: pd.Series, order: tuple,
                       test_days: int = DEFAULT_TEST_DAYS,
                       refit_every: int = REFIT_EVERY) -> dict:
    """Recursive one-step-ahead OOS forecasts for ARIMA vs random-walk & AR(1).
    Fast roll: append realized obs with refit=False, re-estimate params every `refit_every`."""
    s = series.dropna()
    n = len(s)
    split = n - test_days
    if split < 250:
        split = n // 2
    idx = s.index
    vals = pd.Series(s.values)   # positional RangeIndex (FRED daily index is irregular)

    def roll(mk_order):
        res = ARIMA(vals.iloc[:split], order=mk_order).fit()
        preds, prev, actual, when = [], [], [], []
        for i in range(split, n):
            fc = float(res.forecast(1).iloc[0])
            preds.append(fc); prev.append(float(vals.iloc[i - 1]))
            actual.append(float(vals.iloc[i])); when.append(idx[i])
            # extend with the realized obs (index i continues the model index)
            res = res.append(vals.iloc[i:i + 1], refit=(((i - split) % refit_every) == 0))
        return (pd.Series(preds, index=when), pd.Series(actual, index=when),
                pd.Series(prev, index=when))

    p_arima, actual, prev = roll(order)
    p_ar1, _, _ = roll((1, 1, 0))
    p_rw = prev.copy()   # random walk: forecast = last value

    m = {"arima": _metrics(actual, p_arima, prev),
         "ar1": _metrics(actual, p_ar1, prev),
         "rw": _metrics(actual, p_rw, prev)}
    e_arima = (p_arima - actual).values
    e_rw = (p_rw - actual).values
    dm = _dm_test(e_arima, e_rw, h=1)   # <0 & sig => ARIMA beats RW
    return {"order": list(order), "test_start": str(actual.index.min().date()),
            "test_end": str(actual.index.max().date()),
            "metrics": m, "dm_arima_vs_rw": dm,
            "skill_vs_rw_rmse": float(m["arima"]["rmse"] / m["rw"]["rmse"]),
            "series": {"actual": actual, "arima": p_arima, "rw": p_rw}}


def positioning_bridge_weekly(y10_w: pd.Series, composite_pub: pd.Series,
                              order=(1, 1, 1), test_frac: float = 0.35) -> dict:
    """Weekly SARIMAX(y10) with vs without exog=positioning composite (causal), OOS.
    Answers: does positioning add out-of-sample yield-forecast skill beyond the ARIMA?"""
    df = pd.concat([y10_w.rename("y"), composite_pub.rename("z")], axis=1).dropna()
    n = len(df)
    split = int(n * (1 - test_frac))
    dates = df.index
    y = pd.Series(df["y"].values)             # positional RangeIndex
    z = pd.Series(df["z"].values)

    def roll(use_exog):
        def fit(end):
            ex = z.iloc[:end].values.reshape(-1, 1) if use_exog else None
            return SARIMAX(y.iloc[:end], exog=ex, order=order,
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        res = fit(split)
        preds, actual, prev, when = [], [], [], []
        for i in range(split, n):
            exf = z.iloc[i:i + 1].values.reshape(-1, 1) if use_exog else None
            fc = float(res.forecast(1, exog=exf).iloc[0])
            preds.append(fc); actual.append(float(y.iloc[i]))
            prev.append(float(y.iloc[i - 1])); when.append(dates[i])
            if ((i - split) % 8 == 0):
                res = fit(i + 1)
            else:
                exa = z.iloc[i:i + 1].values.reshape(-1, 1) if use_exog else None
                res = res.append(y.iloc[i:i + 1], exog=exa, refit=False)
        return (pd.Series(preds, index=when), pd.Series(actual, index=when),
                pd.Series(prev, index=when))

    p_x, actual, prev = roll(True)
    p_n, _, _ = roll(False)
    m_x, m_n = _metrics(actual, p_x, prev), _metrics(actual, p_n, prev)
    dm = _dm_test((p_x - actual).values, (p_n - actual).values, h=1)  # <0 => exog helps
    return {"order": list(order), "n_oos": int(n - split),
            "with_positioning": m_x, "arima_only": m_n,
            "dm_exog_vs_noexog": dm,
            "rmse_ratio_x_over_plain": float(m_x["rmse"] / m_n["rmse"])}


def live_forecast(series: pd.Series, order: tuple, steps: int = 21) -> dict:
    """Refit on full sample -> point + 80/95% CI fan for the next `steps` business days."""
    s = series.dropna()
    res = ARIMA(pd.Series(s.values), order=order).fit()   # positional index (irregular dates)
    fc = res.get_forecast(steps)
    mean = fc.predicted_mean
    ci80 = fc.conf_int(alpha=0.20); ci95 = fc.conf_int(alpha=0.05)
    fut_idx = pd.bdate_range(s.index.max() + pd.tseries.offsets.BDay(1), periods=steps)
    mean.index = ci80.index = ci95.index = fut_idx
    return {"last_obs": float(s.iloc[-1]), "last_date": str(s.index.max().date()),
            "steps": steps, "mean": mean, "ci80": ci80, "ci95": ci95,
            "point_1w": float(mean.iloc[4]) if steps > 4 else None,
            "point_1m": float(mean.iloc[-1])}


def run_arima(sd) -> dict:
    """Full ARIMA study dict from a StudyData bundle."""
    dgs10 = sd.dgs10_daily
    adf = adf_report(dgs10)
    order, aic = select_order(dgs10)
    wf = walk_forward_daily(dgs10, order)
    bridge = positioning_bridge_weekly(sd.y10_w_pub, sd.composite_pub, order=(1, 1, 1))
    live = live_forecast(dgs10, order)
    return {"adf": adf, "order": list(order), "order_aic": aic,
            "walk_forward_daily": wf, "positioning_bridge_weekly": bridge, "live": live}
