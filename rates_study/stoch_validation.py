"""Walk-forward validation + component selection for the US10Y conditional fan.

Every candidate component of the live fan must earn its place here, out of sample:
  P1  coverage/PIT of the bands (Kupiec, PIT-KS)
  P2  drift honesty: {trend160, zero, shrunk50, trend5y} on point RMSE + Brier (winner -> default)
  P3  distributional scoreboard (CRPS + log-score + coverage) across VARIANTS:
        heston_base    GARCH(1,1)-mapped Heston, zero drift (current live model)
        tilt50/tilt100 positioning-crowding drift tilt (lambda=0.5/1.0), expanding estimate
        widen          stress-regime dispersion widening (tests: anything beyond GARCH v0?)
        event          calendar-aware vol (NFP first-Friday + FOMC), expanding multiplier
        vasicek / cir  mean-reverting comparators fitted per-origin
  The accepted configuration {process, tilt_lambda, widen, event} is written to
  stoch_validation.json and consumed by stochastic.py --drift/--config auto.

    python -m rates_study.stoch_validation
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from . import calendar_events as cal
from . import data as data_mod
from .arima import _dm_test
from .stochastic import (GARCH_WIN, HORIZON, RESULTS, TREND_WIN,
                         garch11_mle, heston_simulate, vasicek_fit)

ORANGE = "#FF8200"; GREY = "#3C3C3C"; RED = "#C00000"; BLUE = "#1F4E79"; GREEN = "#2E7D32"
N_SIM = 2000
FIRST_ORIGIN = "2010-01-01"
K_CROWD = 1.5
WIDEN_M = 1.15
TILT_HORIZON_W = 6            # ~30bd in weeks for the expanding tilt estimate


# --------------------------------------------------------------------------- #
def kupiec(hits: np.ndarray, p: float) -> dict:
    n = len(hits); x = int(hits.sum())
    phat = x / n if n else np.nan
    if n == 0 or phat in (0.0, 1.0):
        return {"n": n, "hit_rate": phat, "lr": np.nan, "p_value": np.nan}
    lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    return {"n": n, "hit_rate": float(phat), "lr": float(lr),
            "p_value": float(1 - sps.chi2.cdf(lr, df=1))}


def crps_sample(x: np.ndarray, y: float) -> float:
    """CRPS from an ensemble: mean|X-y| - 0.5 E|X-X'| (sorted-sample formula)."""
    xs = np.sort(np.asarray(x, float)); n = len(xs)
    t1 = float(np.mean(np.abs(xs - y)))
    t2 = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * xs) / (n * n))
    return t1 - t2


def nll_kde(x: np.ndarray, y: float) -> float:
    try:
        d = float(sps.gaussian_kde(x)(y)[0])
    except Exception:
        d = 0.0
    return float(-np.log(max(d, 1e-8)))


def _disp(term: np.ndarray, expo: float) -> np.ndarray:
    """Scale the log-dispersion of an ensemble around its median by `expo`."""
    med = np.median(term)
    return med * (term / med) ** expo


# --------------------------------------------------------------------------- #
def run_validation():
    sd = data_mod.load()
    y = sd.dgs10_daily.dropna()
    r_all = np.log(y).diff().dropna()
    n = len(y)
    dy_abs = y.diff().abs()

    # regime inputs (aligned daily, as-of)
    z_wk = sd.composite_pub.dropna()                          # weekly crowding z (2016+)
    z_daily = z_wk.reindex(z_wk.index.union(y.index)).ffill().reindex(y.index)
    rate_vol = y.diff().rolling(21).std() * 100.0             # bp/day
    # weekly frame for the expanding tilt estimate
    y10_wk = sd.y10_w_pub.reindex(z_wk.index)
    fwd_wk = (y10_wk.shift(-TILT_HORIZON_W) - y10_wk)         # 6w fwd change, %
    ev_all = cal.event_dates(str(y.index.min().date()), "2027-12-31")
    is_event = pd.Series(y.index.normalize().isin(ev_all), index=y.index)

    i_first = max(GARCH_WIN + 1, int(np.searchsorted(y.index, pd.Timestamp(FIRST_ORIGIN))))
    origins = list(range(i_first, n - HORIZON, HORIZON))
    print(f"origins: {len(origins)}  events in sample: {int(is_event.sum())} "
          f"(NFP+FOMC)")

    drifts = ("trend160", "zero", "shrunk50", "trend5y")
    fc_err = {d: [] for d in drifts}; p_up = {d: [] for d in drifts}; up_real = []
    VARIANTS = ("heston_base", "tilt50", "tilt100", "widen", "event", "vasicek", "cir")
    S = {v: {"crps": [], "nll": [], "in90": [], "in50": [], "med_err": []}
         for v in VARIANTS}
    pit = {"heston_base": [], "gbm": []}
    in_g = {"in90": [], "in50": []}
    tilt_active = 0

    for k, i in enumerate(origins):
        t = y.index[i]
        y0 = float(y.iloc[i]); y_real = float(y.iloc[i + HORIZON])
        r_hist = r_all.iloc[:i]
        r160 = r_hist.iloc[-TREND_WIN:]
        sig = float(r160.std(ddof=1))
        mu160 = float(r160.mean()) + 0.5 * sig ** 2
        r5y = r_hist.iloc[-GARCH_WIN:]
        mu5y = float(r5y.mean()) + 0.5 * float(r5y.std(ddof=1)) ** 2
        mus = {"trend160": mu160, "zero": 0.0, "shrunk50": 0.5 * mu160, "trend5y": mu5y}

        # ---- P2 drift ----
        up_real.append(1.0 if y_real > y0 else 0.0)
        for d in drifts:
            mu = mus[d]
            fc_err[d].append(y0 * np.exp(mu * HORIZON) - y_real)
            m = (mu - 0.5 * sig ** 2) * HORIZON
            p_up[d].append(float(1 - sps.norm.cdf(-m / (sig * np.sqrt(HORIZON)))))

        # GBM zero-drift PIT/coverage
        m0 = (-0.5 * sig ** 2) * HORIZON
        u = float(sps.norm.cdf((np.log(y_real / y0) - m0) / (sig * np.sqrt(HORIZON))))
        pit["gbm"].append(u)
        in_g["in90"].append(1.0 if 0.05 <= u <= 0.95 else 0.0)
        in_g["in50"].append(1.0 if 0.25 <= u <= 0.75 else 0.0)

        # ---- base Heston (GARCH-mapped, zero drift) ----
        g = garch11_mle(r5y.values)
        kap = max(1e-4, 1 - g["persistence"]); th = max(1e-10, g["uncond_var"])
        xi = g["alpha"] * np.sqrt(2 * th)
        innov = g["r_centered"] ** 2 - g["v_series"]
        h = {"mu_daily": 0.0, "v0_daily": g["v_last"], "kappa_daily": kap,
             "theta_daily": th, "xi_daily": xi,
             "rho": float(np.corrcoef(g["r_centered"], innov)[0, 1])}
        term = heston_simulate(y0, h, HORIZON, N_SIM, seed=1000 + k)[:, -1]

        # ---- variant terminals (transforms of the same ensemble; honest & cheap) ----
        terms = {"heston_base": term}
        # positioning tilt (expanding, only when currently crowded)
        z_now = float(z_daily.iloc[i]) if np.isfinite(z_daily.iloc[i]) else np.nan
        tilt_y = 0.0
        if np.isfinite(z_now) and abs(z_now) > K_CROWD:
            prior = fwd_wk[(z_wk.index < t)].dropna()
            zp = z_wk.reindex(prior.index)
            side = prior[zp > K_CROWD] if z_now > 0 else prior[zp < -K_CROWD]
            if len(side) >= 30:
                tilt_y = float(side.mean() - prior.mean())    # % yield over ~30bd
                tilt_active += 1
        for lam, nm in ((0.5, "tilt50"), (1.0, "tilt100")):
            terms[nm] = term * np.exp(lam * tilt_y / y0)
        # stress widening (expanding tercile of rate vol)
        rv_hist = rate_vol.iloc[:i].dropna()
        stressed = (len(rv_hist) > 500 and
                    float(rate_vol.iloc[i]) > float(rv_hist.quantile(2 / 3)))
        terms["widen"] = _disp(term, WIDEN_M) if stressed else term
        # event-aware vol
        ev_hist = is_event.iloc[:i]
        d_hist = dy_abs.iloc[:i]
        me = float(d_hist[ev_hist.values].mean()); mn = float(d_hist[~ev_hist.values].mean())
        k_ev = me / mn if (np.isfinite(me) and mn > 0 and ev_hist.sum() > 20) else 1.0
        n_ev = int(is_event.iloc[i + 1:i + 1 + HORIZON].sum())
        scale = np.sqrt(((HORIZON - n_ev) + n_ev * k_ev ** 2) / HORIZON)
        terms["event"] = _disp(term, scale)
        # mean-reverting processes (per-origin fit)
        vp = vasicek_fit(y.iloc[:i])
        kv, thv = vp["kappa_daily"], vp["theta"]
        mean_v = thv + (y0 - thv) * np.exp(-kv * HORIZON)
        sd_v = vp["sigma_vasicek"] * np.sqrt((1 - np.exp(-2 * kv * HORIZON)) / (2 * kv))
        rng = np.random.default_rng(5000 + k)
        terms["vasicek"] = mean_v + sd_v * rng.standard_normal(N_SIM)
        Yc = np.full(N_SIM, y0)
        zc = rng.standard_normal((N_SIM, HORIZON))
        for s in range(HORIZON):
            Yc = np.maximum(Yc + kv * (thv - Yc) + vp["sigma_cir"] * np.sqrt(np.maximum(Yc, 1e-6)) * zc[:, s], 1e-4)
        terms["cir"] = Yc

        for v in VARIANTS:
            x = terms[v]
            S[v]["crps"].append(crps_sample(x, y_real))
            S[v]["nll"].append(nll_kde(x, y_real))
            S[v]["in90"].append(1.0 if np.percentile(x, 5) <= y_real <= np.percentile(x, 95) else 0.0)
            S[v]["in50"].append(1.0 if np.percentile(x, 25) <= y_real <= np.percentile(x, 75) else 0.0)
            S[v]["med_err"].append(float(np.median(x)) - y_real)
        pit["heston_base"].append(float((term <= y_real).mean()))

    # ---- P2 drift scoreboard ----
    e0 = np.array(fc_err["zero"]); drift_rows = {}
    for d in drifts:
        e = np.array(fc_err[d])
        drift_rows[d] = {"rmse": float(np.sqrt(np.mean(e ** 2))),
                         "brier_pup": float(np.mean((np.array(p_up[d]) - np.array(up_real)) ** 2)),
                         "dm_vs_zero": _dm_test(e, e0, h=1) if d != "zero" else {"dm": 0.0, "p_value": 1.0}}
    drift_winner = min(drift_rows, key=lambda d: drift_rows[d]["rmse"])

    # ---- P3 variant scoreboard ----
    board = {}
    base_crps = np.array(S["heston_base"]["crps"])
    for v in VARIANTS:
        crps = np.array(S[v]["crps"])
        board[v] = {"crps": float(crps.mean()), "nll": float(np.mean(S[v]["nll"])),
                    "cov90": kupiec(np.array(S[v]["in90"]), 0.90),
                    "cov50": kupiec(np.array(S[v]["in50"]), 0.50),
                    "rmse_med": float(np.sqrt(np.mean(np.array(S[v]["med_err"]) ** 2))),
                    "dm_crps_vs_base": (_dm_test(np.sqrt(np.maximum(crps, 0)),
                                                 np.sqrt(np.maximum(base_crps, 0)), h=1)
                                        if v != "heston_base" else {"dm": 0.0, "p_value": 1.0})}
    # component acceptance (CRPS must improve AND 90% coverage not deteriorate)
    def better(v):
        b = board[v]; base = board["heston_base"]
        return (b["crps"] < base["crps"] and
                abs(b["cov90"]["hit_rate"] - 0.90) <= abs(base["cov90"]["hit_rate"] - 0.90) + 0.02)
    tilt_pick = min(("heston_base", "tilt50", "tilt100"), key=lambda v: board[v]["crps"])
    accepted = {"process": min(("heston_base", "vasicek", "cir"),
                               key=lambda v: (abs(board[v]["cov90"]["hit_rate"] - 0.90),
                                              board[v]["crps"])),
                "tilt_lambda": {"heston_base": 0.0, "tilt50": 0.5, "tilt100": 1.0}[tilt_pick],
                "tilt_improves": bool(tilt_pick != "heston_base" and better(tilt_pick)),
                "widen": bool(better("widen")),
                "event_vol": bool(better("event")),
                "n_tilt_active_origins": int(tilt_active)}

    out = {"n_origins": len(origins), "horizon_bdays": HORIZON,
           "drift_test": {"scoreboard": drift_rows, "winner": drift_winner,
                          "base_rate_up": float(np.mean(up_real))},
           "coverage": {"gbm": {"band90": kupiec(np.array(in_g["in90"]), 0.90),
                                "band50": kupiec(np.array(in_g["in50"]), 0.50)},
                        "heston": {"band90": board["heston_base"]["cov90"],
                                   "band50": board["heston_base"]["cov50"]}},
           "pit_ks_gbm": float(sps.kstest(pit["gbm"], "uniform").pvalue),
           "pit_ks_heston": float(sps.kstest(pit["heston_base"], "uniform").pvalue),
           "variant_scoreboard": board, "accepted": accepted}
    (RESULTS / "stoch_validation.json").write_text(json.dumps(out, indent=2, default=str))

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    vs = list(VARIANTS)
    axes[0].bar(vs, [board[v]["crps"] * 100 for v in vs], color=BLUE, alpha=0.8)
    axes[0].axhline(board["heston_base"]["crps"] * 100, color=GREY, ls="--", lw=0.8)
    axes[0].set_title("CRPS by variant (bp, lower=better)"); axes[0].tick_params(axis="x", rotation=30)
    rates = [board[v]["cov90"]["hit_rate"] for v in vs]
    axes[1].bar(vs, rates, color=[GREEN if abs(r - 0.9) < 0.04 else RED for r in rates], alpha=0.8)
    axes[1].axhline(0.90, color=GREY, lw=1.2); axes[1].set_ylim(0, 1)
    axes[1].set_title("90%-band realized coverage"); axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(RESULTS / "fig_variant_scoreboard.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(pit["heston_base"], bins=10, range=(0, 1), color=RED, alpha=0.6)
    axes[0].axhline(len(origins) / 10, color=GREY, ls="--", lw=0.8)
    axes[0].set_title(f"PIT (Heston base), KS-p={out['pit_ks_heston']:.2f}")
    ds = list(drifts)
    axes[1].bar(ds, [drift_rows[d]["rmse"] for d in ds], color=ORANGE, alpha=0.85)
    axes[1].set_title(f"Drift 30bd RMSE (winner: {drift_winner})")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_fan_coverage.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    print("\nVARIANT SCOREBOARD (CRPS bp | NLL | cov90 | med-RMSE | DM-p vs base):")
    for v in vs:
        b = board[v]
        print(f"  {v:12s} {b['crps']*100:6.2f} | {b['nll']:5.2f} | "
              f"{b['cov90']['hit_rate']:.2f} | {b['rmse_med']:.4f} | "
              f"{b['dm_crps_vs_base']['p_value']:.3f}")
    print(f"\nACCEPTED CONFIG: {accepted}")
    print(f"DRIFT winner: {drift_winner} | tilt active at {tilt_active} origins")
    print(f"Wrote stoch_validation.json + figures to {RESULTS}")
    return out


if __name__ == "__main__":
    run_validation()
