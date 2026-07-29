"""US10Y yield as an asset price: GBM + Heston SV Monte Carlo — validated edition.

Models (yield treated exactly like a tradable price, per the desk request):
  * GBM:    dY = mu*Y dt + sigma*Y dW
  * Heston: dY = mu*Y dt + sqrt(v)*Y dW1 ;  dv = kappa*(theta-v)dt + xi*sqrt(v)dW2
  * Benchmarks: Vasicek / CIR on the yield level (mean-reverting comparators).

Upgrades over v1 (see stoch_validation.py for the evidence):
  * Heston params from a GARCH(1,1) MLE mapped to (kappa, theta, xi) with v0 = the
    filter's CURRENT conditional variance - no smoothed rolling-window proxy, so the
    vol-of-vol and rho are no longer attenuated by construction. Block-bootstrap CIs.
  * Antithetic variates + MC standard errors on quantiles.
  * DRIFT is a choice, not an assumption: {trend160 (since ~Dec-2025), zero (RW),
    shrunk50, trend5y}. Default = the walk-forward OOS winner recorded by
    stoch_validation.py (falls back to zero). The trend fan stays as a labeled scenario.
  * Desk layer: barrier touch probabilities, 30d yield VaR/ES, DV01-translated P&L fan.
  * Outlook panel reconciling positioning / ARIMA / stochastic views.

    python -m rates_study.stochastic [--drift auto|trend160|zero|shrunk50|trend5y]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import minimize

from . import data as data_mod

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_rates_study"
RESULTS.mkdir(parents=True, exist_ok=True)

DRIFT_START = "2025-12-01"
TREND_WIN = 160                # trailing window mirroring "since Dec-2025"
GARCH_WIN = 1260               # 5y for variance dynamics
HORIZON = 30
N_PATHS = 10_000
SEED = 20260723
BARRIERS = (5.00, 4.25)
NOTIONAL = 10_000_000

ORANGE = "#FF8200"; GREY = "#3C3C3C"; RED = "#C00000"; BLUE = "#1F4E79"; GREEN = "#2E7D32"


# --------------------------------------------------------------------------- #
# Drift + GBM
# --------------------------------------------------------------------------- #
def logret(y: pd.Series) -> pd.Series:
    return np.log(y.dropna()).diff().dropna()


def drift_options(y: pd.Series) -> dict:
    """Candidate daily drifts for dY/Y (mu = mean logret + sigma^2/2)."""
    r = logret(y)
    def mu_of(rr):
        return float(rr.mean()) + 0.5 * float(rr.std(ddof=1)) ** 2
    mu160 = mu_of(r.iloc[-TREND_WIN:])
    return {"trend160": mu160, "zero": 0.0, "shrunk50": 0.5 * mu160,
            "trend5y": mu_of(r.iloc[-GARCH_WIN:])}


def estimate_gbm(y: pd.Series, window: int = TREND_WIN) -> dict:
    r = logret(y).iloc[-window:]
    sigma_d = float(r.std(ddof=1))
    mu_d = float(r.mean()) + 0.5 * sigma_d ** 2
    return {"window_start": str(r.index.min().date()), "window_end": str(r.index.max().date()),
            "n_obs": int(len(r)), "mu_daily": mu_d, "sigma_daily": sigma_d,
            "mu_annual": mu_d * 252, "sigma_annual": sigma_d * np.sqrt(252)}


def gbm_forecast(y0: float, mu: float, sigma: float, horizon: int = HORIZON,
                 qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    t = np.arange(1, horizon + 1)
    out = {"expected": y0 * np.exp(mu * t)}
    for q in qs:
        z = sps.norm.ppf(q)
        out[f"q{int(q*100):02d}"] = y0 * np.exp((mu - 0.5 * sigma ** 2) * t
                                                + sigma * np.sqrt(t) * z)
    return pd.DataFrame(out, index=t)


# --------------------------------------------------------------------------- #
# GARCH(1,1) MLE -> Heston mapping (P4)
# --------------------------------------------------------------------------- #
def garch11_mle(r: np.ndarray) -> dict:
    """Plain GARCH(1,1) via Nelder-Mead MLE. Returns params + conditional-variance path."""
    r = np.asarray(r, float)
    r = r - r.mean()
    uvar = float(r.var(ddof=1))
    n = len(r)

    def filt(w, a, b):
        v = np.empty(n); v[0] = uvar
        for t in range(1, n):
            v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
        return np.maximum(v, 1e-14)

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        v = filt(w, a, b)
        return 0.5 * float(np.sum(np.log(v) + r ** 2 / v))

    res = minimize(nll, x0=np.array([uvar * 0.05, 0.06, 0.90]), method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-12, "fatol": 1e-9})
    w, a, b = res.x
    v = filt(w, a, b)
    return {"omega": float(w), "alpha": float(a), "beta": float(b),
            "persistence": float(a + b), "uncond_var": float(w / max(1e-12, 1 - a - b)),
            "v_last": float(w + a * r[-1] ** 2 + b * v[-1]),
            "v_series": v, "r_centered": r, "converged": bool(res.success)}


def heston_from_garch(y: pd.Series, mu_daily: float, window: int = GARCH_WIN) -> dict:
    """Map a GARCH(1,1) fit to daily-unit Heston params.
      kappa = 1-(alpha+beta) ; theta = omega/(1-alpha-beta) ; xi = alpha*sqrt(2*theta)
      (from Var(dv|v)=alpha^2*2v^2 ~ xi^2*v at v=theta) ; rho = corr(r, r^2 - v) ;
      v0 = the filter's one-step-ahead conditional variance TODAY."""
    r = logret(y).iloc[-window:].values
    g = garch11_mle(r)
    kappa = max(1e-4, 1.0 - g["persistence"])
    theta = max(1e-10, g["uncond_var"])
    xi = g["alpha"] * np.sqrt(2.0 * theta)
    innov = g["r_centered"] ** 2 - g["v_series"]
    rho = float(np.corrcoef(g["r_centered"], innov)[0, 1])
    v0 = g["v_last"]
    feller = 2 * kappa * theta / max(1e-12, xi ** 2)
    return {"mu_daily": mu_daily, "v0_daily": float(v0), "kappa_daily": float(kappa),
            "theta_daily": float(theta), "xi_daily": float(xi), "rho": rho,
            "feller_ratio": float(feller), "feller_ok": bool(feller >= 1.0),
            "v0_ann_vol": float(np.sqrt(v0 * 252)), "theta_ann_vol": float(np.sqrt(theta * 252)),
            "garch": {k: g[k] for k in ("omega", "alpha", "beta", "persistence", "converged")},
            "calib_window_days": window}


def heston_param_cis(y: pd.Series, mu_daily: float, n_boot: int = 100,
                     block: int = 63, seed: int = SEED) -> dict:
    """Moving-block bootstrap CIs for the mapped Heston params (and Feller)."""
    rng = np.random.default_rng(seed)
    r = logret(y).iloc[-GARCH_WIN:].values
    n = len(r)
    keys = ("kappa_daily", "theta_ann_vol", "xi_daily", "rho", "feller_ratio")
    draws = {k: [] for k in keys}
    nblocks = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nblocks)
        rb = np.concatenate([r[s:s + block] for s in starts])[:n]
        try:
            g = garch11_mle(rb)
            kappa = max(1e-4, 1 - g["persistence"]); theta = max(1e-10, g["uncond_var"])
            xi = g["alpha"] * np.sqrt(2 * theta)
            innov = g["r_centered"] ** 2 - g["v_series"]
            vals = {"kappa_daily": kappa, "theta_ann_vol": np.sqrt(theta * 252),
                    "xi_daily": xi,
                    "rho": float(np.corrcoef(g["r_centered"], innov)[0, 1]),
                    "feller_ratio": 2 * kappa * theta / max(1e-12, xi ** 2)}
            for k in keys:
                draws[k].append(vals[k])
        except Exception:
            continue
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in draws.items() if len(v) > 20}


# --------------------------------------------------------------------------- #
# Simulators (antithetic)
# --------------------------------------------------------------------------- #
def heston_simulate(y0: float, h: dict, horizon: int = HORIZON,
                    n_paths: int = N_PATHS, seed: int = SEED) -> np.ndarray:
    """Full-truncation Euler, dt=1 day, antithetic variates (paths come in +/- pairs)."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    mu, kappa, theta, xi, rho = (h["mu_daily"], h["kappa_daily"], h["theta_daily"],
                                 h["xi_daily"], h["rho"])
    Y = np.full((n_paths, horizon + 1), y0)
    v = np.full(n_paths, h["v0_daily"])
    for t in range(1, horizon + 1):
        zv_h = rng.standard_normal(half); zi_h = rng.standard_normal(half)
        z_v = np.concatenate([zv_h, -zv_h]); z_i = np.concatenate([zi_h, -zi_h])
        z_s = rho * z_v + np.sqrt(1 - rho ** 2) * z_i
        vp = np.maximum(v, 0.0)
        Y[:, t] = Y[:, t - 1] * np.exp((mu - 0.5 * vp) + np.sqrt(vp) * z_s)
        v = v + kappa * (theta - vp) + xi * np.sqrt(vp) * z_v
    return Y


def gbm_simulate(y0: float, mu: float, sigma: float, horizon: int = HORIZON,
                 n_paths: int = N_PATHS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    zh = rng.standard_normal((half, horizon))
    z = np.vstack([zh, -zh])
    steps = (mu - 0.5 * sigma ** 2) + sigma * z
    return y0 * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)]))


def vasicek_fit(y: pd.Series, window: int = GARCH_WIN) -> dict:
    """dY = kappa(theta - Y)dt + sigma dW via OLS on daily changes (5y window)."""
    yy = y.dropna().iloc[-window:]
    dy = yy.diff().dropna(); ylag = yy.shift(1).reindex(dy.index)
    b, a = np.polyfit(ylag.values, dy.values, 1)
    kappa = max(1e-6, -b); theta = a / kappa
    resid = dy.values - (a + b * ylag.values)
    sig_v = float(np.std(resid, ddof=1))
    sig_cir = float(np.std(resid / np.sqrt(np.maximum(ylag.values, 1e-6)), ddof=1))
    return {"kappa_daily": float(kappa), "theta": float(theta),
            "half_life_days": float(np.log(2) / kappa), "sigma_vasicek": sig_v,
            "sigma_cir": sig_cir, "mean_reverting": bool(b < 0)}


def meanrev_simulate(y0: float, p: dict, kind: str = "vasicek", horizon: int = HORIZON,
                     n_paths: int = N_PATHS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    Y = np.full((n_paths, horizon + 1), y0)
    for t in range(1, horizon + 1):
        zh = rng.standard_normal(half); z = np.concatenate([zh, -zh])
        yp = np.maximum(Y[:, t - 1], 1e-6)
        vol = p["sigma_vasicek"] if kind == "vasicek" else p["sigma_cir"] * np.sqrt(yp)
        Y[:, t] = Y[:, t - 1] + p["kappa_daily"] * (p["theta"] - Y[:, t - 1]) + vol * z
    return Y


def summarize_paths(Y: np.ndarray, y0: float, qs=(5, 25, 50, 75, 95)) -> dict:
    term = Y[:, -1]
    # MC standard error of quantiles via 10 batches
    batches = np.array_split(term, 10)
    se = {f"q{q:02d}_se": float(np.std([np.percentile(b, q) for b in batches], ddof=1)
                                / np.sqrt(10)) for q in (5, 50, 95)}
    return {"terminal": {
                "mean": float(term.mean()), "median": float(np.median(term)),
                "std": float(term.std(ddof=1)), "skew": float(pd.Series(term).skew()),
                "kurtosis": float(pd.Series(term).kurt()),
                **{f"q{q:02d}": float(np.percentile(term, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                "p_above_start": float((term > y0).mean()), **se},
            "fan": {f"q{q:02d}": np.percentile(Y, q, axis=0) for q in qs}}


# --------------------------------------------------------------------------- #
# Desk layer (P5)
# --------------------------------------------------------------------------- #
def desk_metrics(Y: np.ndarray, y0: float, barriers=BARRIERS,
                 notional: float = NOTIONAL) -> dict:
    from src.dv01 import dv01_per_notional
    up, dn = barriers
    dy_bp = (Y[:, -1] - y0) * 100.0
    dv01 = dv01_per_notional(y0, 10.0)               # per 1 notional per 1bp
    pnl = -dv01 * notional * dy_bp                   # long 10y position
    var95, var99 = np.percentile(dy_bp, [95, 99])    # loss side = yields up
    es95 = float(dy_bp[dy_bp >= var95].mean()); es99 = float(dy_bp[dy_bp >= var99].mean())
    return {"p_touch_up": float((Y.max(axis=1) >= up).mean()), "barrier_up": up,
            "p_touch_dn": float((Y.min(axis=1) <= dn).mean()), "barrier_dn": dn,
            "dy30_bp": {"mean": float(dy_bp.mean()), "q05": float(np.percentile(dy_bp, 5)),
                        "q95": float(np.percentile(dy_bp, 95))},
            "yield_var_bp": {"var95": float(var95), "var99": float(var99),
                             "es95": es95, "es99": es99},
            "dv01_per_notional": float(dv01), "notional": notional,
            "pnl_long10y": {"mean": float(pnl.mean()),
                            "var95": float(np.percentile(pnl, 5)),
                            "var99": float(np.percentile(pnl, 1)),
                            "q05": float(np.percentile(pnl, 5)),
                            "q95": float(np.percentile(pnl, 95))}}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _hist_ax(ax, y, lookback=90):
    h = y.dropna().iloc[-lookback:]
    ax.plot(h.index, h.values, color=GREY, lw=1.3, label="US10Y actual")


def fig_fan(y, drift_name, mus, g, fc, summ, path):
    y0 = float(y.dropna().iloc[-1])
    fut = pd.bdate_range(y.dropna().index.max() + pd.tseries.offsets.BDay(1), periods=len(fc))
    fan = summ["fan"]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    _hist_ax(ax, y)
    ax.plot(fut, fan["q50"][1:], color=RED, lw=1.8, label=f"Heston median ({drift_name} drift)")
    ax.fill_between(fut, fan["q05"][1:], fan["q95"][1:], color=RED, alpha=0.10, label="5–95%")
    ax.fill_between(fut, fan["q25"][1:], fan["q75"][1:], color=RED, alpha=0.22, label="25–75%")
    # drift scenarios (GBM medians)
    t = np.arange(1, len(fc) + 1)
    for nm, mu in mus.items():
        if nm == drift_name:
            continue
        ax.plot(fut, y0 * np.exp((mu - 0.5 * g["sigma_daily"] ** 2) * t), ls="--", lw=1.0,
                alpha=0.8, label=f"scenario: {nm} drift")
    ax.set_title(f"US10Y 30bd fan — Heston (GARCH-mapped), drift='{drift_name}' "
                 f"(OOS-validated default); alternative drifts dashed")
    ax.set_ylabel("%"); ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_process_comparison(y0, terms: dict, path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = {"GBM": ORANGE, "Heston": RED, "Vasicek": BLUE, "CIR": GREEN}
    for nm, term in terms.items():
        kde_x = np.linspace(min(t.min() for t in terms.values()),
                            max(t.max() for t in terms.values()), 300)
        kde = sps.gaussian_kde(term)
        ax.plot(kde_x, kde(kde_x), color=colors[nm], lw=1.6,
                label=f"{nm}: med {np.median(term):.2f}, 90% [{np.percentile(term,5):.2f},"
                      f"{np.percentile(term,95):.2f}]")
    ax.axvline(y0, color=GREY, lw=1.2, label=f"today {y0:.2f}%")
    ax.set_title("Day-30 terminal distribution by process (same data, zero drift where applicable)")
    ax.set_xlabel("US10Y (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_pnl(desk: dict, Y: np.ndarray, y0: float, path):
    from src.dv01 import dv01_per_notional
    dv01 = desk["dv01_per_notional"]
    pnl_paths = -dv01 * desk["notional"] * (Y - y0) * 100.0 / 1e3   # k EUR/USD
    t = np.arange(Y.shape[1])
    fig, ax = plt.subplots(figsize=(9, 4))
    for q, a in ((5, 0.10), (25, 0.22)):
        ax.fill_between(t, np.percentile(pnl_paths, q, axis=0),
                        np.percentile(pnl_paths, 100 - q, axis=0), color=BLUE, alpha=a)
    ax.plot(t, np.percentile(pnl_paths, 50, axis=0), color=BLUE, lw=1.6, label="median P&L")
    ax.axhline(0, color=GREY, lw=0.7)
    p = desk["pnl_long10y"]
    ax.set_title(f"Long 10y {desk['notional']/1e6:.0f}M P&L fan — 30d VaR95 "
                 f"{p['var95']/1e3:,.0f}k, VaR99 {p['var99']/1e3:,.0f}k "
                 f"(DV01 {dv01*desk['notional']:,.0f}/bp)")
    ax.set_xlabel("business days"); ax.set_ylabel("P&L (thousands)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_outlook(y0, summ, drift_name, path):
    """Reconcile the stack's three 30d views with evidence grades."""
    views = []
    try:
        pj = json.loads((ROOT / "positioning/results/positioning_score.json").read_text())
        rr = pj["latest"].get("regime_read", {})
        p8 = rr.get("by_horizon", {}).get("8", {})
        views.append(("Positioning (contrarian)",
                      p8.get("p_selloff", np.nan),
                      f"crowded-{'long' if rr.get('regime')=='crowded_long' else 'short'} z="
                      f"{rr.get('composite_z', float('nan')):+.1f}; hist. base rate, 8w",
                      "descriptive (CIs include 0.5)"))
    except Exception:
        pass
    try:
        aj = json.loads((RESULTS / "us10y_arima_results.json").read_text())
        views.append(("ARIMA (≈ random walk)", 0.50,
                      f"OOS ≈ RW (ratio {aj['walk_forward_daily']['skill_vs_rw_rmse']:.3f})",
                      "OOS-validated: no directional signal"))
    except Exception:
        pass
    views.append((f"Stochastic fan ({drift_name} drift)",
                  summ["terminal"]["p_above_start"],
                  f"median {summ['terminal']['q50']:.2f}%, 90% "
                  f"[{summ['terminal']['q05']:.2f},{summ['terminal']['q95']:.2f}]",
                  "coverage-backtested (see stoch_validation)"))
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    names = [v[0] for v in views]; ps = [v[1] for v in views]
    cols = [ORANGE, GREY, RED][:len(views)]
    bars = ax.barh(names, ps, color=cols, alpha=0.85)
    ax.axvline(0.5, color=GREY, ls="--", lw=0.8)
    for b, v in zip(bars, views):
        ax.text(0.01, b.get_y() + b.get_height() / 2,
                f"{v[2]}  |  {v[3]}", va="center", fontsize=7, color="white")
    ax.set_xlim(0, 1); ax.set_xlabel("P(US10Y higher over the next ~30 days)")
    ax.set_title(f"30-day outlook — three views, one market (today {y0:.2f}%)")
    # context strip: oil shock regime + curve-crowding read (from the sibling studies)
    ctx = []
    try:
        stx = json.loads((ROOT / "results_oil_rates/oil_rates_results.json").read_text())["shock_type"]
        ctx.append(f"oil tape: {stx['recent']['read']} ({stx['recent']['n_demand']}D/"
                   f"{stx['recent']['n_supply']}S last 21d) -> use "
                   f"{'demand' if stx['recent']['read']=='demand' else 'weaker supply'}-day betas")
    except Exception:
        pass
    try:
        ccx = json.loads((ROOT / "positioning/results/positioning_score.json").read_text()
                         )["latest"]["curve_crowding"]
        z = ccx.get("z_by_category", {})
        ctx.append(f"curve crowding: {ccx.get('regime','?')} (levfund "
                   f"{z.get('lev_money', float('nan')):+.1f} steepener / assetmgr "
                   f"{z.get('asset_mgr', float('nan')):+.1f})")
    except Exception:
        pass
    if ctx:
        fig.text(0.01, -0.04, "context — " + "   |   ".join(ctx), fontsize=7, color=GREY)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
def apply_accepted(Y: np.ndarray, accepted: dict, y0: float, tilt_y: float = 0.0,
                   disp_expo: float = 1.0) -> np.ndarray:
    """Apply the OOS-accepted conditioning components to a simulated path array.
    With the current validation verdict (everything rejected) this is a no-op; the
    machinery exists so a future re-validation that accepts a component flows through."""
    out = Y.copy()
    lam = float(accepted.get("tilt_lambda", 0.0))
    if lam > 0 and accepted.get("tilt_improves") and tilt_y != 0.0:
        t = np.arange(Y.shape[1]) / max(1, Y.shape[1] - 1)
        out = out * np.exp(lam * tilt_y / y0 * t)[None, :]
    if disp_expo != 1.0 and (accepted.get("widen") or accepted.get("event_vol")):
        med = np.median(out, axis=0)
        out = med[None, :] * (out / med[None, :]) ** disp_expo
    return out


def main(drift: str = "auto"):
    sd = data_mod.load()
    y = sd.dgs10_daily
    y0 = float(y.dropna().iloc[-1])
    print(f"last obs {y.dropna().index.max().date()} = {y0:.2f}%")

    mus = drift_options(y)
    chosen = drift
    accepted = {"process": "heston_base", "tilt_lambda": 0.0, "tilt_improves": False,
                "widen": False, "event_vol": False, "source": "defaults (no validation file)"}
    if drift == "auto":
        try:
            vj = json.loads((RESULTS / "stoch_validation.json").read_text())
            chosen = vj["drift_test"]["winner"]
            accepted = {**vj.get("accepted", accepted), "source": "stoch_validation.json"}
            print(f"drift 'auto' -> OOS winner: {chosen} | accepted components: "
                  f"process={accepted['process']}, tilt={accepted['tilt_lambda']}, "
                  f"widen={accepted['widen']}, event={accepted['event_vol']}")
        except Exception:
            chosen = "zero"
            print("drift 'auto' -> no validation file, defaulting to zero (RW)")
    mu = mus[chosen]

    g = estimate_gbm(y)                              # sigma window = trend window
    print(f"GBM sigma={g['sigma_annual']:.1%}/y | drifts: " +
          ", ".join(f"{k}={v*252:+.1%}/y" for k, v in mus.items()))
    h = heston_from_garch(y, mu)
    print(f"Heston(GARCH-mapped): v0={h['v0_ann_vol']:.1%}/y theta={h['theta_ann_vol']:.1%}/y "
          f"kappa={h['kappa_daily']*252:.1f}/y xi={h['xi_daily']:.5f} rho={h['rho']:+.2f} "
          f"Feller={h['feller_ratio']:.2f}")
    cis = heston_param_cis(y, mu)
    print("  bootstrap 95% CIs:", {k: [round(a, 4), round(b, 4)] for k, (a, b) in cis.items()})

    fc = gbm_forecast(y0, mu, g["sigma_daily"])
    Yh = apply_accepted(heston_simulate(y0, h), accepted, y0)
    summ = summarize_paths(Yh, y0)

    # benchmarks (zero drift comparators + mean reversion)
    Yg = gbm_simulate(y0, mu, g["sigma_daily"])
    vp = vasicek_fit(y)
    print(f"Vasicek/CIR: kappa={vp['kappa_daily']*252:.2f}/y theta={vp['theta']:.2f}% "
          f"half-life={vp['half_life_days']:.0f}d mean_reverting={vp['mean_reverting']}")
    Yv = meanrev_simulate(y0, vp, "vasicek")
    Yc = meanrev_simulate(y0, vp, "cir")

    desk = desk_metrics(Yh, y0)

    fig_fan(y, chosen, mus, g, fc, summ, RESULTS / "fig_heston_fan.png")
    fig_process_comparison(y0, {"GBM": Yg[:, -1], "Heston": Yh[:, -1],
                                "Vasicek": Yv[:, -1], "CIR": Yc[:, -1]},
                           RESULTS / "fig_process_comparison.png")
    fig_pnl(desk, Yh, y0, RESULTS / "fig_pnl_fan.png")
    fig_outlook(y0, summ, chosen, RESULTS / "fig_outlook.png")

    out = {"as_of": str(y.dropna().index.max().date()), "y0": y0,
           "horizon_bdays": HORIZON, "n_paths": N_PATHS, "seed": SEED,
           "accepted_components": accepted,
           "drift": {"chosen": chosen, "options_annual": {k: v * 252 for k, v in mus.items()}},
           "gbm": {**g, "day30": {c: float(fc[c].iloc[-1]) for c in fc.columns}},
           "heston": {**{k: v for k, v in h.items() if k != "garch"},
                      "garch": h["garch"], "param_cis_95": cis,
                      "terminal": summ["terminal"]},
           "benchmarks": {"vasicek_cir": vp,
                          "terminal_medians": {"vasicek": float(np.median(Yv[:, -1])),
                                               "cir": float(np.median(Yc[:, -1]))}},
           "desk": desk,
           "caveat": ("Yield treated as a lognormal asset price (no mean reversion in "
                      "GBM/Heston); drift defaults to the walk-forward OOS winner; the "
                      "since-Dec-2025 trend is a labeled scenario, not the default.")}
    (RESULTS / "us10y_gbm_heston.json").write_text(json.dumps(out, indent=2))
    t = summ["terminal"]
    print(f"\nDay-30 ({chosen} drift): median {t['q50']:.2f}% (±{t['q50_se']*100:.1f}bp MC-se) "
          f"90% [{t['q05']:.2f},{t['q95']:.2f}] P(up)={t['p_above_start']:.0%}")
    print(f"Desk: P(touch {BARRIERS[0]}%)={desk['p_touch_up']:.0%} "
          f"P(touch {BARRIERS[1]}%)={desk['p_touch_dn']:.0%} | 10M long-10y VaR95 "
          f"{desk['pnl_long10y']['var95']/1e3:,.0f}k")
    print(f"Wrote figures + us10y_gbm_heston.json to {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", default="auto",
                    choices=["auto", "trend160", "zero", "shrunk50", "trend5y"])
    args = ap.parse_args()
    main(drift=args.drift)
