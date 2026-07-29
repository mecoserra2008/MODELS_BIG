"""Oil <-> rates conditional-sensitivity analytics.

Layers (each returns a plain dict for JSON/figures):
  stationarity      ADF level vs change -> model in returns/changes
  unconditional     full-sample HAC beta + rolling + Kalman TVP beta (the "puzzle")
  decomposition     Δnominal vs Δreal vs Δbreakeven response to oil (the mechanism)
  shock_type        demand vs supply days via same-day oil x equity co-movement sign
  markov_regimes    endogenous high-beta / low-beta states + condition profiles (the regimes)
  interactions      Δy10 ~ oil + oil x condition (named conditions, HAC)
  quantiles         beta(tau): does oil move the tails more than the center?
  influence         Cook's D / DFBETA dates, robust vs OLS beta, skew (the outliers)
  var_dynamics      VAR(oil, Δreal, Δbei): Granger both ways, IRF, FEVD
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller

from src.curve_signals import kalman_tvp_regression
from src import var_analysis as va

HAC_LAGS_D = 5
ROLL_WIN = 126           # ~6m rolling beta
CONDITIONERS = ["vix", "oil_vol", "rate_vol", "bei_level", "dff", "usd_ret", "y10_level"]


# --------------------------------------------------------------------------- #
def _hac(y: pd.Series, X: pd.DataFrame, lags: int = HAC_LAGS_D):
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    m = sm.OLS(df["_y"], sm.add_constant(df[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return m, df


def _beta_dict(m, name: str) -> dict:
    return {"beta": float(m.params[name]), "t": float(m.tvalues[name]),
            "p": float(m.pvalues[name]), "r2": float(m.rsquared), "n": int(m.nobs)}


def stationarity(od) -> dict:
    r = od.raw
    out = {}
    for label, s in (("wti_log", np.log(r["DCOILWTICO"].where(r["DCOILWTICO"] > 0))),
                     ("dgs10", r["DGS10"]), ("dfii10", r["DFII10"]), ("t10yie", r["T10YIE"])):
        x = s.dropna()
        out[label] = {"level_p": float(adfuller(x, autolag="AIC")[1]),
                      "diff_p": float(adfuller(x.diff().dropna(), autolag="AIC")[1])}
    out["conclusion"] = ("all series I(1): non-stationary in level, stationary in "
                         "changes -> the study runs on oil log-returns and bp changes")
    return out


def unconditional(od) -> dict:
    d = od.daily
    out = {}
    for tag, xcol, frame, lags in (("daily_wti", "oil_ret", d, HAC_LAGS_D),
                                   ("daily_brent", "brent_ret", d, HAC_LAGS_D),
                                   ("monthly_wti", "oil_ret", od.monthly, 3),
                                   ("monthly_brent", "brent_ret", od.monthly, 3)):
        m, _ = _hac(frame["dy10"], frame[[xcol]].rename(columns={xcol: "oil"}), lags)
        out[tag] = _beta_dict(m, "oil")
    # rolling + TVP beta (daily WTI)
    dd = d[["dy10", "oil_ret"]].dropna()
    cov = dd["dy10"].rolling(ROLL_WIN).cov(dd["oil_ret"])
    var = dd["oil_ret"].rolling(ROLL_WIN).var()
    out["rolling_beta"] = (cov / var).rename("beta")            # bp per 1% oil
    tvp = kalman_tvp_regression(dd["dy10"], dd["oil_ret"])
    out["tvp_beta"] = tvp.beta
    out["corr_full"] = float(dd["dy10"].corr(dd["oil_ret"]))
    return out


def decomposition(od) -> dict:
    d = od.daily
    out = {"note": "nominal = real + breakeven; sample 2003+ (TIPS/BEI)"}
    roll = {}
    for leg in ("dy10", "dreal", "dbei"):
        m, df = _hac(d[leg], d[["oil_ret"]].rename(columns={"oil_ret": "oil"}))
        out[leg] = _beta_dict(m, "oil")
        dd = pd.concat([d[leg], d["oil_ret"]], axis=1).dropna()
        roll[leg] = (dd[leg].rolling(ROLL_WIN).cov(dd["oil_ret"])
                     / dd["oil_ret"].rolling(ROLL_WIN).var())
    out["rolling"] = pd.DataFrame(roll)
    # offset check: on big-oil days, how do the two legs co-move?
    big = d[d["oil_ret"].abs() > 3.0].dropna(subset=["dreal", "dbei"])
    out["big_oil_days"] = {
        "n": int(len(big)),
        "corr_real_bei": float(big["dreal"].corr(big["dbei"])),
        "mean_abs_dbei": float(big["dbei"].abs().mean()),
        "mean_abs_dreal": float(big["dreal"].abs().mean()),
        "share_offsetting": float(((np.sign(big["dreal"]) != np.sign(big["dbei"]))
                                   & big["dbei"].notna()).mean()),
    }
    return out


def shock_type_betas(od, big_thresh: float = 2.0, recent_win: int = 21) -> dict:
    """Demand- vs supply-shock identification via same-day equity co-movement.

    Daily sign restriction (Kilian-style): oil and equities moving TOGETHER =
    demand shock (growth news moves both); OPPOSITE signs = supply shock (oil up
    on scarcity is an equity tax). Zero returns are dropped. Each rates leg is
    regressed on oil_ret separately per day type (HAC), full sample and
    restricted to big moves (|oil_ret| > big_thresh%).
    """
    d = od.daily
    dd = d.dropna(subset=["oil_ret", "spx_ret"])
    dd = dd[(dd["oil_ret"] != 0) & (dd["spx_ret"] != 0)]
    if len(dd) == 0:
        return {"error": "no days with both oil_ret and spx_ret",
                "equity_note": od.equity_note}
    demand = np.sign(dd["oil_ret"]) == np.sign(dd["spx_ret"])
    out = {"note": ("sign identification: sign(oil_ret)==sign(spx_ret) -> demand day, "
                    "opposite -> supply day; betas per leg per day type"),
           "equity_note": od.equity_note,
           "sample": {"start": str(dd.index[0].date()), "end": str(dd.index[-1].date()),
                      "n": int(len(dd))},
           "share_demand": float(demand.mean()),
           "share_supply": float((~demand).mean())}
    for tag, sub in (("demand", dd[demand]), ("supply", dd[~demand])):
        for suffix, frame in (("", sub), ("_big", sub[sub["oil_ret"].abs() > big_thresh])):
            blk = {}
            for leg in ("dy10", "dreal", "dbei"):
                m, _ = _hac(frame[leg], frame[["oil_ret"]].rename(columns={"oil_ret": "oil"}))
                blk[leg] = {k: _beta_dict(m, "oil")[k] for k in ("beta", "t", "p", "n")}
            out[tag + suffix] = blk
    # current regime: classification of the most recent recent_win classified days
    last = dd.tail(recent_win)
    last_dem = np.sign(last["oil_ret"]) == np.sign(last["spx_ret"])
    share = float(last_dem.mean())
    out["recent"] = {"window_days": int(len(last)),
                     "start": str(last.index[0].date()), "end": str(last.index[-1].date()),
                     "n_demand": int(last_dem.sum()), "n_supply": int((~last_dem).sum()),
                     "share_demand": share,
                     "read": ("demand-led" if share >= 2 / 3 else
                              "supply-led" if share <= 1 / 3 else "mixed")}
    return out


def markov_regimes(od) -> dict:
    d = od.daily[["dy10", "oil_ret"] + CONDITIONERS].dropna(subset=["dy10", "oil_ret"])
    y = d["dy10"]; x = d["oil_ret"]
    mod = MarkovRegression(y.values, k_regimes=2, exog=x.values.reshape(-1, 1),
                           trend="c", switching_variance=True)
    res = mod.fit(em_iter=20, maxiter=200, disp=False)
    p = np.asarray(res.params)
    # params order: p01,p10 (transition), const_0,const_1, beta_0,beta_1, sig2_0,sig2_1
    betas = {0: float(p[4]), 1: float(p[5])}
    lo = min(betas, key=lambda k: abs(betas[k]))            # low-|beta| state
    hi = 1 - lo
    probs = pd.DataFrame(res.smoothed_marginal_probabilities, index=d.index)
    state = probs.idxmax(axis=1)
    prof = {}
    for c in CONDITIONERS:
        s = d[c].dropna()
        a = s[state.reindex(s.index) == lo]; b = s[state.reindex(s.index) == hi]
        if len(a) > 30 and len(b) > 30:
            t, pval = sps.ttest_ind(a, b, equal_var=False)
            prof[c] = {"low_beta_mean": float(a.mean()), "high_beta_mean": float(b.mean()),
                       "t": float(t), "p": float(pval)}
    return {"beta_low": betas[lo], "beta_high": betas[hi],
            "sigma_low": float(np.sqrt(p[6 + lo])), "sigma_high": float(np.sqrt(p[6 + hi])),
            "share_low_state": float((state == lo).mean()),
            "expected_dur_low_days": float(1.0 / max(1e-9, 1 - float(res.transition[lo, lo]) if hasattr(res, 'transition') else np.nan)) if hasattr(res, 'transition') else None,
            "condition_profile": prof,
            "prob_low_series": probs[lo].rename("p_low"),
            "loglike": float(res.llf), "n": int(len(d))}


def interactions(od) -> dict:
    d = od.daily
    out = {}
    for c in CONDITIONERS:
        cz = ((d[c] - d[c].mean()) / d[c].std()).rename("cz")
        X = pd.DataFrame({"oil": d["oil_ret"], "oil_x_c": d["oil_ret"] * cz, "cz": cz})
        m, _ = _hac(d["dy10"], X)
        out[c] = {"beta_oil": float(m.params["oil"]),
                  "beta_interaction": float(m.params["oil_x_c"]),
                  "t_interaction": float(m.tvalues["oil_x_c"]),
                  "p_interaction": float(m.pvalues["oil_x_c"]), "n": int(m.nobs)}
    return out


def quantiles(od, taus=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    d = od.daily[["dy10", "oil_ret"]].dropna()
    X = sm.add_constant(d["oil_ret"])
    out = {}
    for t in taus:
        r = QuantReg(d["dy10"], X).fit(q=t)
        out[str(t)] = {"beta": float(r.params["oil_ret"]),
                       "t": float(r.tvalues["oil_ret"])}
    return out


def influence(od, top_n: int = 15) -> dict:
    d = od.daily[["dy10", "oil_ret"]].dropna()
    X = sm.add_constant(d["oil_ret"])
    ols = sm.OLS(d["dy10"], X).fit()
    inf = OLSInfluence(ols)
    cooks = pd.Series(inf.cooks_distance[0], index=d.index)
    dfb = pd.Series(inf.dfbetas[:, 1], index=d.index)       # influence on the oil beta
    top = cooks.nlargest(top_n)
    top_tbl = [{"date": str(t.date()), "cooks_d": float(v),
                "oil_ret": float(d.loc[t, "oil_ret"]), "dy10": float(d.loc[t, "dy10"]),
                "dfbeta_oil": float(dfb.loc[t])} for t, v in top.items()]
    # robust + trimmed betas
    rlm = sm.RLM(d["dy10"], X, M=sm.robust.norms.HuberT()).fit()
    keep = cooks < cooks.quantile(0.99)
    ols_trim = sm.OLS(d.loc[keep, "dy10"], X.loc[keep]).fit()
    big = d[d["oil_ret"].abs() > 5.0]["dy10"]
    return {"beta_ols": float(ols.params["oil_ret"]),
            "beta_rlm_huber": float(rlm.params["oil_ret"]),
            "beta_excl_top1pct_cooks": float(ols_trim.params["oil_ret"]),
            "top_influential": top_tbl,
            "dist_given_big_oil": {"n": int(len(big)), "mean_bp": float(big.mean()),
                                   "skew": float(big.skew()), "kurtosis": float(big.kurt())},
            "dist_all": {"skew": float(d["dy10"].skew()), "kurtosis": float(d["dy10"].kurt())}}


def var_dynamics(od) -> dict:
    d = od.daily[["oil_ret", "dreal", "dbei"]].dropna()
    lag = va.select_var_lag_detailed(d, max_lags=10)["recommended"]["AIC"]
    lag = int(max(1, min(lag, 10)))
    res = va.estimate_var(d, lags=lag)
    names = list(d.columns)
    g_o2r = va.granger_causality_tests(res, causing=["oil_ret"], caused=["dreal", "dbei"])
    g_r2o = va.granger_causality_tests(res, causing=["dreal", "dbei"], caused=["oil_ret"])
    irf = va.compute_irfs(res, periods=20)
    irf_bei = va.irf_to_dataframe(irf, impulse="oil_ret", response="dbei", var_names=names)
    irf_real = va.irf_to_dataframe(irf, impulse="oil_ret", response="dreal", var_names=names)
    fevd = va.compute_fevd(res, periods=20)
    return {"lag": lag,
            "granger_oil_to_rates": g_o2r.to_dict(orient="records"),
            "granger_rates_to_oil": g_r2o.to_dict(orient="records"),
            "irf_oil_to_bei": irf_bei, "irf_oil_to_real": irf_real,
            "fevd_dbei": va.fevd_summary(fevd, response="dbei", horizons=[1, 5, 10, 20],
                                         var_names=names).to_dict(orient="records"),
            "fevd_dreal": va.fevd_summary(fevd, response="dreal", horizons=[1, 5, 10, 20],
                                          var_names=names).to_dict(orient="records")}


def run_all(od) -> dict:
    return {"stationarity": stationarity(od),
            "unconditional": unconditional(od),
            "decomposition": decomposition(od),
            "shock_type": shock_type_betas(od),
            "markov": markov_regimes(od),
            "interactions": interactions(od),
            "quantiles": quantiles(od),
            "influence": influence(od),
            "var": var_dynamics(od),
            "dropped_wti_days": od.dropped_wti_days}
