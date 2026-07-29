"""V8 Change-point / surprise overlay (state-space) — discrete EVENT signals, not added variance.

Rather than another rolling normalization, V8 emits *events*: three causal generators blended into
a directional crowding overlay for weekly CFTC positioning.

  changepoint_prob  Bayesian Online Change-Point Detection (Adams-MacKay 2007), self-implemented in
                    numpy with a Normal-InverseGamma conjugate -> Student-t predictive (no `ruptures`;
                    reference: Gundersen's BOCD). Weekly P(run_length==0) in [0,1] on the standardized
                    source. `bocpd_obs='t'` uses a modest alpha0 (fat-tailed dof=2) so one outlier
                    CFTC week does not masquerade as a break; `'gaussian'` uses a larger alpha0
                    (near-Gaussian predictive).
  nis_surprise      Normalized Innovation Squared off the local-linear-trend Kalman we already run
                    (src.curve_signals.kalman_local_linear_trend): e_t = source - filtered level,
                    nis_t = e_t^2 / rolling-var(e); flagged against a 1-dof chi-square band. Nearly free.
  regime_prob       FILTERED (causal, never smoothed) marginal probability of the crowded/high regime
                    from a 2-3 state statsmodels MarkovRegression (switching variance; no `hmmlearn`).
                    Crowded regime = the state with the largest fitted mean (largest variance on a tie).

output=blend (default) rank-averages the three event intensities (causal trailing-window mid-rank —
full-sample rank would be look-ahead) then SIGNS the result by sign(source - Kalman fair value) so
POSITIVE = crowded-long event, NEGATIVE = crowded-short/unwind event. output in
{changepoint_prob, nis_surprise, regime_prob} returns that single series.

Everything is causal (BOCPD is online; the Kalman/HMM use FILTERED state; standardization is
expanding-window). P(changepoint) is guaranteed in [0,1]. The Kalman and HMM are guarded — a
failure yields a NaN series for that component (noted in meta) while the others still produce.

Sign convention matches base: POSITIVE = crowded net LONG duration. Unbounded (bounded=False).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import t as student_t

from . import base, params as P, transforms
from .base import VersionInputs, VersionResult
from src.curve_signals import kalman_local_linear_trend

RANK_W = 156          # trailing window (3y) for the causal mid-rank blend
_CHI2_99_1DOF = float(chi2.ppf(0.99, 1))   # ~6.635 — the NIS surprise band


# --------------------------------------------------------------------------- #
# Bayesian Online Change-Point Detection (Adams-MacKay 2007) — pure-numpy algo.
# NIG conjugate on a Gaussian obs -> Student-t posterior predictive. scipy is used
# only for the t-pdf (already a repo dep); the recursion itself is hand-written.
# --------------------------------------------------------------------------- #
def _bocpd(x: np.ndarray, hazard: float, obs: str = "t", tau: int = 4) -> np.ndarray:
    """Run-length posterior recursion. Returns, per step, the posterior mass at short run
    lengths P(r_t <= tau) in [0,1] — the actionable "regime recently broke" alarm.

    NOTE: the naive P(r_t == 0 | x_{1:t}) is identically the hazard (a known BOCPD identity:
    the changepoint mass H*sum(R*pred) over the evidence sum(R*pred) cancels to H), so it carries
    NO data information. The changepoint signal instead lives in the SHAPE of the run-length
    posterior: after a genuine break the long-run predictives collapse and mass concentrates at
    small run lengths, so P(r_t <= tau) spikes and then decays as the new run matures.

    x should be standardized (~z) so the NIG hyperparameters are scale-free. `obs='t'` uses a
    modest alpha0 (dof=2, fat tails, robust to a single outlier week); `'gaussian'` a larger
    alpha0 (near-Gaussian predictive)."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T == 0:
        return np.array([])

    if obs == "gaussian":
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 10.0, 9.0   # dof0=20, E[sigma^2]=1
    else:  # 't' — robust, fat-tailed predictive
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0    # dof0=2

    H = float(hazard)
    # per-run-length NIG params (grow by one each step: the new run length 0 gets the prior)
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])
    R = np.array([1.0])                                     # run-length distribution
    cp = np.zeros(T)

    for t in range(T):
        xt = x[t]
        # Student-t posterior predictive of xt under each current run length
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        pred = student_t.pdf(xt, df=2.0 * alpha, loc=mu, scale=scale)

        # growth (run length +1) and changepoint (reset to 0)
        new = np.empty(len(R) + 1)
        new[1:] = R * pred * (1.0 - H)
        new[0] = np.sum(R * pred * H)
        s = new.sum()
        if not np.isfinite(s) or s <= 0.0:                 # numerical guard -> treat as a break
            new = np.zeros(len(R) + 1); new[0] = 1.0; s = 1.0
        new /= s
        R = new
        cp[t] = float(R[:min(tau + 1, len(R))].sum())      # P(r_t <= tau): recent-break mass

        # posterior update: prepend the prior (fresh run length 0), update the survivors by xt
        mu_u = (kappa * mu + xt) / (kappa + 1.0)
        beta_u = beta + (kappa * (xt - mu) ** 2) / (2.0 * (kappa + 1.0))
        mu = np.concatenate(([mu0], mu_u))
        kappa = np.concatenate(([kappa0], kappa + 1.0))
        alpha = np.concatenate(([alpha0], alpha + 0.5))
        beta = np.concatenate(([beta0], beta_u))

    return cp


def _trailing_midrank(s: pd.Series, window: int = RANK_W, min_periods: int = 26) -> pd.Series:
    """Causal percentile of the latest value within its trailing window (mid-rank for ties).
    A full-sample .rank(pct=True) would peek ahead; this only ever looks back."""
    def f(a):
        last = a[-1]
        if not np.isfinite(last):
            return np.nan
        a = a[np.isfinite(a)]
        n = len(a)
        if n == 0:
            return np.nan
        return (np.sum(a < last) + 0.5 * np.sum(a == last)) / n
    return s.rolling(window, min_periods=min_periods).apply(f, raw=True)


def _latest(s: pd.Series):
    s = s.dropna()
    return None if s.empty else float(s.iloc[-1])


# --------------------------------------------------------------------------- #
def _resolve_source(inp: VersionInputs, p: dict) -> tuple[pd.Series, str]:
    """Series the overlay monitors + a note about any fallback. v3/v6 are lazily imported and
    guarded so V8 never hard-depends on another version file (they may build concurrently)."""
    src = p["source"]
    note = ""
    if src == "v0":
        return inp.prod_composite.rename("v0"), note
    if src in ("v3", "v6"):
        try:
            if src == "v3":
                from . import v3_ewma_52w as _m
            else:
                from . import v6_roofing as _m
            return _m.build(inp).composite.rename(src), note
        except Exception as e:                              # pragma: no cover - defensive
            note = f"source={src} failed ({type(e).__name__}); fell back to raw signal. "
    return P.resolve_signal(inp, p), note


def _hmm_regime(sig_clean: pd.Series, k: int) -> dict:
    """Fit a k-state MarkovRegression (switching variance) and return the FILTERED marginal
    probability of the crowded (largest-mean) regime, plus diagnostics. Never raises."""
    out = {"regime_prob": pd.Series(np.nan, index=sig_clean.index, name="regime_prob"),
           "converged": False, "means": None, "sigmas": None, "crowded_idx": None,
           "note": ""}
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MarkovRegression(sig_clean.astype(float), k_regimes=int(k),
                                   trend="c", switching_variance=True)
            res = mod.fit(disp=False)
        means = np.array([float(res.params.get(f"const[{i}]", np.nan)) for i in range(k)])
        sigmas = np.array([float(res.params.get(f"sigma2[{i}]", np.nan)) for i in range(k)])
        # crowded = largest fitted mean; on a (near) tie use the largest variance
        if np.all(np.isfinite(means)) and (means.max() - means.min()) > 1e-8:
            idx = int(np.argmax(means))
        else:
            idx = int(np.nanargmax(sigmas))
        fmp = res.filtered_marginal_probabilities
        if hasattr(fmp, "iloc"):
            rp = fmp.iloc[:, idx]
        else:
            rp = pd.Series(np.asarray(fmp)[:, idx], index=sig_clean.index)
        rp = rp.reindex(sig_clean.index).rename("regime_prob")
        converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
        out.update({"regime_prob": rp, "converged": converged,
                    "means": means.tolist(), "sigmas": sigmas.tolist(), "crowded_idx": idx,
                    "note": "" if converged else "HMM did not fully converge (probs still usable). "})
    except Exception as e:
        out["note"] = f"HMM failed ({type(e).__name__}: {e}); regime_prob=NaN. "
    return out


# --------------------------------------------------------------------------- #
def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v8", params)
    notes: list[str] = []

    # 1) resolve + normalize the source
    src, snote = _resolve_source(inp, p)
    if snote:
        notes.append(snote)
    sig = transforms.apply_norm(src.astype(float), p["norm_mode"])
    sig_clean = sig.dropna().astype(float)
    idx = sig_clean.index

    if len(sig_clean) < 5:                                  # degenerate — return NaN gracefully
        nan_s = pd.Series(np.nan, index=idx, name="v8")
        return VersionResult(name="V8 Change-point / surprise overlay",
                             description="Change-point / surprise event overlay (BOCPD + NIS + HMM).",
                             composite=nan_s, bounded=False, curve=nan_s,
                             meta={"note": "insufficient data", "params": p})

    # 2) BOCPD change-point probability on the scale-free source.
    #    Standardize with a causal burn-in center/scale (robust median/MAD of the earliest
    #    ~52 weeks) so the NIG prior (mu0=0, unit-ish variance) is seated correctly. We do NOT
    #    subtract a rolling mean — that would erase exactly the level shifts BOCPD must detect.
    #    Level shifts AFTER the burn-in are preserved and detectable; the constants use only the
    #    earliest data, so it stays causal.
    nb = min(max(len(sig_clean) // 4, 26), 52)
    burn = sig_clean.iloc[:nb]
    c0 = float(burn.median())
    d0 = float(1.4826 * (burn - burn.median()).abs().median())
    if not np.isfinite(d0) or d0 <= 0.0:
        d0 = float(burn.std(ddof=0)) or 1.0
    z = (sig_clean - c0) / d0
    cp = _bocpd(z.to_numpy(), hazard=float(p["bocpd_hazard"]), obs=p["bocpd_obs"])
    changepoint_prob = pd.Series(cp, index=idx, name="changepoint_prob").clip(0.0, 1.0)

    # 3) NIS surprise off the local-linear-trend Kalman (guarded — needs >=30 pts)
    resid = None
    nis = pd.Series(np.nan, index=idx, name="nis_surprise")
    nis_flag = pd.Series(False, index=idx)
    try:
        lt = kalman_local_linear_trend(sig_clean)
        src_scale = float(np.nanstd(sig_clean.values)) or 1.0
        # Guard the degenerate perfect-fit: on a smooth source the MLE can collapse
        # sigma2.irregular -> 0 so the FILTERED level tracks the data exactly and resid ~ 0,
        # which would kill the NIS surprise and the blend sign. Fall back to a smooth
        # fixed-variance preset (as V5) so the innovation is a real deviation-from-fair-value.
        if float(np.nanstd(lt.resid.values)) < 1e-6 * src_scale:
            V = float(np.nanvar(sig_clean.values, ddof=0)) or 1.0
            lt = kalman_local_linear_trend(sig_clean, fixed_params={
                "sigma2.irregular": 0.30 * V, "sigma2.level": 0.02 * V, "sigma2.trend": 5e-4 * V})
            notes.append("Kalman MLE collapsed to a perfect fit; used a smooth fixed-variance "
                         "preset so the NIS innovation is non-degenerate. ")
        resid = lt.resid.reindex(idx)
        e = resid
        w = int(p["nis_window"])
        S = e.rolling(w, min_periods=max(w // 2, 10)).var(ddof=0).replace(0.0, np.nan)
        nis = (e ** 2 / S).rename("nis_surprise")
        nis_flag = nis > _CHI2_99_1DOF
    except Exception as ex:
        notes.append(f"Kalman/NIS failed ({type(ex).__name__}); nis_surprise=NaN. ")

    # 4) HMM filtered regime probability (guarded)
    hmm = _hmm_regime(sig_clean, int(p["hmm_states"]))
    regime_prob = hmm["regime_prob"]
    if hmm["note"]:
        notes.append(hmm["note"])

    # 5) blend: causal trailing mid-rank of the three intensities, signed by dev-from-fair-value.
    #    (Rank is invariant to the earlier z-standardization, so we rank the raw intensities.)
    ranks = pd.concat([
        _trailing_midrank(changepoint_prob, RANK_W),
        _trailing_midrank(nis.reindex(idx), RANK_W),
        _trailing_midrank(regime_prob.reindex(idx), RANK_W),
    ], axis=1)
    intensity = ranks.mean(axis=1, skipna=True)            # mean of whatever components exist
    if resid is not None:
        sgn = np.sign(resid.reindex(idx).fillna(0.0))
    else:                                                  # Kalman failed — fall back to dev vs MA
        dev = sig_clean - sig_clean.rolling(52, min_periods=13).mean()
        sgn = np.sign(dev.fillna(0.0))
    blend = (intensity * sgn).rename("v8")

    outputs = {
        "blend": blend,
        "changepoint_prob": changepoint_prob.rename("v8"),
        "nis_surprise": nis.rename("v8"),
        "regime_prob": regime_prob.rename("v8"),
    }
    composite = outputs[p["output"]]

    meta = {
        "output": p["output"],
        "source": p["source"],
        "bocpd_hazard": float(p["bocpd_hazard"]),
        "bocpd_obs": p["bocpd_obs"],
        "changepoint_prob_latest": _latest(changepoint_prob),
        "nis_surprise_latest": _latest(nis),
        "nis_flag_latest": bool(nis_flag.dropna().iloc[-1]) if nis_flag.dropna().size else None,
        "regime_prob_latest": _latest(regime_prob),
        "blend_latest": _latest(blend),
        "chi2_threshold_99_1dof": _CHI2_99_1DOF,
        "hmm_states": int(p["hmm_states"]),
        "hmm_converged": bool(hmm["converged"]),
        "hmm_regime_means": hmm["means"],
        "hmm_regime_variances": hmm["sigmas"],
        "hmm_crowded_idx": hmm["crowded_idx"],
        "n_finite": {"changepoint_prob": int(changepoint_prob.notna().sum()),
                     "nis_surprise": int(nis.notna().sum()),
                     "regime_prob": int(regime_prob.notna().sum()),
                     "blend": int(blend.notna().sum())},
        "note": ("EVENT signals (discrete), not added variance: BOCPD P(changepoint) in [0,1], "
                 "1-dof chi-square NIS surprise off the local-linear-trend Kalman, and the HMM "
                 "FILTERED (causal) crowded-regime probability. blend = signed causal-rank average; "
                 "sign(+)=crowded-long event. " + "".join(notes)).strip(),
    }

    return VersionResult(
        name="V8 Change-point / surprise overlay",
        description="BOCPD P(changepoint) + 1-dof NIS surprise + HMM filtered crowded-regime prob, "
                    "blended into a signed causal event overlay (discrete events, not added variance).",
        composite=composite,
        bounded=False,
        curve=changepoint_prob,
        meta=meta,
    )


if __name__ == "__main__":
    from .base import load_inputs

    inp = load_inputs()
    res = build(inp)                                        # defaults: output=blend, obs=t, 2 states
    m = res.meta
    print(res.name)
    print("description:", res.description)
    print("-- defaults (blend) --")
    print("  latest changepoint_prob:", m["changepoint_prob_latest"])
    print("  latest nis_surprise    :", m["nis_surprise_latest"], "(flag>chi2:", m["nis_flag_latest"], ")")
    print("  latest regime_prob     :", m["regime_prob_latest"])
    print("  latest blend           :", m["blend_latest"])
    print("  n finite               :", m["n_finite"])
    print("  chi2 threshold (99,1)  :", round(m["chi2_threshold_99_1dof"], 4))
    print("  HMM converged          :", m["hmm_converged"], " means:", m["hmm_regime_means"],
          " crowded_idx:", m["hmm_crowded_idx"])

    cpp = res.curve.dropna()
    assert ((cpp >= 0.0) & (cpp <= 1.0)).all(), "changepoint_prob out of [0,1]!"
    print("  assert 0<=changepoint_prob<=1 everywhere finite: PASS")

    # biggest BOCPD spikes (skip the first ~8wk run-in: run length is trivially small at series
    # start, so those weeks are P(cp)~1 by construction, not real events). Sanity: expect COVID 2020.
    top = cpp.iloc[8:].sort_values(ascending=False).head(8)
    print("  top-8 BOCPD change-point weeks (excl. init run-in):")
    for dt, v in top.items():
        print(f"     {dt.date()}  P(cp)={v:.3f}")

    for ov in ({"output": "changepoint_prob"}, {"output": "regime_prob"},
               {"bocpd_obs": "gaussian"}, {"hmm_states": 3}):
        r = build(inp, ov)
        c = r.composite.dropna()
        lab = ",".join(f"{k}={v}" for k, v in ov.items())
        print(f"-- {lab:24s} composite finite={c.size:4d} latest="
              f"{(None if c.empty else round(float(c.iloc[-1]),4))}"
              f"  hmm_conv={r.meta['hmm_converged']}")
