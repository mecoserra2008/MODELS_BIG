"""Signal-richness scorecard + STRICT out-of-sample acceptance gate (research angle 5 = the rubric).

This is the backbone that makes "richer" falsifiable. A candidate signal is accepted over the V0
baseline ONLY if it produces more informative deviations per unit time — i.e. richness metrics up
AND the *incremental* deviations (ones the candidate has that V0 lacks) carry forward-yield IC that
survives OOS AND beats an AR(1)/phase-randomized surrogate AND clears the overfitting guards.

Metric families
  richness  : threshold-crossing rate (0,±1,±2σ, annualized), turning-point density + Bienaymé
              z-test (E[p]=⅔(n−2), var=(16n−29)/90), Rice's-formula expected-crossing budget.
  structure : OU half-life (curve_signals.ou_half_life), spectral entropy (scipy periodogram),
              effective sample size n_eff=n(1−ρ1)/(1+ρ1).
  info      : mean IC=spearman(signal, fwd Δy) + Newey–West t; mutual information; turnover.
  gate      : incremental-IC of the added deviations; AR(1)/IAAFT surrogate p; CPCV purged/embargo
              OOS IC; Deflated Sharpe / PBO; t>3 hurdle. -> pass/fail + reasons.

NOTE: frozen-signature STUB written by the orchestrator so notebooks + v9 import immediately; the
richness agent fills in the bodies. Stubs return NaN/empty and never raise.

Sign/scale: a signal is standardized to ~unit σ for crossing counts via `to_z` (bounded [0,100] ->
(x-50)/25). Everything is publication-lagged (base.pub_lag) before aligning to yields — no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import base

HORIZONS = (4, 8, 13)
_WK_PER_YR = 52.0


# --------------------------------------------------------------------------- #
def to_z(s: pd.Series, bounded: bool) -> pd.Series:
    """Standardize to ~unit scale for comparable crossing counts (mirrors predict._to_z)."""
    x = s.dropna().astype(float)
    if bounded:
        return (x - 50.0) / 25.0
    sd = float(x.std(ddof=0))
    return x / sd if sd and abs(sd - 1.0) > 0.1 else x


# --------------------------------------------------------------------------- #
# Small private helpers (kept terse; numpy/pandas only + lazy scipy/sklearn).   #
# --------------------------------------------------------------------------- #
def _sign_changes(x: np.ndarray) -> int:
    """Number of genuine sign changes (crossings) in a 1-D array — consecutive opposite signs."""
    s = np.sign(np.asarray(x, dtype=float))
    if len(s) < 2:
        return 0
    return int(np.sum(s[:-1] * s[1:] < 0))


def _align_to_y(z: pd.Series, y10_w: pd.Series) -> pd.Series:
    """Publication-lag then as-of align onto the weekly y10 grid (EXACT predict.py pattern)."""
    z_pub = base.pub_lag(z.dropna())
    return (z_pub.reindex(y10_w.index.union(z_pub.index)).ffill().reindex(y10_w.index))


def _spearman(a, b) -> float:
    """Spearman rank correlation; NaN when degenerate."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    r = spearmanr(a, b)[0]
    return float(r) if r == r else float("nan")


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """MI(x->y) via sklearn kNN estimator; NaN-guarded, fixed seed."""
    from sklearn.feature_selection import mutual_info_regression
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    try:
        return float(mutual_info_regression(x[m].reshape(-1, 1), y[m], random_state=0)[0])
    except Exception:
        return float("nan")


def _iaaft_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized (IAAFT-style) surrogate: rfft -> random phases, same magnitudes -> irfft.
    Preserves the power spectrum (hence autocorrelation) but destroys phase structure."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = x.mean()
    F = np.fft.rfft(x - mu)
    mag = np.abs(F)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(F))
    phases[0] = 0.0                         # keep DC real (mean added back)
    if n % 2 == 0:
        phases[-1] = 0.0                    # keep Nyquist real
    xs = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return xs + mu


def _ar1_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AR(1) surrogate with matched lag-1 autocorr and residual std (spectrum-matched null)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    sd = float(np.std(xc))
    r1 = float(np.corrcoef(xc[:-1], xc[1:])[0, 1]) if n > 2 and sd > 0 else 0.0
    if not (r1 == r1):
        r1 = 0.0
    resid_sd = sd * np.sqrt(max(1e-12, 1.0 - r1 * r1))
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd if sd > 0 else 1.0)
    e = rng.normal(0.0, resid_sd, size=n)
    for t in range(1, n):
        out[t] = r1 * out[t - 1] + e[t]
    return out + x.mean()


# --------------------------------------------------------------------------- #
def crossing_metrics(z: pd.Series, levels=(0.0, 1.0, 2.0)) -> dict:
    """Annualized threshold-crossing rate at each ±level + turning-point density & Bienaymé z."""
    x = z.dropna().astype(float).values
    n = len(x)
    if n < 3:
        return {"crossings_per_yr": {int(L): float("nan") for L in levels},
                "turns_per_yr": float("nan"), "n_turns": 0, "bienayme_z": float("nan"), "n": int(n)}
    yrs = n / _WK_PER_YR
    cpy: dict = {}
    for L in levels:
        L = float(L)
        if L == 0.0:
            c = _sign_changes(x)                                   # zero-crossings
        else:
            c = _sign_changes(x - L) + _sign_changes(x + L)        # through +L and −L
        cpy[int(L)] = float(c) / yrs
    d = np.diff(x)
    n_turns = _sign_changes(d)                                     # local max+min = sign flips of Δ
    turns_per_yr = float(n_turns) / yrs
    # Bienaymé–Kendall turning-point randomness test.
    E = (2.0 / 3.0) * (n - 2)
    var = (16.0 * n - 29.0) / 90.0
    bz = float((n_turns - E) / np.sqrt(var)) if var > 0 else float("nan")
    return {"crossings_per_yr": cpy, "turns_per_yr": turns_per_yr,
            "n_turns": int(n_turns), "bienayme_z": bz, "n": int(n)}


def rice_budget(z: pd.Series, level: float = 2.0) -> float:
    """Rice's-formula expected ±level up-crossings/yr for the series' own fitted spectrum."""
    x = z.dropna().astype(float).values
    if len(x) < 3:
        return float("nan")
    mu = float(x.mean())
    lam0 = float(np.var(x, ddof=0))          # spectral moment 0 = variance of level
    lam2 = float(np.var(np.diff(x), ddof=0))  # spectral moment 2 ~ variance of increments
    if lam0 <= 0:
        return float("nan")
    rate = (1.0 / (2.0 * np.pi)) * np.sqrt(lam2 / lam0) * np.exp(-((level - mu) ** 2) / (2.0 * lam0))
    return float(rate * _WK_PER_YR)


def structure_metrics(z: pd.Series) -> dict:
    """OU half-life (weeks), spectral entropy [0,1], effective sample size."""
    from src.curve_signals import ou_half_life
    from scipy.signal import periodogram
    x = z.dropna().astype(float)
    n = len(x)
    hl = float(ou_half_life(x))
    # Spectral entropy: Shannon entropy of the normalized PSD / log(#freqs); ~1 => white.
    H = float("nan")
    if n >= 8:
        _, pxx = periodogram(x.values, detrend="constant")
        pxx = pxx[1:]                         # drop the DC bin (removed by detrend anyway)
        tot = float(pxx.sum())
        if tot > 0 and len(pxx) > 1:
            p = pxx / tot
            pnz = p[p > 0]
            H = float(-(pnz * np.log(pnz)).sum() / np.log(len(pxx)))
    rho1 = float(x.autocorr(1)) if n > 2 else float("nan")
    if rho1 == rho1 and rho1 < 1.0 and (1.0 + rho1) != 0.0:
        neff = float(min(max(n * (1.0 - rho1) / (1.0 + rho1), 2.0), float(n)))
    else:
        neff = float("nan")
    return {"ou_half_life_wk": hl, "spectral_entropy": H, "n_eff": neff}


def ic_metrics(signal: pd.Series, y10_w: pd.Series, horizons=HORIZONS,
               bounded: bool = False) -> dict:
    """Per-horizon Spearman IC of signal vs forward Δ10Y (bp) + n_eff t + mutual information."""
    z_on_y = _align_to_y(to_z(signal, bounded), y10_w)
    out: dict = {}
    for H in horizons:
        H = int(H)
        fwd = (y10_w.shift(-H) - y10_w) * 100.0
        d = pd.concat([z_on_y.rename("z"), fwd.rename("fwd")], axis=1).dropna()
        if len(d) < 10 or d["z"].std(ddof=0) == 0:
            out[str(H)] = {"ic": float("nan"), "t": float("nan"), "mi": float("nan"), "n": int(len(d))}
            continue
        ic = _spearman(d["z"].values, d["fwd"].values)
        neff = structure_metrics(d["z"])["n_eff"]               # honest n from the aligned z
        if ic == ic and neff == neff and neff > 2 and abs(ic) < 1.0:
            t = float(ic * np.sqrt((neff - 2.0) / (1.0 - ic * ic)))
        else:
            t = float("nan")
        mi = _mutual_info(d["z"].values, d["fwd"].values)
        out[str(H)] = {"ic": float(ic), "t": t, "mi": float(mi), "n": int(len(d))}
    return out


def incremental_ic(cand: pd.Series, baseline: pd.Series, y10_w: pd.Series,
                   horizon: int = 8, bounded: bool = False) -> dict:
    """IC on the ADDED-deviation subset: weeks where `cand` crosses a band but `baseline` does not.
    If ~0, the extra richness is noise by construction."""
    H = int(horizon)
    zc = _align_to_y(to_z(cand, bounded), y10_w)
    zb = _align_to_y(to_z(baseline, False), y10_w)               # baseline treated as z-family
    fwd = (y10_w.shift(-H) - y10_w) * 100.0
    d = pd.concat([zc.rename("zc"), zb.rename("zb"), fwd.rename("fwd")], axis=1).dropna()
    # Added deviations = candidate flags an extreme (|z|>1) while baseline is quiet (|z|<1).
    mask = (d["zc"].abs() > 1.0) & (d["zb"].abs() < 1.0)
    sub = d[mask]
    ns = int(len(sub))
    if ns < 20:
        return {"ic": float("nan"), "n_subset": ns, "hit_rate": float("nan"),
                "note": "insufficient added-deviation sample"}
    ic = _spearman(sub["zc"].values, sub["fwd"].values)
    hit = float(np.mean(np.sign(sub["zc"].values) == np.sign(sub["fwd"].values)))
    return {"ic": float(ic), "n_subset": ns, "hit_rate": hit, "note": "ok"}


def surrogate_test(cand: pd.Series, baseline: pd.Series, y10_w: pd.Series, horizon: int = 8,
                   bounded: bool = False, n: int = 200, kind: str = "iaaft") -> dict:
    """Empirical p that the added-deviation IC beats AR(1)/phase-randomized (IAAFT) surrogates."""
    obs = incremental_ic(cand, baseline, y10_w, horizon, bounded)["ic"]
    res = {"p": float("nan"), "observed_ic": float(obs) if obs == obs else float("nan"),
           "surrogate_mean": float("nan"), "kind": kind, "n": int(n)}
    if not (obs == obs):
        return res
    raw = cand.dropna().astype(float)
    if len(raw) < 20:
        return res
    rng = np.random.default_rng(12345)                          # fixed seed -> reproducible
    gen = _ar1_surrogate if kind == "ar1" else _iaaft_surrogate
    absics = []
    for _ in range(int(n)):
        ss = pd.Series(gen(raw.values, rng), index=raw.index, name=raw.name)
        r = incremental_ic(ss, baseline, y10_w, horizon, bounded)["ic"]
        if r == r:
            absics.append(abs(r))
    if not absics:
        return res
    absics = np.asarray(absics)
    res["p"] = float(np.mean(absics >= abs(obs)))
    res["surrogate_mean"] = float(absics.mean())               # mean |IC| under the matched null
    return res


def cpcv_ic(signal: pd.Series, y10_w: pd.Series, horizon: int = 8, bounded: bool = False,
            n_splits: int = 6, k_test: int = 2) -> dict:
    """Combinatorial purged CV IC with embargo=horizon (reuse base.pub_lag alignment)."""
    from itertools import combinations
    H = int(horizon)
    z_on_y = _align_to_y(to_z(signal, bounded), y10_w)
    fwd = (y10_w.shift(-H) - y10_w) * 100.0
    d = pd.concat([z_on_y.rename("z"), fwd.rename("fwd")], axis=1).dropna().reset_index(drop=True)
    N = len(d)
    if N < 3 * n_splits:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "frac_same_sign": float("nan"), "n_paths": 0}
    blocks = np.array_split(np.arange(N), n_splits)
    full_ic = _spearman(d["z"].values, d["fwd"].values)
    ics = []
    for combo in combinations(range(n_splits), k_test):
        idx: list = []
        for b in combo:
            blk = blocks[b]
            if len(blk) > 2 * H:                                # embargo H at each edge (purge labels)
                idx.extend(blk[H: len(blk) - H].tolist())
        if len(idx) < 10:
            continue
        sub = d.iloc[idx]
        if sub["z"].std(ddof=0) == 0:
            continue
        ic = _spearman(sub["z"].values, sub["fwd"].values)
        if ic == ic:
            ics.append(ic)
    if not ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "frac_same_sign": float("nan"), "n_paths": 0}
    ics = np.asarray(ics)
    fss = (float(np.mean(np.sign(ics) == np.sign(full_ic))) if full_ic == full_ic else float("nan"))
    return {"ic_mean": float(ics.mean()), "ic_std": float(ics.std(ddof=0)),
            "frac_same_sign": fss, "n_paths": int(len(ics))}


def deflated_sharpe(sr: float, n_trials: int, n_obs: int, skew: float = 0.0,
                    kurt: float = 3.0) -> dict:
    """Deflated Sharpe / PSR given how many variants were tried (E[max SR] under the null)."""
    from scipy.stats import norm
    gamma = 0.5772156649015329                                 # Euler–Mascheroni
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr    # PSR variance shape factor
    n_trials = int(n_trials)
    if n_obs is None or n_obs < 2:
        return {"dsr": float("nan"), "sr0_expected_max": float("nan"), "n_trials": n_trials}
    if n_trials < 2:
        e_max = 0.0                                            # nothing to deflate for
    else:
        var_sr = max((1.0 / n_obs) * denom, 1e-12)
        e_max = float(np.sqrt(var_sr) * ((1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
                                         + gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))))
    if denom <= 0:
        return {"dsr": float("nan"), "sr0_expected_max": float(e_max), "n_trials": n_trials}
    dsr = float(norm.cdf((sr - e_max) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)))
    return {"dsr": dsr, "sr0_expected_max": float(e_max), "n_trials": n_trials}


def pbo_cscv(returns_matrix: pd.DataFrame) -> float:
    """Probability of backtest overfitting via CSCV (fraction of OOS ranks below median)."""
    from itertools import combinations
    R = returns_matrix.dropna(how="any")
    M = R.shape[1]
    N = R.shape[0]
    if M < 2 or N < 8:
        return float("nan")
    S = min(10, N)                                             # even # of row-blocks
    if S % 2 == 1:
        S -= 1
    if S < 2:
        return float("nan")
    blocks = np.array_split(np.arange(N), S)
    lambdas = []
    for is_combo in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in is_combo])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in is_combo])
        R_is, R_oos = R.iloc[is_rows], R.iloc[oos_rows]
        sh_is = R_is.mean() / R_is.std(ddof=0).replace(0.0, np.nan)
        sh_oos = R_oos.mean() / R_oos.std(ddof=0).replace(0.0, np.nan)
        if sh_is.dropna().empty or sh_oos.dropna().empty:
            continue
        best = sh_is.idxmax()
        rank = sh_oos.rank(method="average")[best]             # 1..M (higher=better OOS)
        omega = float(rank) / (M + 1.0)                        # relative OOS rank in (0,1)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        lambdas.append(np.log(omega / (1.0 - omega)))          # logit
    if not lambdas:
        return float("nan")
    return float(np.mean(np.asarray(lambdas) < 0.0))           # PBO = P(IS-best underperforms OOS)


def score(composite: pd.Series, y10_w: pd.Series, bounded: bool = False,
          baseline: pd.Series | None = None, horizons=HORIZONS, name: str = "cand") -> dict:
    """Full scorecard for one signal: {name, richness, structure, info, gate}.
    `baseline` (V0 composite) enables the incremental-IC / surrogate gate; omit for baseline itself."""
    out: dict = {"name": name}
    try:
        z = to_z(composite, bounded)
    except Exception as e:                                     # pragma: no cover
        return {"name": name, "richness": {"_error": str(e)}, "structure": {"_error": str(e)},
                "info": {"_error": str(e)}, "gate": None}
    # Each sub-metric is isolated: on failure store nan/None + an '_error' string, never raise.
    try:
        rich = dict(crossing_metrics(z))
        rich["rice_2σ_per_yr"] = rice_budget(z, 2.0)
        out["richness"] = rich
    except Exception as e:                                     # pragma: no cover
        out["richness"] = {"_error": str(e)}
    try:
        out["structure"] = structure_metrics(z)
    except Exception as e:                                     # pragma: no cover
        out["structure"] = {"_error": str(e)}
    try:
        out["info"] = ic_metrics(composite, y10_w, horizons, bounded)
    except Exception as e:                                     # pragma: no cover
        out["info"] = {"_error": str(e)}
    if baseline is not None:
        try:
            out["gate"] = gate(composite, baseline, y10_w, bounded=bounded)
        except Exception as e:                                 # pragma: no cover
            out["gate"] = {"pass": False, "reasons": [f"gate error: {e}"], "_error": str(e)}
    else:
        out["gate"] = None
    return out


def gate(cand: pd.Series, baseline: pd.Series, y10_w: pd.Series, horizon: int = 8,
         bounded: bool = False, n_trials: int = 1) -> dict:
    """Strict OOS decision. Returns {pass: bool, reasons: [...], incremental_ic, surrogate_p,
    cpcv_ic, dsr, pbo, t}. pass requires: incremental IC>0 & surrogate_p<0.05 & cpcv_ic>0 &
    dsr>0 & pbo<0.5 & |t|>3."""
    H = int(horizon)
    inc = incremental_ic(cand, baseline, y10_w, H, bounded)
    inc_ic = inc.get("ic", float("nan"))
    sur = surrogate_test(cand, baseline, y10_w, H, bounded)
    sp = sur.get("p", float("nan"))
    cp = cpcv_ic(cand, y10_w, H, bounded)
    cpm = cp.get("ic_mean", float("nan"))
    im = ic_metrics(cand, y10_w, (H,), bounded).get(str(H), {})
    t = im.get("t", float("nan"))
    # Deflated Sharpe only when a trial count is supplied (>1); otherwise skip that clause.
    dsr = float("nan")
    if n_trials and n_trials > 1:
        ic8, naln = im.get("ic", float("nan")), im.get("n", 0)
        if ic8 == ic8 and naln > 2:
            dsr = deflated_sharpe(ic8 * np.sqrt(_WK_PER_YR), n_trials, naln)["dsr"]  # IC->pseudo-SR

    c_inc = (inc_ic == inc_ic) and (inc_ic > 0.0)
    c_sur = (sp == sp) and (sp < 0.05)
    c_cp = (cpm == cpm) and (cpm > 0.0)
    c_t = (t == t) and (abs(t) > 3.0)
    c_dsr = (not (dsr == dsr)) or (dsr > 0.0)                  # pass if not computed, else require >0
    passed = bool(c_inc and c_sur and c_cp and c_t and c_dsr)

    reasons = []
    if not c_inc:
        reasons.append(f"incremental IC not positive ({inc_ic:.3f}); "
                       f"{inc.get('note', '')} (n_subset={inc.get('n_subset', 0)})")
    if not c_sur:
        reasons.append(f"added-IC does not beat surrogates (p={sp:.3f} ≥ 0.05, kind={sur.get('kind')})")
    if not c_cp:
        reasons.append(f"CPCV OOS IC not positive (mean={cpm:.3f}, n_paths={cp.get('n_paths', 0)})")
    if not c_t:
        reasons.append(f"|t| below hurdle ({t:.2f} ≤ 3, n_eff-adjusted)")
    if (dsr == dsr) and not (dsr > 0.0):
        reasons.append(f"deflated Sharpe not positive ({dsr:.3f})")
    if passed:
        reasons.append("all clauses passed")
    return {"pass": passed, "reasons": reasons, "incremental_ic": float(inc_ic),
            "surrogate_p": float(sp), "cpcv_ic": float(cpm), "dsr": float(dsr),
            "pbo": float("nan"), "t": float(t)}     # pbo needs a multi-strategy matrix (pbo_cscv)


def scorecard(results: dict, y10_w: pd.Series, baseline_key: str = "v0",
              horizons=HORIZONS) -> pd.DataFrame:
    """Assemble a per-version scorecard table from {key: VersionResult|composite Series}.
    Columns: crossings/yr @0/1/2σ, turns/yr, Bienaymé z, half-life, spectral entropy, mean IC(8w),
    IC t, incremental IC, surrogate p, cpcv IC, gate pass. baseline_key is the reference (V0)."""
    cols = ["cross0_per_yr", "cross1_per_yr", "cross2_per_yr", "turns_per_yr", "bienayme_z",
            "half_life_wk", "spectral_entropy", "ic_8w", "ic_t_8w", "incr_ic", "surrogate_p",
            "cpcv_ic", "gate_pass"]

    def _comp(v):
        if hasattr(v, "composite"):
            return v.composite, bool(getattr(v, "bounded", False))
        return v, False

    def _nan_row():
        r = {c: np.nan for c in cols}; r["gate_pass"] = False; return r

    base_comp, _ = _comp(results[baseline_key])
    rows: dict = {}
    for key, v in results.items():
        try:
            comp, bnd = _comp(v)
            if comp is None or comp.dropna().shape[0] < 10:      # stub / empty -> NaN row
                rows[key] = _nan_row(); continue
            s = score(comp, y10_w, bounded=bnd,
                      baseline=(None if key == baseline_key else base_comp),
                      name=key, horizons=horizons)
            r = s.get("richness", {}) or {}
            st = s.get("structure", {}) or {}
            info = s.get("info", {}) or {}
            g = s.get("gate", None)
            cpy = r.get("crossings_per_yr", {}) or {}
            info8 = info.get("8", {}) if isinstance(info, dict) else {}
            rows[key] = {
                "cross0_per_yr": cpy.get(0, np.nan),
                "cross1_per_yr": cpy.get(1, np.nan),
                "cross2_per_yr": cpy.get(2, np.nan),
                "turns_per_yr": r.get("turns_per_yr", np.nan),
                "bienayme_z": r.get("bienayme_z", np.nan),
                "half_life_wk": st.get("ou_half_life_wk", np.nan),
                "spectral_entropy": st.get("spectral_entropy", np.nan),
                "ic_8w": info8.get("ic", np.nan),
                "ic_t_8w": info8.get("t", np.nan),
                "incr_ic": (g.get("incremental_ic", np.nan) if g else np.nan),
                "surrogate_p": (g.get("surrogate_p", np.nan) if g else np.nan),
                "cpcv_ic": (g.get("cpcv_ic", np.nan) if g else np.nan),
                "gate_pass": (bool(g.get("pass", False)) if g else False),
            }
        except Exception:                                        # pragma: no cover - never raise
            rows[key] = _nan_row()
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)


def frontier(scorecard_df: pd.DataFrame) -> pd.DataFrame:
    """Return the (turns/yr, forward hit-rate, gate-pass) points for the efficient-frontier plot.
    hit_rate is a monotone proxy 0.5 + ic_8w/2 (IC in [-1,1] -> hit in [0,1]); it is NOT a measured
    directional hit-rate, only a rank-preserving stand-in for plotting richness vs. informativeness."""
    df = scorecard_df
    return pd.DataFrame({
        "version": list(df.index),
        "turns_per_yr": df["turns_per_yr"].to_numpy(dtype=float),
        "hit_rate": 0.5 + df["ic_8w"].to_numpy(dtype=float) / 2.0,
        "gate_pass": df["gate_pass"].to_numpy(),
    }).reset_index(drop=True)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from positioning.versions import compare, richness

    inp, results = compare.build_all()
    sc = richness.scorecard({k: results[k] for k in ("v0", "v1", "v2", "v3", "v4", "v5")}, inp.y10_w)
    print(sc.to_string())
    s = richness.score(results["v3"].composite, inp.y10_w, bounded=False,
                       baseline=results["v0"].composite, name="v3")
    import json
    print(json.dumps({k: (v if not hasattr(v, "items")
                          else {kk: str(vv)[:40] for kk, vv in v.items()})
                      for k, v in s.items() if k != "info"}, default=str)[:1200])
