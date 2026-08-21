"""
Discrete-time issuance hazard: P(issuer issues in a given week | covariates).

Link = complementary log-log, which is the grouped-interval form of a
continuous exponential / Cox proportional-hazard -- so the weekly hazard
aggregates cleanly to any horizon via the survival product S(k)=prod(1-h_j).

TimingModel stores only the fitted coefficients + the DesignSpec, so it pickles
cheaply and scores future weeks by rebuilding the identical design matrix.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import build_design, design_from_spec, DesignSpec, penalized_mask


def cloglog_inv(eta: np.ndarray) -> np.ndarray:
    """Inverse cloglog link: hazard = 1 - exp(-exp(eta))."""
    return 1.0 - np.exp(-np.exp(np.clip(eta, -30, 20)))


@dataclass
class TimingModel:
    params: pd.Series
    spec: DesignSpec
    llf: float = np.nan
    n_obs: int = 0
    bse: pd.Series = None

    def predict_hazard(self, df: pd.DataFrame, chunk: int = 100_000) -> np.ndarray:
        """Weekly hazard for each row; scored in chunks to bound memory."""
        out = np.empty(len(df), dtype=float)
        beta = None
        for start in range(0, len(df), chunk):
            block = df.iloc[start:start + chunk]
            X = design_from_spec(block, self.spec)
            if beta is None:
                beta = self.params.reindex(X.columns).values
            out[start:start + len(block)] = cloglog_inv(X.values @ beta)
        return out


def fit_timing_model(grid: pd.DataFrame, max_rows: int = 250_000, seed: int = 0) -> TimingModel:
    """Fit the cloglog hazard on the issuer-week panel.

    Rare-event handling: keep every issuance week (Y=1) and inverse-probability
    weight a random sub-sample of the non-issuance weeks.  IPW MLE is consistent
    for *all* coefficients (not just the intercept) and keeps the dense IRLS
    design small enough to fit in memory.
    """
    pos = grid[grid["Y"] == 1]
    neg = grid[grid["Y"] == 0]
    n_neg = min(len(neg), max(max_rows - len(pos), len(pos)))
    tau = n_neg / len(neg)                                   # control inclusion prob
    neg_s = neg.sample(n=n_neg, random_state=seed)
    sub = pd.concat([pos, neg_s]).sort_index()
    w = np.where(sub["Y"].values == 1, 1.0, 1.0 / tau)      # IPW: 1/inclusion-prob

    y, X, spec = build_design(sub)
    fam = sm.families.Binomial(link=sm.families.links.CLogLog())
    # Ridge only the many categorical dummies (they cause quasi-separation and
    # non-convergence); leave the intercept + core hazard terms unpenalized so
    # the base rate stays calibrated.
    alpha = np.where(penalized_mask(list(X.columns)), 1e-3, 0.0)
    model = sm.GLM(y, X, family=fam, freq_weights=w)
    res = model.fit_regularized(alpha=alpha, L1_wt=0.0)
    params = pd.Series(np.asarray(res.params), index=X.columns)
    return TimingModel(params=params, spec=spec, llf=np.nan, n_obs=len(grid), bse=None)


if __name__ == "__main__":
    from panel import build_panel
    grid, _, _ = build_panel()
    tm = fit_timing_model(grid)
    samp = grid.sample(n=200_000, random_state=1)
    h = tm.predict_hazard(samp)
    print("fitted hazard  mean:", round(h.mean(), 4), " actual Y:", round(samp["Y"].mean(), 4))
    print("n_obs:", f"{tm.n_obs:,}")
    # top duration / frequency coefficients (sanity: negative on log_wsl, positive on log_trail)
    for k in ["log_wsl", "log_trail", "rating_filled", "quarter_end"]:
        print(f"  beta[{k}] = {tm.params.get(k, float('nan')):+.3f}")
    # calibration by predicted-decile
    d = pd.DataFrame({"y": samp["Y"].values, "h": h})
    d["bin"] = pd.qcut(d["h"], 10, duplicates="drop")
    print("\ncalibration (pred vs actual) by hazard decile:")
    print(d.groupby("bin", observed=True).agg(pred=("h", "mean"), actual=("y", "mean"),
                                              n=("y", "size")).round(4).to_string())
