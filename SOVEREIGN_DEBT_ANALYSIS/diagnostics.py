"""
Out-of-sample validation and diagnostic figures for the issuance model.

Validation is temporal (train on weeks before the split, test after) because
that is exactly how the model is used: fit on history, predict the future.
We report the per-week hazard's AUC / Brier / calibration AND a horizon-level
calibration (predicted vs realized P(issue within 4 weeks)) -- the quantity the
user actually asks for.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

import config as C
from timing_model import fit_timing_model

SPLIT = "2025-01-01"
INK = "#1f4e79"
ACCENT = "#c0504d"


def temporal_validation(grid: pd.DataFrame, split: str = SPLIT) -> dict:
    train = grid[grid["date"] < split]
    test = grid[grid["date"] >= split].copy()
    tm = fit_timing_model(train)
    test["h"] = tm.predict_hazard(test)

    auc = roc_auc_score(test["Y"], test["h"])
    brier = brier_score_loss(test["Y"], test["h"])

    # per-week calibration deciles
    d = test[["Y", "h"]].copy()
    d["bin"] = pd.qcut(d["h"].rank(method="first"), 10, labels=False)
    cal = d.groupby("bin").agg(pred=("h", "mean"), actual=("Y", "mean"), n=("Y", "size")).reset_index()

    # horizon calibration: predicted vs realized P(issue within next 4 weeks)
    hz = _horizon_calibration(test, horizon=4)

    return {"split": split, "n_train": int(len(train)), "n_test": int(len(test)),
            "auc": float(auc), "brier": float(brier),
            "base_rate_test": float(test["Y"].mean()),
            "calibration": cal, "horizon4": hz, "model": tm}


def _horizon_calibration(test: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """Per issuer-week predicted P(issue within `horizon` weeks) vs realized."""
    t = test.sort_values(["issuer", "wk"]).copy()
    pred, real, keep = [], [], []
    for _, g in t.groupby("issuer"):
        h = g["h"].to_numpy(); y = g["Y"].to_numpy(); n = len(g)
        for i in range(n - horizon):
            s = np.prod(1.0 - h[i + 1:i + 1 + horizon])
            pred.append(1.0 - s)
            real.append(int(y[i + 1:i + 1 + horizon].any()))
            keep.append(True)
    df = pd.DataFrame({"pred": pred, "real": real})
    if len(df) < 50:
        return df
    df["bin"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False)
    return df.groupby("bin").agg(pred=("pred", "mean"), actual=("real", "mean"),
                                 n=("real", "size")).reset_index()


# ---------------------------------------------------------------- figures
def fig_calibration(val: dict, path: str):
    cal, hz = val["calibration"], val["horizon4"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, tab, title in [(ax[0], cal, "Weekly hazard"),
                          (ax[1], hz, "P(issue within 4 weeks)")]:
        lim = max(tab["pred"].max(), tab["actual"].max()) * 1.1
        a.plot([0, lim], [0, lim], "--", color="#999", lw=1)
        a.scatter(tab["pred"], tab["actual"], color=INK, s=40, zorder=3)
        a.set_xlabel("predicted"); a.set_ylabel("realized")
        a.set_title(title); a.grid(alpha=0.25)
    fig.suptitle(f"Out-of-sample calibration (test {val['split']}+, AUC={val['auc']:.3f}, "
                 f"Brier={val['brier']:.4f})", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_hazard_duration(grid: pd.DataFrame, model, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    # duration dependence
    w = pd.cut(grid["weeks_since_last"], [0, 1, 2, 4, 8, 13, 26, 52, 104, 260])
    emp = grid.groupby(w, observed=True)["Y"].mean()
    ax[0].plot([iv.mid for iv in emp.index], emp.values, "o-", color=INK)
    ax[0].set_xlabel("weeks since last issuance"); ax[0].set_ylabel("weekly issuance prob")
    ax[0].set_title("Duration dependence (baseline hazard)"); ax[0].grid(alpha=0.25)
    # trailing frequency dependence
    b = pd.cut(grid["issues_trailing_52w"], [-1, 0, 2, 6, 12, 26, 52, 10000])
    emp2 = grid.groupby(b, observed=True)["Y"].mean()
    ax[1].bar(range(len(emp2)), emp2.values, color=ACCENT)
    ax[1].set_xticks(range(len(emp2)))
    ax[1].set_xticklabels([str(iv) for iv in emp2.index], rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("weekly issuance prob")
    ax[1].set_title("State dependence (issuance in trailing 52w)"); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_seasonality(grid: pd.DataFrame, path: str):
    m = grid.groupby("month")["Y"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(m.index, m.values / grid["Y"].mean(), color=INK)
    ax.axhline(1.0, color="#999", ls="--")
    ax.set_xlabel("calendar month"); ax.set_ylabel("issuance rate / average")
    ax.set_title("Issuance seasonality (relative to average week)")
    ax.set_xticks(range(1, 13)); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_issuer_survival(model, issuers, path: str, horizon: int = 52):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in issuers:
        if name not in model.issuer_static.index:
            continue
        fh = model.forward_hazards(name, horizon)
        ax.plot(fh["week_ahead"], 1.0 - fh["survival"], label=name[:34], lw=2)
    ax.set_xlabel("weeks ahead"); ax.set_ylabel("P(issued at least once by then)")
    ax.set_title("Cumulative issuance probability by horizon")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_all(grid, model, results_dir=C.RESULTS_DIR):
    import os
    val = temporal_validation(grid)
    fig_calibration(val, os.path.join(results_dir, "fig_calibration.png"))
    fig_hazard_duration(grid, model, os.path.join(results_dir, "fig_hazard_duration.png"))
    fig_seasonality(grid, os.path.join(results_dir, "fig_seasonality.png"))
    fig_issuer_survival(model, ["Federal Home Loan Bank System", "The Goldman Sachs Group, Inc.",
                                "Government of Argentina", "Deutsche Bank AG"],
                        os.path.join(results_dir, "fig_issuer_survival.png"))
    metrics = {k: val[k] for k in ["split", "n_train", "n_test", "auc", "brier", "base_rate_test"]}
    with open(os.path.join(results_dir, "validation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
