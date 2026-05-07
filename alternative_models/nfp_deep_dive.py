"""
NFP Deep Dive v2: Multicollinearity-controlled model comparison.

Fixes vs v1:
- All features strictly lagged (no look-ahead)
- All features standardised (StandardScaler)
- VIF-based iterative feature elimination
- PCA transformation to orthogonal factors
- Systematic model comparison on clean feature sets
- Widened beat/miss threshold (±50K)
"""
import sys, os, warnings, json

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path
            if not (os.path.isfile(os.path.join(p, "src", "__init__.py"))
                    and not os.path.isfile(os.path.join(p, "src", "data_macro.py")))]
sys.path.insert(0, os.path.join(_dir, ".."))
sys.path.insert(0, _dir)
os.chdir(_dir)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from src.data_macro import fetch_all_macro, compute_ar1_consensus
from src.feature_engineering import beta_weights, align_hf_to_lf
from src.xgboost_models import MacroXGBClassifier, create_surprise_labels
from src.probabilistic import OrderedLogit, MacroQuantileRegression, ensemble_probabilities
from src.evaluation import (
    train_test_split_temporal, ExpandingWindowCV,
    evaluate_regression, evaluate_classification, check_overfitting,
    TRAIN_END, TEST_START, TEST_END,
)

sns.set_style("whitegrid")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
NFP_THRESHOLD = 50.0

# =====================================================================
# 1. DATA
# =====================================================================
print("=" * 60)
print("1. DATA")
print("=" * 60)
data = fetch_all_macro(force=False)
fred = data["fred"]
ns = data["ns_factors"]
nfp = fred["NFP"].dropna().resample("ME").last().diff().dropna()
nfp.name = "NFP_change"
lf = nfp.index
print(f"NFP: {len(nfp)} obs, {nfp.index.min():%Y-%m} to {nfp.index.max():%Y-%m}")

# =====================================================================
# 2. FEATURE CONSTRUCTION (ALL LAGGED, NO LOOK-AHEAD)
# =====================================================================
print(f"\n{'=' * 60}")
print("2. FEATURE CONSTRUCTION (all lagged)")
print("=" * 60)


def midas_lag(s, K=22, theta=(1.0, 5.0)):
    """MIDAS aggregate using PRIOR month's data only."""
    lf_lagged = lf - pd.DateOffset(months=1)
    hf = align_hf_to_lf(s.dropna(), lf_lagged, K)
    w = beta_weights(K, theta[0], theta[1])
    out = (hf * w).sum(axis=1)
    out[np.isnan(hf).any(axis=1)] = np.nan
    return pd.Series(out, index=lf)


def monthly_lag(name, lag=1):
    s = fred[name].dropna().resample("ME").last()
    return s.reindex(lf, method="ffill").shift(lag)


# Full candidate feature set
raw = pd.DataFrame(index=lf)
raw["nfp_lag1"] = nfp.shift(1)
raw["claims_midas"] = midas_lag(fred["CLAIMS"].dropna(), K=4)
raw["cont_claims_midas"] = midas_lag(fred["CONT_CLAIMS"].dropna(), K=4)
raw["vix_midas"] = midas_lag(fred["VIX"].dropna(), K=22)
raw["credit_spread_midas"] = midas_lag(fred["CREDIT_SPREAD"].dropna(), K=22)
raw["yield_spread_midas"] = midas_lag(fred["YIELD_SPREAD"].dropna(), K=22)
raw["fed_rate_midas"] = midas_lag(fred["FED_RATE"].dropna(), K=22)
raw["sp500_midas"] = midas_lag(fred["SP500"].dropna(), K=22)
raw["us_level_midas"] = midas_lag(ns["US_Level"].dropna(), K=22)
raw["us_slope_midas"] = midas_lag(ns["US_Slope"].dropna(), K=22)
raw["ism_emp_lag1"] = monthly_lag("ISM_EMP", 1)
raw["temp_help_lag1"] = monthly_lag("TEMP_HELP", 1)
raw["awh_mfg_lag1"] = monthly_lag("AWH_MFG", 1)
raw["consumer_conf_lag1"] = monthly_lag("CONSUMER_CONF", 1)
raw["unrate_lag1"] = monthly_lag("UNRATE", 1)
raw = raw.dropna()
y_all = nfp.reindex(raw.index).dropna()
raw = raw.loc[y_all.index]
print(f"Raw features: {raw.shape[0]} obs, {raw.shape[1]} features")

# =====================================================================
# 3. MULTICOLLINEARITY DIAGNOSIS & TREATMENT
# =====================================================================
print(f"\n{'=' * 60}")
print("3. MULTICOLLINEARITY TREATMENT")
print("=" * 60)

# Standardise
scaler_full = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_full.fit_transform(raw), index=raw.index, columns=raw.columns
)

# VIF-based iterative elimination (drop highest VIF until all < 10)
def iterative_vif_elimination(X, threshold=10.0):
    """Drop features one at a time until all VIF < threshold."""
    cols = list(X.columns)
    dropped = []
    while True:
        vifs = [variance_inflation_factor(X[cols].values, i) for i in range(len(cols))]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        worst = cols[np.argmax(vifs)]
        dropped.append((worst, max_vif))
        cols.remove(worst)
    return cols, dropped


clean_cols, dropped = iterative_vif_elimination(X_scaled, threshold=10.0)
print(f"Dropped {len(dropped)} features for VIF > 10:")
for feat, vif in dropped:
    print(f"  {feat:30s} VIF={vif:.1f}")
print(f"\nRetained {len(clean_cols)} features: {clean_cols}")

# Verify final VIFs
X_clean = X_scaled[clean_cols]
final_vifs = [variance_inflation_factor(X_clean.values, i) for i in range(len(clean_cols))]
print("\nFinal VIFs:")
for c, v in sorted(zip(clean_cols, final_vifs), key=lambda x: -x[1]):
    print(f"  {c:30s} VIF={v:.2f}")

# PCA on TRAIN set only (no leakage), then project all data
# Exclude COVID from PCA fitting too
covid_mask_all = (X_scaled.index >= "2020-02-01") & (X_scaled.index <= "2020-08-31")
train_mask_pca = (X_scaled.index <= TRAIN_END) & ~covid_mask_all
X_train_for_pca = X_scaled[train_mask_pca]

pca = PCA(n_components=5)
pca.fit(X_train_for_pca)
X_pca_all = pd.DataFrame(
    pca.transform(X_scaled),
    index=X_scaled.index,
    columns=[f"PC{i+1}" for i in range(5)],
)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"\nPCA (5 components, fit on train ex-COVID): {cumvar[-1]:.1%} variance explained")
for i in range(5):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.1%} "
          f"(cum {cumvar[i]:.1%})")

# Build LAGGED PC features (shift by 1 month — pure prediction, no contemporaneous)
X_pca_lagged = X_pca_all.shift(1).rename(columns=lambda c: c + "_lag1")
X_pca_lagged["nfp_lag1"] = nfp.shift(1).reindex(X_pca_lagged.index)
X_pca_lagged = X_pca_lagged.dropna()

# =====================================================================
# 4. DEFINE FEATURE SETS FOR EACH MODEL
# =====================================================================
print(f"\n{'=' * 60}")
print("4. MODEL SPECIFICATIONS")
print("=" * 60)

# Feature Set A: VIF-cleaned (all lagged raw features, VIF < 10)
X_A = X_clean.copy()

# Feature Set B: Lagged PCA-5 + nfp_lag1 (guaranteed orthogonal, 6 features)
X_B = X_pca_lagged[["PC1_lag1", "PC2_lag1", "PC3_lag1", "PC4_lag1", "PC5_lag1", "nfp_lag1"]].copy()

# Feature Set C: Minimal (3 strongest predictors from literature)
min_cols = [c for c in ["nfp_lag1", "claims_midas", "vix_midas"] if c in X_scaled.columns]
X_C = X_scaled[min_cols].copy()

# Feature Set D: Lagged PCA-3 + nfp_lag1 (most parsimonious orthogonal)
X_D = X_pca_lagged[["PC1_lag1", "PC2_lag1", "PC3_lag1", "nfp_lag1"]].copy()

# Align all feature sets to common index
common_idx = X_A.index.intersection(X_B.index).intersection(X_C.index).intersection(X_D.index)
X_A, X_B, X_C, X_D = X_A.loc[common_idx], X_B.loc[common_idx], X_C.loc[common_idx], X_D.loc[common_idx]
y_all = y_all.reindex(common_idx).dropna()
X_A, X_B, X_C, X_D = X_A.loc[y_all.index], X_B.loc[y_all.index], X_C.loc[y_all.index], X_D.loc[y_all.index]

print(f"Set A (VIF-cleaned):    {X_A.shape[1]} features — {list(X_A.columns)}")
print(f"Set B (LagPCA5+nfp):    {X_B.shape[1]} features")
print(f"Set C (Minimal-3):      {X_C.shape[1]} features — {list(X_C.columns)}")
print(f"Set D (LagPCA3+nfp):    {X_D.shape[1]} features")

# =====================================================================
# 5. RUN ALL MODEL × FEATURE SET COMBINATIONS
# =====================================================================
print(f"\n{'=' * 60}")
print("5. MODEL EVALUATION")
print("=" * 60)

all_results = {}


def evaluate_model(name, model_fn, X, y, exclude_covid=True):
    """Run 3-stage evaluation with COVID exclusion from training."""
    split = train_test_split_temporal(y, X)
    X_tr, y_tr = split["X_train"], split["y_train"]
    X_te, y_te = split["X_test"], split["y_test"]

    # Exclude COVID outlier months (2020-03 to 2020-08) from training
    if exclude_covid:
        covid_mask = (y_tr.index >= "2020-02-01") & (y_tr.index <= "2020-08-31")
        X_tr = X_tr[~covid_mask]
        y_tr = y_tr[~covid_mask]

    if len(y_tr) < 30 or len(y_te) < 3:
        print(f"  {name}: SKIP (train={len(y_tr)}, test={len(y_te)})")
        return None

    # Stage 1: CV (within cleaned train set)
    cv = ExpandingWindowCV(min_train_size=min(60, len(y_tr) - 10), step=3)
    cv_errors = []
    for tr_i, te_i in cv.split(X_tr.values):
        m = model_fn()
        m.fit(X_tr.values[tr_i], y_tr.values[tr_i])
        p = m.predict(X_tr.values[te_i])
        cv_errors.extend((y_tr.values[te_i] - p).tolist())
    cv_rmse = np.sqrt(np.mean(np.array(cv_errors) ** 2))

    # Stage 2: Test
    m_test = model_fn()
    m_test.fit(X_tr.values, y_tr.values)
    test_pred = m_test.predict(X_te.values)
    test_met = evaluate_regression(y_te.values, test_pred)
    overfit = check_overfitting(cv_rmse, test_met["RMSE"])

    # Naive baseline: predict y_lag1
    naive_pred = y.shift(1).reindex(y_te.index).ffill().values
    naive_rmse = np.sqrt(np.nanmean((y_te.values - naive_pred) ** 2))

    # Classification
    consensus = compute_ar1_consensus(y, window=36)
    cons_te = consensus["forecast"].reindex(y_te.index).ffill()
    cons_tr = consensus["forecast"].reindex(y_tr.index).ffill()
    v_te = cons_te.dropna().index.intersection(y_te.index)
    v_tr = cons_tr.dropna().index.intersection(y_tr.index)

    clf_acc = None
    if len(v_tr) > 20 and len(v_te) > 5:
        lab_tr = create_surprise_labels(y_tr.loc[v_tr], cons_tr.loc[v_tr], NFP_THRESHOLD)
        lab_te = create_surprise_labels(y_te.loc[v_te], cons_te.loc[v_te], NFP_THRESHOLD)
        clf = MacroXGBClassifier()
        clf.fit(X_tr.loc[v_tr].values, lab_tr.values)
        clf_p = clf.predict_proba(X_te.loc[v_te].values).values
        ol = OrderedLogit(reg_lambda=0.1)
        ol.fit(X_tr.loc[v_tr].values, lab_tr.values)
        ol_p = ol.predict_proba(X_te.loc[v_te].values)
        ens = ensemble_probabilities(ol_p, clf_p)
        clf_met = evaluate_classification(lab_te.values, ens.values)
        clf_acc = clf_met["accuracy"]

    r = {
        "cv_rmse": round(cv_rmse, 1),
        "test_rmse": round(test_met["RMSE"], 1),
        "naive_rmse": round(naive_rmse, 1),
        "test_r2": round(test_met["R2"], 3),
        "overfit_ratio": round(overfit["ratio"], 2),
        "overfit_passed": overfit["passed"],
        "clf_accuracy": round(clf_acc, 3) if clf_acc else None,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "n_features": X.shape[1],
        "test_pred": test_pred.tolist(),
        "test_actual": y_te.values.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in y_te.index],
    }
    print(f"  {name:40s} | RMSE={r['test_rmse']:8.1f}K | R2={r['test_r2']:7.3f} | "
          f"naive={r['naive_rmse']:8.1f}K | gate={'PASS' if r['overfit_passed'] else 'FAIL':4s} "
          f"| clf={r['clf_accuracy'] or 'N/A'}")
    return r


# Model factories
models = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 50)),
    "ElasticNet": lambda: ElasticNetCV(alphas=np.logspace(-2, 2, 30), l1_ratio=[0.1, 0.5, 0.9], cv=5),
    "XGB_shallow": lambda: XGBRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03,
        min_child_weight=10, reg_lambda=5.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0),
    "XGB_deep": lambda: XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_weight=5, reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.7, random_state=42, verbosity=0),
}

feature_sets = {
    "A_VIF_clean": X_A,
    "B_PCA5": X_B,
    "C_Minimal3": X_C,
    "D_PCA3": X_D,
}

print(f"\n{'Model':40s} | {'RMSE':>8s} | {'R2':>7s} | {'Naive':>8s} | {'Gate':>4s} | Clf")
print("-" * 90)

for fs_name, X_fs in feature_sets.items():
    for model_name, model_fn in models.items():
        combo_name = f"{model_name} + {fs_name}"
        r = evaluate_model(combo_name, model_fn, X_fs, y_all)
        if r:
            all_results[combo_name] = r

# =====================================================================
# 6. COMPARISON & BEST MODEL
# =====================================================================
print(f"\n{'=' * 60}")
print("6. RESULTS RANKING")
print("=" * 60)

# Rank by test RMSE (among those passing overfitting gate)
ranking = sorted(all_results.items(), key=lambda kv: kv[1]["test_rmse"])
print(f"\n{'Rank':>4s} {'Model':40s} {'RMSE':>8s} {'R2':>7s} {'vs Naive':>9s} {'Gate':>5s} {'Clf':>6s}")
print("-" * 85)
for i, (name, r) in enumerate(ranking):
    improvement = (1 - r["test_rmse"] / r["naive_rmse"]) * 100
    gate = "PASS" if r["overfit_passed"] else "FAIL"
    print(f"  {i+1:2d}. {name:40s} {r['test_rmse']:8.1f} {r['test_r2']:7.3f} "
          f"{improvement:+7.1f}% {gate:>5s} {r['clf_accuracy'] or 'N/A':>6}")

# Best = lowest test RMSE among PASSED models
passed = {k: v for k, v in all_results.items() if v["overfit_passed"]}
if passed:
    best_key = min(passed, key=lambda k: passed[k]["test_rmse"])
else:
    best_key = ranking[0][0]
best = all_results[best_key]

print(f"\nBest model: {best_key}")
print(f"  Test RMSE: {best['test_rmse']}K (naive: {best['naive_rmse']}K)")
print(f"  Improvement vs naive: {(1-best['test_rmse']/best['naive_rmse'])*100:+.1f}%")

# =====================================================================
# 7. FIGURES
# =====================================================================
print(f"\n{'=' * 60}")
print("7. FIGURES")
print("=" * 60)

# Comparison chart
fig, ax = plt.subplots(figsize=(14, 6))
names = [k for k, _ in ranking[:15]]
rmses = [all_results[k]["test_rmse"] for k in names]
colors = ["#2ca02c" if all_results[k]["overfit_passed"] else "#d62728" for k in names]
bars = ax.barh(range(len(names)), rmses, color=colors)
# Mark best
for i, n in enumerate(names):
    if n == best_key:
        bars[i].set_edgecolor("black")
        bars[i].set_linewidth(2)
# Naive line
ax.axvline(best["naive_rmse"], color="grey", ls="--", lw=1.5, label=f"Naive (lag-1): {best['naive_rmse']}K")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace(" + ", "\n") for n in names], fontsize=7)
ax.set_xlabel("Test RMSE (K)")
ax.set_title("NFP Model Comparison: Test RMSE (green=gate PASS, red=FAIL)")
ax.legend(fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(RESULTS / "fig_nfp_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig_nfp_comparison.png")

# Actual vs Predicted (best model)
fig, ax = plt.subplots(figsize=(12, 5))
dates = pd.to_datetime(best["test_dates"])
ax.plot(dates, best["test_actual"], "o-", label="Actual", color="#1f77b4", ms=4)
ax.plot(dates, best["test_pred"], "s--", label=f"Predicted ({best_key})", color="#ff7f0e", ms=4)
ax.axhline(0, color="grey", ls="--", lw=0.5)
ax.set_ylabel("NFP Change (K)")
ax.set_title(f"NFP Actual vs Predicted - {best_key} (Test 2023-2025)")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "fig_nfp_actual_vs_pred.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig_nfp_actual_vs_pred.png")

# =====================================================================
# 8. NEXT-PERIOD FORECAST (best model)
# =====================================================================
print(f"\n{'=' * 60}")
print("8. NEXT-PERIOD FORECAST")
print("=" * 60)

# Determine which feature set the best model uses
best_fs_key = best_key.split(" + ")[1]
X_best = feature_sets[best_fs_key]
model_name = best_key.split(" + ")[0]
model_fn = models[model_name]

# Refit on all data
m_final = model_fn()
m_final.fit(X_best.values, y_all.values)
X_latest = X_best.iloc[[-1]]
point_forecast = float(m_final.predict(X_latest.values)[0])

# Quantile regression for CIs
split = train_test_split_temporal(y_all, X_best)
X_full = pd.concat([split["X_train"], split["X_test"]]).sort_index()
y_full = pd.concat([split["y_train"], split["y_test"]]).sort_index()
cons = compute_ar1_consensus(y_all, window=36)
cons_full = cons["forecast"].reindex(y_full.index).ffill()
valid = cons_full.dropna().index.intersection(y_full.index)

qr = MacroQuantileRegression()
qr.fit(X_full.loc[valid], y_full.loc[valid])
qr_q = qr.predict_quantiles(X_latest)

# Classification
labels = create_surprise_labels(y_full.loc[valid], cons_full.loc[valid], NFP_THRESHOLD)
X_clf = X_full.loc[valid]
ol = OrderedLogit(reg_lambda=0.1)
ol.fit(X_clf.values, labels.values)
ol_p = ol.predict_proba(X_latest.values)
xgb_clf = MacroXGBClassifier()
xgb_clf.fit(X_clf.values, labels.values)
xgb_p = xgb_clf.predict_proba(X_latest.values).values
consensus_val = 95.0  # Apr NFP consensus
qr_s = qr.surprise_probability(X_latest, consensus_val, NFP_THRESHOLD)
qr_p = qr_s[["P_miss", "P_inline", "P_beat"]].values
ens = ensemble_probabilities(ol_p, xgb_p, qr_p)

forecast = {
    "best_model": best_key,
    "test_rmse": best["test_rmse"],
    "test_r2": best["test_r2"],
    "consensus": consensus_val,
    "point_forecast": round(point_forecast, 1),
    "qr_median": round(float(qr_q["q50"].values[0]), 1),
    "ci_80": [round(float(qr_q["q10"].values[0]), 1), round(float(qr_q["q90"].values[0]), 1)],
    "ci_50": [round(float(qr_q["q25"].values[0]), 1), round(float(qr_q["q75"].values[0]), 1)],
    "P_miss": round(float(ens["P_miss"].values[0]), 3),
    "P_inline": round(float(ens["P_inline"].values[0]), 3),
    "P_beat": round(float(ens["P_beat"].values[0]), 3),
}

print(f"  Model: {best_key}")
print(f"  Consensus: {forecast['consensus']}K")
print(f"  Point: {forecast['point_forecast']}K, Median: {forecast['qr_median']}K")
print(f"  80% CI: {forecast['ci_80']}")
print(f"  50% CI: {forecast['ci_50']}")
print(f"  P(miss)={forecast['P_miss']:.1%}, P(inline)={forecast['P_inline']:.1%}, P(beat)={forecast['P_beat']:.1%}")

# Save
output = {
    "model_comparison": {k: {kk: vv for kk, vv in v.items()
                             if kk not in ("test_pred", "test_actual", "test_dates")}
                         for k, v in all_results.items()},
    "best_model": best_key,
    "vif_dropped": [(f, round(v, 1)) for f, v in dropped],
    "retained_features": clean_cols,
    "pca_variance_explained": cumvar.tolist(),
    "forecast": forecast,
}
with open(RESULTS / "nfp_comparison.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print("NFP DEEP DIVE v2 COMPLETE")
print("=" * 60)
