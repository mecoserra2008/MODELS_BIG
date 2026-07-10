"""
NFP Deep Dive v3: Multicollinearity-controlled model + genuine out-of-sample forecast.

Fixes vs v1:
- All features strictly lagged (no look-ahead)
- All features standardised (StandardScaler)
- VIF-based iterative feature elimination
- PCA transformation to orthogonal factors
- Systematic model comparison on clean feature sets
- Widened beat/miss threshold (±50K)

New in v3:
- Live FRED refresh (FORCE_REFRESH) so the run uses the latest data
- Expanded candidate regressor set (earnings, sentiment, participation, oil,
  retail, claims/equity momentum) — VIF elimination prunes redundancy
- Genuine out-of-sample forecast row for the *next* (unpublished) NFP print,
  built on an extended monthly index (fixes the old in-sample iloc[-1] bug)
- t-SNE regime-clustering diagnostic figure (visualisation only, not a regressor)
- Binary P(above)/P(below consensus) probabilities (sum to 100%)
- Separate binary probabilistic model for P(NFP change > 120K) vs P(< 120K),
  combining a quantile-CDF estimate and a direct logistic classifier
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
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from src.data_macro import fetch_fred_bulk, load_ns_factors, compute_ar1_consensus
from src.feature_engineering import beta_weights, align_hf_to_lf
from src.xgboost_models import MacroXGBClassifier, create_surprise_labels
from src.probabilistic import OrderedLogit, MacroQuantileRegression, ensemble_probabilities
from src.evaluation import (
    train_test_split_temporal, ExpandingWindowCV,
    evaluate_regression, evaluate_classification, check_overfitting,
    TRAIN_END, TEST_START, TEST_END,
)
from nowcast_app.tsne_viz import run_tsne

sns.set_style("whitegrid")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# CONFIG — review before every run
# ---------------------------------------------------------------------
# Pull latest FRED data (needs FRED_API_KEY). Override with NFP_FORCE_REFRESH=0
# to run from the cached fred_bulk.csv (still valid: features are strictly lagged).
FORCE_REFRESH = os.environ.get("NFP_FORCE_REFRESH", "1") != "0"
NFP_THRESHOLD = 50.0       # ±band (K) for miss / inline / beat
LEVEL_THRESHOLD = 120.0    # threshold (K) for the standalone P(>120K) model
# Street consensus (K) for the print being forecast. MUST be set to the real
# number before running — used for the above/below-consensus probabilities.
JUNE_CONSENSUS = 110.0     # June 2026 street consensus (K)

# =====================================================================
# 1. DATA
# =====================================================================
print("=" * 60)
print("1. DATA")
print("=" * 60)
TODAY = pd.Timestamp.today().normalize()
fred = fetch_fred_bulk(end=TODAY.strftime("%Y-%m-%d"), force=FORCE_REFRESH)
print(f"FRED data range: {fred.index.min():%Y-%m-%d} to {fred.index.max():%Y-%m-%d}")
# Nelson-Siegel yield-curve factors are produced by the separate VAR pipeline and
# may be absent; the yield-curve MIDAS features are dropped gracefully if so.
try:
    ns = load_ns_factors()
    print(f"NS factors loaded: {ns.shape[0]} rows")
except Exception as e:
    ns = None
    print(f"NS factors unavailable ({e}); dropping us_level/us_slope MIDAS features.")
nfp = fred["NFP"].dropna().resample("ME").last().diff().dropna()
nfp.name = "NFP_change"
lf = nfp.index
print(f"NFP: {len(nfp)} obs, {nfp.index.min():%Y-%m} to {nfp.index.max():%Y-%m}")

# The print we are forecasting = the month AFTER the last published NFP.
# Strictly-lagged features make this row fully constructible from <= last-month data.
LAST_ACTUAL = lf.max()
FORECAST_MONTH = (LAST_ACTUAL + pd.offsets.MonthEnd(1))
lf_ext = lf.append(pd.DatetimeIndex([FORECAST_MONTH]))
# BLS releases a reference month's Employment Situation in the first days of the
# NEXT month, so the print out today (2026-07-02) is the June 2026 reference month.
RELEASE_LABEL = f"{FORECAST_MONTH:%B %Y} NFP (released {TODAY:%Y-%m-%d})"
print(f"Last published NFP: {LAST_ACTUAL:%Y-%m}")
print(f"Forecasting (out-of-sample): {RELEASE_LABEL}")
# Guard against off-by-one from data revisions: the today-print should be June 2026.
EXPECTED_FORECAST_MONTH = pd.Timestamp("2026-06-30")
if FORECAST_MONTH != EXPECTED_FORECAST_MONTH:
    print(f"\n*** WARNING: expected to forecast 2026-06 (the 2026-07-02 print) but "
          f"last actual is {LAST_ACTUAL:%Y-%m} → forecasting {FORECAST_MONTH:%Y-%m}. "
          f"Check the FRED refresh. ***\n")
if JUNE_CONSENSUS is None:
    print("\n*** WARNING: JUNE_CONSENSUS is None — set the real street consensus "
          "before trusting the above/below-consensus probabilities. ***\n")

# =====================================================================
# 2. FEATURE CONSTRUCTION (ALL LAGGED, NO LOOK-AHEAD)
# =====================================================================
print(f"\n{'=' * 60}")
print("2. FEATURE CONSTRUCTION (all lagged)")
print("=" * 60)


def midas_lag(s, K=22, theta=(1.0, 5.0)):
    """MIDAS aggregate using PRIOR month's data only (built on extended index)."""
    lf_lagged = lf_ext - pd.DateOffset(months=1)
    hf = align_hf_to_lf(s.dropna(), lf_lagged, K)
    w = beta_weights(K, theta[0], theta[1])
    out = (hf * w).sum(axis=1)
    out[np.isnan(hf).any(axis=1)] = np.nan
    return pd.Series(out, index=lf_ext)


def monthly_lag(name, lag=1, transform=None):
    """Monthly series, resampled to month-end and lagged. Optional transform:
    'mom' (% change) or 'diff' before lagging."""
    s = fred[name].dropna().resample("ME").last()
    if transform == "mom":
        s = s.pct_change() * 100.0
    elif transform == "diff":
        s = s.diff()
    return s.reindex(lf_ext, method="ffill").shift(lag)


# Full candidate feature set — built on the extended index so the forecast
# month gets its own (fully lagged) row. VIF elimination prunes redundancy.
raw = pd.DataFrame(index=lf_ext)
raw["nfp_lag1"] = nfp.reindex(lf_ext).shift(1)
raw["claims_midas"] = midas_lag(fred["CLAIMS"].dropna(), K=4)
raw["cont_claims_midas"] = midas_lag(fred["CONT_CLAIMS"].dropna(), K=4)
raw["vix_midas"] = midas_lag(fred["VIX"].dropna(), K=22)
raw["credit_spread_midas"] = midas_lag(fred["CREDIT_SPREAD"].dropna(), K=22)
raw["yield_spread_midas"] = midas_lag(fred["YIELD_SPREAD"].dropna(), K=22)
raw["fed_rate_midas"] = midas_lag(fred["FED_RATE"].dropna(), K=22)
# NB: FRED's SP500 series is only ~10y (licensing), which would cap the sample at
# ~2015 and drop the 2008 crisis, so equity level/momentum are intentionally excluded.
if ns is not None:
    raw["us_level_midas"] = midas_lag(ns["US_Level"].dropna(), K=22)
    raw["us_slope_midas"] = midas_lag(ns["US_Slope"].dropna(), K=22)
raw["ism_emp_lag1"] = monthly_lag("ISM_EMP", 1)
raw["temp_help_lag1"] = monthly_lag("TEMP_HELP", 1)
raw["awh_mfg_lag1"] = monthly_lag("AWH_MFG", 1)
raw["consumer_conf_lag1"] = monthly_lag("CONSUMER_CONF", 1)
raw["unrate_lag1"] = monthly_lag("UNRATE", 1)

# --- v3 additional candidate regressors (guarded on availability) ---
if "AHE" in fred:
    raw["ahe_mom_lag1"] = monthly_lag("AHE", 1, transform="mom")          # earnings momentum
if "UMCSENT" in fred:
    raw["umcsent_lag1"] = monthly_lag("UMCSENT", 1)                        # Michigan sentiment
if "PARTICIPATION" in fred:
    raw["participation_lag1"] = monthly_lag("PARTICIPATION", 1)            # labour-force participation
if "OIL_WTI" in fred:
    raw["oil_midas"] = midas_lag(fred["OIL_WTI"].dropna(), K=22)           # oil (cost/energy channel)
if "RETAIL" in fred:
    raw["retail_mom_lag1"] = monthly_lag("RETAIL", 1, transform="mom")     # retail demand momentum
if "CLAIMS" in fred:
    # 4-week change in initial claims: drop NaNs FIRST (weekly series on a daily
    # grid) before differencing, else the diff is all-NaN.
    raw["claims_mom"] = midas_lag(fred["CLAIMS"].dropna().diff(4).dropna(), K=4)

# Keep the forecast row aside (features present, NFP not yet published), then
# drop NaNs on the in-sample rows.
raw_forecast_row = raw.loc[[FORECAST_MONTH]].copy()
_insample = raw.drop(index=FORECAST_MONTH)
_min_feat = _insample.notna().sum().idxmin()
_min_n = int(_insample.notna().sum().min())
print(f"  Sample-limiting feature: {_min_feat} ({_min_n} non-null obs)")
raw = _insample.dropna()
y_all = nfp.reindex(raw.index).dropna()
raw = raw.loc[y_all.index]
print(f"Raw features: {raw.shape[0]} obs, {raw.shape[1]} features "
      f"(+1 out-of-sample row for {FORECAST_MONTH:%Y-%m})")
print(f"Forecast-row completeness: "
      f"{raw_forecast_row.notna().sum().sum()}/{raw_forecast_row.shape[1]} features present")

# =====================================================================
# 3. MULTICOLLINEARITY DIAGNOSIS & TREATMENT
# =====================================================================
print(f"\n{'=' * 60}")
print("3. MULTICOLLINEARITY TREATMENT")
print("=" * 60)

# Standardise: fit scaler on in-sample rows only, transform in-sample + forecast row
raw_all = pd.concat([raw, raw_forecast_row])
scaler_full = StandardScaler().fit(raw)
X_scaled_ext = pd.DataFrame(
    scaler_full.transform(raw_all), index=raw_all.index, columns=raw_all.columns
)
X_scaled = X_scaled_ext.loc[raw.index]  # in-sample slice used for VIF/PCA fitting

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
# Transform the FULL extended matrix (in-sample + forecast row) so the forecast
# month gets PC scores too.
X_pca_ext = pd.DataFrame(
    pca.transform(X_scaled_ext),
    index=X_scaled_ext.index,
    columns=[f"PC{i+1}" for i in range(5)],
)
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"\nPCA (5 components, fit on train ex-COVID): {cumvar[-1]:.1%} variance explained")
for i in range(5):
    print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.1%} "
          f"(cum {cumvar[i]:.1%})")

# Build LAGGED PC features (shift by 1 month — pure prediction, no contemporaneous).
# On the extended index, the forecast month's PCk_lag1 = last in-sample month's PC.
X_pca_lagged = X_pca_ext.shift(1).rename(columns=lambda c: c + "_lag1")
X_pca_lagged["nfp_lag1"] = raw_all["nfp_lag1"].reindex(X_pca_lagged.index)
# Keep the forecast row even though its own PC (unused) rows are fine; only drop
# in-sample rows with missing lags.
fc_pca_row = X_pca_lagged.loc[[FORECAST_MONTH]]
X_pca_lagged = X_pca_lagged.drop(index=FORECAST_MONTH).dropna()
X_pca_lagged = pd.concat([X_pca_lagged, fc_pca_row])

# =====================================================================
# 4. DEFINE FEATURE SETS FOR EACH MODEL
# =====================================================================
print(f"\n{'=' * 60}")
print("4. MODEL SPECIFICATIONS")
print("=" * 60)

pca5_cols = ["PC1_lag1", "PC2_lag1", "PC3_lag1", "PC4_lag1", "PC5_lag1", "nfp_lag1"]
pca3_cols = ["PC1_lag1", "PC2_lag1", "PC3_lag1", "nfp_lag1"]
min_cols = [c for c in ["nfp_lag1", "claims_midas", "vix_midas"] if c in X_scaled.columns]

# In-sample feature sets (forecast row extracted separately below)
X_A = X_clean.copy()                                   # VIF-cleaned raw (lagged)
X_B = X_pca_lagged.loc[X_pca_lagged.index != FORECAST_MONTH, pca5_cols].copy()
X_C = X_scaled[min_cols].copy()
X_D = X_pca_lagged.loc[X_pca_lagged.index != FORECAST_MONTH, pca3_cols].copy()

# Align all feature sets to common index
common_idx = X_A.index.intersection(X_B.index).intersection(X_C.index).intersection(X_D.index)
X_A, X_B, X_C, X_D = X_A.loc[common_idx], X_B.loc[common_idx], X_C.loc[common_idx], X_D.loc[common_idx]
y_all = y_all.reindex(common_idx).dropna()
X_A, X_B, X_C, X_D = X_A.loc[y_all.index], X_B.loc[y_all.index], X_C.loc[y_all.index], X_D.loc[y_all.index]

# Out-of-sample forecast row per feature set (identical columns to the in-sample set)
forecast_rows = {
    "A_VIF_clean": X_scaled_ext.loc[[FORECAST_MONTH], clean_cols],
    "B_PCA5": X_pca_lagged.loc[[FORECAST_MONTH], pca5_cols],
    "C_Minimal3": X_scaled_ext.loc[[FORECAST_MONTH], min_cols],
    "D_PCA3": X_pca_lagged.loc[[FORECAST_MONTH], pca3_cols],
}

print(f"Set A (VIF-cleaned):    {X_A.shape[1]} features — {list(X_A.columns)}")
print(f"Set B (LagPCA5+nfp):    {X_B.shape[1]} features")
print(f"Set C (Minimal-3):      {X_C.shape[1]} features — {list(X_C.columns)}")
print(f"Set D (LagPCA3+nfp):    {X_D.shape[1]} features")
for k, fr in forecast_rows.items():
    if fr.isna().any().any():
        print(f"  WARNING: forecast row for {k} has NaNs: "
              f"{list(fr.columns[fr.isna().any()].values)}")

# =====================================================================
# 4b. t-SNE REGIME DIAGNOSTIC (visualisation only — NOT a regressor)
# =====================================================================
# sklearn t-SNE has no out-of-sample transform and is stochastic, so it is used
# purely to reveal non-linear regime clustering of the labour-market feature space.
print(f"\n{'=' * 60}")
print("4b. t-SNE REGIME DIAGNOSTIC")
print("=" * 60)
try:
    emb = run_tsne(X_A, perplexity=30.0, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    covid_mask = (X_A.index >= "2020-02-01") & (X_A.index <= "2020-08-31")
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_all.values, cmap="RdYlBu",
                    s=35, edgecolor="grey", linewidth=0.3, vmin=-300, vmax=500)
    # Highlight COVID months
    ax.scatter(emb[covid_mask, 0], emb[covid_mask, 1], facecolors="none",
               edgecolors="black", s=120, linewidth=1.5, label="COVID 2020")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("NFP change (K)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("t-SNE of VIF-clean labour-market features (colour = NFP change)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_nfp_tsne.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved fig_nfp_tsne.png ({len(emb)} months embedded)")
except Exception as e:
    print(f"  t-SNE skipped: {e}")

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

    # In-sample R^2 (∈[0,1]) on the training fit
    tr_pred = m_test.predict(X_tr.values)
    ss_res_tr = float(np.sum((y_tr.values - tr_pred) ** 2))
    ss_tot_tr = float(np.sum((y_tr.values - y_tr.values.mean()) ** 2))
    r2_insample = 1.0 - ss_res_tr / (ss_tot_tr + 1e-12)

    # Naive baseline: predict y_lag1
    naive_pred = y.shift(1).reindex(y_te.index).ffill().values
    naive_rmse = np.sqrt(np.nanmean((y_te.values - naive_pred) ** 2))
    # OOS R^2 vs naive (Campbell-Thompson): positive ⇔ beats the naive benchmark
    r2_oos_vs_naive = 1.0 - (test_met["RMSE"] / (naive_rmse + 1e-12)) ** 2

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
        "test_r2": round(test_met["R2"], 3),          # OOS R2 vs mean (may be <0)
        "r2_insample": round(r2_insample, 3),         # in-sample R2 (>=0)
        "r2_oos_vs_naive": round(r2_oos_vs_naive, 3), # OOS R2 vs naive (>0 ⇔ beats naive)
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

# Rank by test RMSE (among those passing overfitting gate). Lead with non-negative,
# intuitive metrics: RMSE, %-vs-naive, in-sample R2, and OOS R2 vs naive.
ranking = sorted(all_results.items(), key=lambda kv: kv[1]["test_rmse"])
print(f"\n{'Rank':>4s} {'Model':30s} {'RMSE':>7s} {'vsNaive':>8s} "
      f"{'R2_in':>6s} {'R2oos/nv':>9s} {'Gate':>5s} {'Clf':>6s}")
print("-" * 84)
for i, (name, r) in enumerate(ranking):
    improvement = (1 - r["test_rmse"] / r["naive_rmse"]) * 100
    gate = "PASS" if r["overfit_passed"] else "FAIL"
    print(f"  {i+1:2d}. {name:30s} {r['test_rmse']:7.1f} {improvement:+7.1f}% "
          f"{r['r2_insample']:6.3f} {r['r2_oos_vs_naive']:+9.3f} {gate:>5s} "
          f"{r['clf_accuracy'] or 'N/A':>6}")
print("  (R2_in = in-sample R2 ∈[0,1]; R2oos/nv = OOS R2 vs naive, >0 ⇔ beats naive)")

# Best = lowest test RMSE among PASSED models
passed = {k: v for k, v in all_results.items() if v["overfit_passed"]}
if passed:
    best_key = min(passed, key=lambda k: passed[k]["test_rmse"])
else:
    best_key = ranking[0][0]
best = all_results[best_key]

print(f"\nBest model: {best_key}")
print(f"  Test RMSE: {best['test_rmse']}K (naive: {best['naive_rmse']}K) "
      f"| improvement vs naive: {(1-best['test_rmse']/best['naive_rmse'])*100:+.1f}%")
print(f"  R2(in-sample)={best['r2_insample']:+.3f}  R2(OOS vs naive)={best['r2_oos_vs_naive']:+.3f}")

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
# 8. GENUINE OUT-OF-SAMPLE FORECAST FOR THE NEXT PRINT
# =====================================================================
print(f"\n{'=' * 60}")
print(f"8. OUT-OF-SAMPLE FORECAST — {RELEASE_LABEL}")
print("=" * 60)

# Determine which feature set the best model uses
best_fs_key = best_key.split(" + ")[1]
X_best = feature_sets[best_fs_key]           # in-sample, aligned to y_all
model_name = best_key.split(" + ")[0]
model_fn = models[model_name]

# The genuine out-of-sample row for the unpublished print (strictly lagged features)
X_latest = forecast_rows[best_fs_key][X_best.columns]

# Point forecast: refit best model on all in-sample data EXCLUDING the COVID 2020
# outliers (±20,000K swings would otherwise corrupt the coefficients) — consistent
# with how every model was trained during evaluation.
covid_all = (y_all.index >= "2020-02-01") & (y_all.index <= "2020-08-31")
X_fit, y_fit = X_best[~covid_all], y_all[~covid_all]
m_final = model_fn()
m_final.fit(X_fit.values, y_fit.values)
point_forecast = float(m_final.predict(X_latest.values)[0])

# ---------------------------------------------------------------------
# 8a. FIT METRICS — lead with non-negative, standard numbers
# ---------------------------------------------------------------------
# In-sample R^2 (∈[0,1]) on the (COVID-excluded) fit the model actually uses
insample_pred = m_final.predict(X_fit.values)
ss_res = float(np.sum((y_fit.values - insample_pred) ** 2))
ss_tot = float(np.sum((y_fit.values - y_fit.values.mean()) ** 2))
r2_insample = 1.0 - ss_res / (ss_tot + 1e-12)
# Out-of-sample R^2 vs the NAIVE benchmark (Campbell-Thompson): positive ⇔ beats naive
r2_oos_vs_naive = 1.0 - (best["test_rmse"] / (best["naive_rmse"] + 1e-12)) ** 2
# Out-of-sample R^2 vs the MEAN (sklearn convention) — CAN be < 0 by construction
r2_oos_vs_mean = best["test_r2"]

# ---------------------------------------------------------------------
# 8b. ONE COHERENT PREDICTIVE DISTRIBUTION (all probabilities from it)
# ---------------------------------------------------------------------
# Collect GENUINE out-of-sample residuals for the best model: expanding-window CV
# within the (COVID-excluded) train set, plus the held-out test window.
split = train_test_split_temporal(y_all, X_best)
X_tr, y_tr = split["X_train"], split["y_train"]
X_te, y_te = split["X_test"], split["y_test"]
covid_tr = (y_tr.index >= "2020-02-01") & (y_tr.index <= "2020-08-31")
X_trc, y_trc = X_tr[~covid_tr], y_tr[~covid_tr]

resid = []
cv = ExpandingWindowCV(min_train_size=min(60, len(y_trc) - 10), step=3)
for tr_i, te_i in cv.split(X_trc.values):
    m = model_fn()
    m.fit(X_trc.values[tr_i], y_trc.values[tr_i])
    resid.extend((y_trc.values[te_i] - m.predict(X_trc.values[te_i])).tolist())
m_te = model_fn()
m_te.fit(X_trc.values, y_trc.values)
resid.extend((y_te.values - m_te.predict(X_te.values)).tolist())
resid = np.asarray(resid, dtype=float)

# Predictive distribution = point forecast + OOS residuals (empirical bootstrap).
# Fallback to a Gaussian if too few residuals to form an empirical CDF.
if len(resid) >= 30:
    draws = point_forecast + resid
    dist_type = "empirical_resid_bootstrap"
else:
    sigma = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else best["cv_rmse"]
    draws = point_forecast + np.random.default_rng(42).normal(0.0, sigma, 10000)
    dist_type = "gaussian"
draws = np.sort(draws)

def F(x: float) -> float:
    """Predictive CDF P(NFP change <= x) from the single distribution."""
    return float(np.mean(draws <= x))

def pctl(q: float) -> float:
    return float(np.percentile(draws, q))

# --- Consensus value for this print (user-supplied street number preferred) ---
if JUNE_CONSENSUS is not None:
    consensus_val = float(JUNE_CONSENSUS)
    cons_source = "user"
else:
    recent = y_all.iloc[-36:].values
    b = np.cov(recent[:-1], recent[1:])[0, 1] / (np.var(recent[:-1]) + 1e-10)
    a = recent[1:].mean() - b * recent[:-1].mean()
    consensus_val = float(a + b * y_all.iloc[-1])
    cons_source = "AR1_pseudo"

C, d, L = consensus_val, NFP_THRESHOLD, LEVEL_THRESHOLD

# All probabilities derived from the SAME CDF ⇒ coherent & monotone.
p_below_cons = F(C)
p_above_cons = 1.0 - p_below_cons
p_miss = F(C - d)
p_inline = F(C + d) - F(C - d)
p_beat = 1.0 - F(C + d)
p_below_120 = F(L)
p_above_120 = 1.0 - p_below_120

median = pctl(50)
ci_80 = [pctl(10), pctl(90)]
ci_50 = [pctl(25), pctl(75)]

# --- Coherence guards (fail loudly if the distribution logic breaks) ---
assert abs((p_above_cons + p_below_cons) - 1.0) < 1e-9, "consensus probs must sum to 1"
assert abs((p_miss + p_inline + p_beat) - 1.0) < 1e-9, "band probs must sum to 1"
assert abs((p_above_120 + p_below_120) - 1.0) < 1e-9, "level probs must sum to 1"
# Monotone across sets: with C=110 < L=120 < C+d=160 ⇒ P(>160) ≤ P(>120) ≤ P(>110)
assert p_beat <= p_above_120 + 1e-9 <= p_above_cons + 2e-9, "coherence chain violated"

# --- Secondary MODEL-BASED cross-check (OL + XGB + QR), stored in JSON only ---
try:
    cons_series = compute_ar1_consensus(y_all, window=36)["forecast"]
    y_full = pd.concat([split["y_train"], split["y_test"]]).sort_index()
    X_full = pd.concat([split["X_train"], split["X_test"]]).sort_index()
    cons_full = cons_series.reindex(y_full.index).ffill()
    valid = cons_full.dropna().index.intersection(y_full.index)
    labels = create_surprise_labels(y_full.loc[valid], cons_full.loc[valid], NFP_THRESHOLD)
    ol = OrderedLogit(reg_lambda=0.1); ol.fit(X_full.loc[valid].values, labels.values)
    xgb_clf = MacroXGBClassifier(); xgb_clf.fit(X_full.loc[valid].values, labels.values)
    qr = MacroQuantileRegression(); qr.fit(X_full.loc[valid], y_full.loc[valid])
    qr_p = qr.surprise_probability(X_latest, consensus_val, NFP_THRESHOLD)[["P_miss", "P_inline", "P_beat"]].values
    ens = ensemble_probabilities(ol.predict_proba(X_latest.values),
                                 xgb_clf.predict_proba(X_latest.values).values, qr_p)
    crosscheck = {"P_miss": round(float(ens["P_miss"].values[0]), 3),
                  "P_inline": round(float(ens["P_inline"].values[0]), 3),
                  "P_beat": round(float(ens["P_beat"].values[0]), 3)}
except Exception as e:
    crosscheck = {"error": str(e)}

forecast = {
    "forecast_month": FORECAST_MONTH.strftime("%Y-%m"),
    "release_label": RELEASE_LABEL,
    "best_model": best_key,
    "test_rmse": best["test_rmse"],
    "naive_rmse": best["naive_rmse"],
    # Headline (non-negative, standard) fit metrics
    "r2_insample": round(r2_insample, 3),
    "r2_oos_vs_naive": round(r2_oos_vs_naive, 3),
    "r2_oos_vs_mean": round(r2_oos_vs_mean, 3),  # can be <0 by construction (vs mean)
    "consensus": round(consensus_val, 1),
    "consensus_source": cons_source,
    "predictive_dist": dist_type,
    "n_resid": int(len(resid)),
    "point_forecast": round(point_forecast, 1),
    "median": round(median, 1),
    "ci_80": [round(ci_80[0], 1), round(ci_80[1], 1)],
    "ci_50": [round(ci_50[0], 1), round(ci_50[1], 1)],
    # All from ONE distribution ⇒ coherent, each set sums to 1
    "P_above_consensus": round(p_above_cons, 3),
    "P_below_consensus": round(p_below_cons, 3),
    "P_miss": round(p_miss, 3),
    "P_inline": round(p_inline, 3),
    "P_beat": round(p_beat, 3),
    "level_threshold": LEVEL_THRESHOLD,
    "P_above_120k": round(p_above_120, 3),
    "P_below_120k": round(p_below_120, 3),
    "ensemble_crosscheck": crosscheck,
}

print(f"  Model: {best_key}  (feature set {best_fs_key})")
print(f"  Forecasting: {RELEASE_LABEL}  [last actual {LAST_ACTUAL:%Y-%m}]")
print(f"  Fit: R2(in-sample)={r2_insample:+.3f}  R2(OOS vs naive)={r2_oos_vs_naive:+.3f}  "
      f"| RMSE={best['test_rmse']}K vs naive {best['naive_rmse']}K")
print(f"       R2(OOS vs mean)={r2_oos_vs_mean:+.3f}  (vs-mean metric; <0 ⇒ worse than the mean)")
print(f"  Predictive distribution: {dist_type} (n_resid={len(resid)})")
print(f"  Consensus: {consensus_val:.0f}K ({cons_source})")
print(f"  Point forecast: {point_forecast:.1f}K | median: {median:.1f}K")
print(f"  80% CI: [{ci_80[0]:.0f}, {ci_80[1]:.0f}]  |  50% CI: [{ci_50[0]:.0f}, {ci_50[1]:.0f}]")
print(f"  P(above consensus)={p_above_cons:.1%}  P(below consensus)={p_below_cons:.1%}")
print(f"  [±{NFP_THRESHOLD:.0f}K band] P(miss)={p_miss:.1%}  P(inline)={p_inline:.1%}  P(beat)={p_beat:.1%}")
print(f"  P(NFP > {LEVEL_THRESHOLD:.0f}K)={p_above_120:.1%}  P(< {LEVEL_THRESHOLD:.0f}K)={p_below_120:.1%}")
print(f"  Coherence OK: P(>{C+d:.0f})={p_beat:.1%} ≤ P(>{L:.0f})={p_above_120:.1%} "
      f"≤ P(>{C:.0f})={p_above_cons:.1%}")

# --- Forecast fan-chart figure (from the single predictive distribution) ---
try:
    fig, ax = plt.subplots(figsize=(7, 5))
    for lo_q, hi_q, lbl in [(5, 95, "5–95%"), (10, 90, "10–90%"), (25, 75, "25–75%")]:
        lo, hi = pctl(lo_q), pctl(hi_q)
        ax.bar(0, hi - lo, bottom=lo, width=0.5, alpha=0.25, color="#1f77b4", label=lbl)
    ax.scatter(0, point_forecast, color="#d62728", zorder=5, s=70, label="Point forecast")
    ax.scatter(0, median, color="#000", marker="_", s=400, label="Median")
    ax.axhline(consensus_val, color="grey", ls="--", lw=1.3, label=f"Consensus {consensus_val:.0f}K")
    ax.axhline(LEVEL_THRESHOLD, color="green", ls=":", lw=1.3, label=f"{LEVEL_THRESHOLD:.0f}K threshold")
    ax.set_xticks([])
    ax.set_ylabel("NFP change (K)")
    ax.set_title(f"{FORECAST_MONTH:%b %Y} NFP forecast distribution ({best_key})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(RESULTS / "fig_nfp_forecast_fan.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig_nfp_forecast_fan.png")
except Exception as e:
    print(f"  Fan chart skipped: {e}")

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

# =====================================================================
# 9. HISTORICAL PROBABILITY TIME SERIES vs a FIXED 110K LEVEL
# =====================================================================
# For every month, what probability did the model assign to NFP being ABOVE /
# INLINE / BELOW a FIXED 110K level (NOT the moving historical consensus)?
# Point forecasts come from a one-step expanding-window backtest (no look-ahead);
# the predictive distribution is point + the pooled backtest residuals.
print(f"\n{'=' * 60}")
print("9. HISTORICAL PROBABILITY TIME SERIES (fixed 110K)")
print("=" * 60)

FIXED_LEVEL = float(JUNE_CONSENSUS) if JUNE_CONSENSUS is not None else 110.0
BAND = NFP_THRESHOLD  # inline = within ±BAND of the fixed level

Xb, yb = X_best, y_all                       # in-sample feature set + target
idx = yb.index
covid_bt = (idx >= "2020-02-01") & (idx <= "2020-08-31")
MIN_TRAIN = 60

yhat_bt = pd.Series(index=idx, dtype=float)   # one-step OOS point forecasts
resid_bt = []
for t in range(MIN_TRAIN, len(idx)):
    ytr = yb.iloc[:t]
    Xtr = Xb.iloc[:t]
    cmask = covid_bt[:t]
    Xtr2, ytr2 = Xtr[~cmask], ytr[~cmask]     # exclude COVID from training
    if len(ytr2) < 30:
        continue
    m = model_fn()
    m.fit(Xtr2.values, ytr2.values)
    pred = float(m.predict(Xb.iloc[[t]].values)[0])
    yhat_bt.iloc[t] = pred
    if not covid_bt[t]:                        # don't pollute the error pool with COVID
        resid_bt.append(float(yb.iloc[t] - pred))
resid_bt = np.asarray(resid_bt, dtype=float)

valid_idx = yhat_bt.dropna().index
lo_th, hi_th = FIXED_LEVEL - BAND, FIXED_LEVEL + BAND
P_below, P_inline, P_above = [], [], []
for t in valid_idx:
    draws = yhat_bt[t] + resid_bt              # coherent predictive distribution
    pb = float(np.mean(draws < lo_th))
    pa = float(np.mean(draws > hi_th))
    P_below.append(pb)
    P_above.append(pa)
    P_inline.append(1.0 - pb - pa)
P_below, P_inline, P_above = map(np.array, (P_below, P_inline, P_above))
print(f"  {len(valid_idx)} months, {valid_idx.min():%Y-%m} to {valid_idx.max():%Y-%m} "
      f"(inline = within ±{BAND:.0f}K of {FIXED_LEVEL:.0f}K)")

fig, ax = plt.subplots(figsize=(15, 6))
ax.stackplot(valid_idx, P_below, P_inline, P_above,
             labels=[f"Below (<{lo_th:.0f}K)", f"Inline ({lo_th:.0f}–{hi_th:.0f}K)",
                     f"Above (>{hi_th:.0f}K)"],
             colors=["#d62728", "#bdbdbd", "#2ca02c"], alpha=0.9)
# Shade COVID gap
ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-08-31"),
           color="white", alpha=0.0)
ax.set_ylim(0, 1)
ax.set_xlim(valid_idx.min(), valid_idx.max())
ax.set_ylabel("Probability")
ax.set_xlabel("Date")
ax.set_title(f"Model probability of NFP change Above / Inline / Below a fixed "
             f"{FIXED_LEVEL:.0f}K level ({best_key}, one-step OOS backtest)")
ax.legend(loc="upper center", ncol=3, fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig(RESULTS / "fig_nfp_prob_timeseries_110k.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig_nfp_prob_timeseries_110k.png")

print(f"\n{'=' * 60}")
print("NFP DEEP DIVE v3 COMPLETE")
print("=" * 60)
