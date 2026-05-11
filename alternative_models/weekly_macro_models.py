"""
Weekly Macro Models: US CPI, US PPI, US Retail Sales, UK Labour.

Replicates the NFP deep dive pattern (COVID-excluded, VIF elimination,
lagged PCA, 20-combo model search) for each of the 4 targets.
Generates figures, results JSON, and coefficient tables.
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

from src.data_macro import fetch_all_macro, compute_ar1_consensus, fetch_fred_bulk
from src.feature_engineering import beta_weights, align_hf_to_lf
from src.xgboost_models import MacroXGBClassifier, create_surprise_labels
from src.probabilistic import OrderedLogit, MacroQuantileRegression, ensemble_probabilities
from src.evaluation import (
    train_test_split_temporal, ExpandingWindowCV,
    evaluate_regression, evaluate_classification, check_overfitting,
    TRAIN_END,
)

sns.set_style("whitegrid")
RESULTS = Path("results")

# =====================================================================
# DATA
# =====================================================================
print("=" * 60)
print("FETCHING DATA")
print("=" * 60)
# Re-fetch to pick up new series
fred = fetch_fred_bulk(force=True)
data = fetch_all_macro(force=False)
# Use the fresh fred
data["fred"] = fred
ns = data["ns_factors"]

# =====================================================================
# TARGET DEFINITIONS
# =====================================================================

TARGETS = {
    "US_CPI_YOY": {
        "label": "US CPI YoY (%)",
        "consensus": 3.75,
        "previous": 3.3,
        "threshold": 0.2,
        "build": lambda: fred["CPI"].dropna().resample("ME").last().pct_change(12).dropna() * 100,
        "hf_regressors": {
            "OIL_WTI": ("fred", 22), "GAS_PRICE": ("fred", 4),
            "VIX": ("fred", 22), "FED_RATE": ("fred", 22),
            "YIELD_SPREAD": ("fred", 22), "SP500": ("fred", 22),
            "US_Level": ("ns", 22), "US_Slope": ("ns", 22),
            "CLAIMS": ("fred", 4),
        },
        "same_freq": {"CPI_YOY": 0, "PPI": 1, "UNRATE": 1},  # 0=self-lag, 1=other-lag
    },
    "US_PPI_MOM": {
        "label": "US PPI MoM (%)",
        "consensus": 0.5,
        "previous": 0.5,
        "threshold": 0.3,
        "build": lambda: fred["PPI"].dropna().resample("ME").last().pct_change().dropna() * 100,
        "hf_regressors": {
            "OIL_WTI": ("fred", 22), "CREDIT_SPREAD": ("fred", 22),
            "EURUSD": ("fred", 22), "US_Level": ("ns", 22),
            "VIX": ("fred", 22),
        },
        "same_freq": {"PPI_MOM": 0, "CPI": 1},
    },
    "US_RETAIL_MOM": {
        "label": "US Retail Sales MoM (%)",
        "consensus": 0.3,
        "previous": 1.7,
        "threshold": 0.5,
        "build": lambda: fred["RETAIL"].dropna().resample("ME").last().pct_change().dropna() * 100,
        "hf_regressors": {
            "SP500": ("fred", 22), "VIX": ("fred", 22),
            "FED_RATE": ("fred", 22), "GAS_PRICE": ("fred", 4),
            "US_Slope": ("ns", 22),
        },
        "same_freq": {"RETAIL_MOM": 0, "UMCSENT": 1, "UNRATE": 1, "CPI": 1},
    },
    "UK_UNEMP": {
        "label": "UK Unemployment (%)",
        "consensus": 4.9,
        "previous": 4.9,
        "threshold": 0.2,
        "build": lambda: fred["UK_UNEMP"].dropna().resample("ME").last(),
        "hf_regressors": {
            "UK_Level": ("ns", 22), "UK_Slope": ("ns", 22),
            "OIL_WTI": ("fred", 22), "VIX": ("fred", 22),
        },
        "same_freq": {"UK_UNEMP": 0, "UK_BOE_RATE": 1, "UK_WAGES": 1},
    },
}


def midas_lag(s, lf_dates, K=22, theta=(1.0, 5.0)):
    lf_lagged = lf_dates - pd.DateOffset(months=1)
    hf = align_hf_to_lf(s.dropna(), lf_lagged, K)
    w = beta_weights(K, theta[0], theta[1])
    out = (hf * w).sum(axis=1)
    out[np.isnan(hf).any(axis=1)] = np.nan
    return pd.Series(out, index=lf_dates)


def monthly_lag(name, lf_dates, lag=1):
    s = fred[name].dropna().resample("ME").last()
    return s.reindex(lf_dates, method="ffill").shift(lag)


def iterative_vif_elimination(X, threshold=10.0):
    cols = list(X.columns)
    dropped = []
    while len(cols) > 1:
        vifs = [variance_inflation_factor(X[cols].values, i) for i in range(len(cols))]
        max_vif = max(vifs)
        if max_vif < threshold:
            break
        worst = cols[np.argmax(vifs)]
        dropped.append((worst, max_vif))
        cols.remove(worst)
    return cols, dropped


# =====================================================================
# MODEL FACTORIES
# =====================================================================
models = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: RidgeCV(alphas=np.logspace(-2, 4, 50)),
    "ElasticNet": lambda: ElasticNetCV(alphas=np.logspace(-2, 2, 30), l1_ratio=[.1, .5, .9], cv=5),
    "XGB_shallow": lambda: XGBRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.03,
        min_child_weight=10, reg_lambda=5.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0),
    "XGB_deep": lambda: XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_weight=5, reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.7, random_state=42, verbosity=0),
}


def evaluate_model(name, model_fn, X, y_full, exclude_covid=True):
    split = train_test_split_temporal(y_full, X)
    X_tr, y_tr = split["X_train"], split["y_train"]
    X_te, y_te = split["X_test"], split["y_test"]
    if exclude_covid:
        cm = (y_tr.index >= "2020-02-01") & (y_tr.index <= "2020-08-31")
        X_tr, y_tr = X_tr[~cm], y_tr[~cm]
    if len(y_tr) < 30 or len(y_te) < 3:
        return None

    # CV
    cv = ExpandingWindowCV(min_train_size=min(60, len(y_tr) - 10), step=3)
    cv_err = []
    for tri, tei in cv.split(X_tr.values):
        m = model_fn()
        m.fit(X_tr.values[tri], y_tr.values[tri])
        p = m.predict(X_tr.values[tei])
        cv_err.extend((y_tr.values[tei] - p).tolist())
    cv_rmse = np.sqrt(np.mean(np.array(cv_err) ** 2))

    # Test
    m_test = model_fn()
    m_test.fit(X_tr.values, y_tr.values)
    pred = m_test.predict(X_te.values)
    met = evaluate_regression(y_te.values, pred)
    overfit = check_overfitting(cv_rmse, met["RMSE"])
    naive = y_full.shift(1).reindex(y_te.index).ffill().values
    naive_rmse = np.sqrt(np.nanmean((y_te.values - naive) ** 2))

    # Coefficients (for linear models)
    coefs = None
    if hasattr(m_test, "coef_"):
        coefs = dict(zip(X.columns, m_test.coef_))

    # Classification
    cons = compute_ar1_consensus(y_full, window=36)
    threshold_val = 0.2  # will be overridden per-target
    cons_tr = cons["forecast"].reindex(y_tr.index).ffill()
    cons_te = cons["forecast"].reindex(y_te.index).ffill()
    v_tr = cons_tr.dropna().index.intersection(y_tr.index)
    v_te = cons_te.dropna().index.intersection(y_te.index)
    clf_acc = None
    if len(v_tr) > 20 and len(v_te) > 5:
        lab_tr = create_surprise_labels(y_tr.loc[v_tr], cons_tr.loc[v_tr], threshold_val)
        lab_te = create_surprise_labels(y_te.loc[v_te], cons_te.loc[v_te], threshold_val)
        try:
            clf = MacroXGBClassifier()
            clf.fit(X_tr.loc[v_tr].values, lab_tr.values)
            clf_p = clf.predict_proba(X_te.loc[v_te].values).values
            ol = OrderedLogit(reg_lambda=0.1)
            ol.fit(X_tr.loc[v_tr].values, lab_tr.values)
            ol_p = ol.predict_proba(X_te.loc[v_te].values)
            ens = ensemble_probabilities(ol_p, clf_p)
            clf_met = evaluate_classification(lab_te.values, ens.values)
            clf_acc = clf_met["accuracy"]
        except Exception:
            pass

    return {
        "cv_rmse": round(cv_rmse, 4), "test_rmse": round(met["RMSE"], 4),
        "naive_rmse": round(naive_rmse, 4), "test_r2": round(met["R2"], 4),
        "overfit_ratio": round(overfit["ratio"], 2),
        "overfit_passed": overfit["passed"],
        "clf_accuracy": round(clf_acc, 3) if clf_acc else None,
        "n_train": len(y_tr), "n_test": len(y_te),
        "coefs": coefs,
        "test_pred": pred.tolist(), "test_actual": y_te.values.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in y_te.index],
    }


# =====================================================================
# MAIN LOOP: process each target
# =====================================================================
all_target_results = {}

for tgt_key, tgt_cfg in TARGETS.items():
    print(f"\n{'#' * 60}")
    print(f"# TARGET: {tgt_key} — {tgt_cfg['label']}")
    print(f"{'#' * 60}")

    try:
        y = tgt_cfg["build"]()
        if len(y) < 50:
            print(f"  SKIP: only {len(y)} obs")
            continue
        lf = y.index

        # Build raw features
        raw = pd.DataFrame(index=lf)
        raw[f"{tgt_key}_lag1"] = y.shift(1)
        raw[f"{tgt_key}_lag2"] = y.shift(2)

        for reg_name, (source, K) in tgt_cfg["hf_regressors"].items():
            if source == "fred" and reg_name in fred.columns:
                raw[f"{reg_name}_midas"] = midas_lag(fred[reg_name].dropna(), lf, K)
            elif source == "ns" and reg_name in ns.columns:
                raw[f"{reg_name}_midas"] = midas_lag(ns[reg_name].dropna(), lf, K)

        for sf_name, lag_type in tgt_cfg["same_freq"].items():
            if lag_type == 0:
                continue  # self-lag already added above
            if sf_name in fred.columns:
                raw[f"{sf_name}_lag1"] = monthly_lag(sf_name, lf, lag=1)

        raw = raw.dropna()
        y_clean = y.reindex(raw.index).dropna()
        raw = raw.loc[y_clean.index]
        print(f"  Data: {len(y_clean)} obs, {raw.shape[1]} raw features, "
              f"{y_clean.index.min():%Y-%m} to {y_clean.index.max():%Y-%m}")

        # Standardise
        scaler = StandardScaler()
        X_s = pd.DataFrame(scaler.fit_transform(raw), index=raw.index, columns=raw.columns)

        # VIF elimination
        clean_cols, dropped = iterative_vif_elimination(X_s, threshold=10.0)
        print(f"  VIF: dropped {len(dropped)} features, retained {len(clean_cols)}")
        X_A = X_s[clean_cols]

        # PCA on train ex-COVID
        covid_mask = (X_s.index >= "2020-02-01") & (X_s.index <= "2020-08-31")
        train_pca_mask = (X_s.index <= TRAIN_END) & ~covid_mask
        pca = PCA(n_components=min(5, len(X_s.columns)))
        pca.fit(X_s[train_pca_mask])
        n_comp = min(5, pca.n_components_)
        X_pca_all = pd.DataFrame(pca.transform(X_s), index=X_s.index,
                                 columns=[f"PC{i+1}" for i in range(n_comp)])
        cumvar = np.sum(pca.explained_variance_ratio_)
        print(f"  PCA: {n_comp} components, {cumvar:.0%} variance")

        # Lagged PCs
        X_pca_lag = X_pca_all.shift(1).rename(columns=lambda c: c + "_lag1")
        X_pca_lag[f"{tgt_key}_lag1"] = y.shift(1).reindex(X_pca_lag.index)
        X_pca_lag = X_pca_lag.dropna()

        # Feature sets
        n3 = min(3, n_comp)
        pc3_cols = [f"PC{i+1}_lag1" for i in range(n3)] + [f"{tgt_key}_lag1"]
        pc5_cols = [f"PC{i+1}_lag1" for i in range(n_comp)] + [f"{tgt_key}_lag1"]

        # Minimal: target_lag1 + top 2 HF by name
        hf_names = [c for c in clean_cols if "_midas" in c][:2]
        min_cols = [f"{tgt_key}_lag1"] + hf_names
        min_cols = [c for c in min_cols if c in X_s.columns]

        feature_sets = {
            "A_VIF": X_A,
            "B_PCA5": X_pca_lag[pc5_cols] if all(c in X_pca_lag.columns for c in pc5_cols) else None,
            "C_Min": X_s[min_cols] if all(c in X_s.columns for c in min_cols) else None,
            "D_PCA3": X_pca_lag[pc3_cols] if all(c in X_pca_lag.columns for c in pc3_cols) else None,
        }
        feature_sets = {k: v for k, v in feature_sets.items() if v is not None}

        # Align all to common index
        common = y_clean.index
        for fs_name, fs in feature_sets.items():
            common = common.intersection(fs.index)
        y_eval = y_clean.reindex(common).dropna()
        feature_sets = {k: v.loc[y_eval.index] for k, v in feature_sets.items()}

        # Run all combos
        print(f"\n  {'Model':40s} | {'RMSE':>8s} | {'R2':>8s} | {'Naive':>8s} | {'Gate':>4s} | {'Clf':>5s}")
        print("  " + "-" * 85)
        model_results = {}
        for fs_name, X_fs in feature_sets.items():
            for model_name, model_fn in models.items():
                combo = f"{model_name} + {fs_name}"
                r = evaluate_model(combo, model_fn, X_fs, y_eval)
                if r:
                    imp = (1 - r["test_rmse"] / r["naive_rmse"]) * 100
                    gate = "PASS" if r["overfit_passed"] else "FAIL"
                    print(f"  {combo:40s} | {r['test_rmse']:8.4f} | {r['test_r2']:+8.4f} | "
                          f"{r['naive_rmse']:8.4f} | {gate:>4s} | {r['clf_accuracy'] or 'N/A':>5}")
                    model_results[combo] = r

        if not model_results:
            print("  NO MODELS SUCCEEDED")
            continue

        # Select best
        passed = {k: v for k, v in model_results.items() if v["overfit_passed"]}
        pool = passed if passed else model_results
        best_key = min(pool, key=lambda k: pool[k]["test_rmse"])
        best = model_results[best_key]
        print(f"\n  BEST: {best_key} — RMSE={best['test_rmse']}, R2={best['test_r2']}")

        # Forecast
        best_fs_key = best_key.split(" + ")[1]
        best_model_name = best_key.split(" + ")[0]
        X_best = feature_sets[best_fs_key]

        # Refit on all data
        m_final = models[best_model_name]()
        split = train_test_split_temporal(y_eval, X_best)
        X_full = pd.concat([split["X_train"], split["X_test"]]).sort_index()
        y_full = pd.concat([split["y_train"], split["y_test"]]).sort_index()
        # COVID exclude from refit
        cm2 = (y_full.index >= "2020-02-01") & (y_full.index <= "2020-08-31")
        m_final.fit(X_full[~cm2].values, y_full[~cm2].values)

        X_latest = X_best.iloc[[-1]]
        point = float(m_final.predict(X_latest.values)[0])

        # Coefficients
        final_coefs = None
        if hasattr(m_final, "coef_"):
            final_coefs = {col: round(float(c), 6) for col, c in zip(X_best.columns, m_final.coef_)}

        # QR for CI
        cons = compute_ar1_consensus(y_eval, window=36)
        cons_full = cons["forecast"].reindex(y_full.index).ffill()
        valid = cons_full.dropna().index.intersection(y_full[~cm2].index)
        if len(valid) > 20:
            qr = MacroQuantileRegression()
            qr.fit(X_full.loc[valid], y_full.loc[valid])
            qr_q = qr.predict_quantiles(X_latest)
            qr_s = qr.surprise_probability(X_latest, tgt_cfg["consensus"], tgt_cfg["threshold"])

            labels = create_surprise_labels(y_full.loc[valid], cons_full.loc[valid], tgt_cfg["threshold"])
            ol = OrderedLogit(reg_lambda=0.1)
            ol.fit(X_full.loc[valid].values, labels.values)
            ol_p = ol.predict_proba(X_latest.values)
            xgb_clf = MacroXGBClassifier()
            xgb_clf.fit(X_full.loc[valid].values, labels.values)
            xgb_p = xgb_clf.predict_proba(X_latest.values).values
            qr_p = qr_s[["P_miss", "P_inline", "P_beat"]].values
            ens = ensemble_probabilities(ol_p, xgb_p, qr_p)

            forecast = {
                "point": round(point, 4),
                "qr_median": round(float(qr_q["q50"].values[0]), 4),
                "ci_80": [round(float(qr_q["q10"].values[0]), 4), round(float(qr_q["q90"].values[0]), 4)],
                "ci_50": [round(float(qr_q["q25"].values[0]), 4), round(float(qr_q["q75"].values[0]), 4)],
                "P_miss": round(float(ens["P_miss"].values[0]), 3),
                "P_inline": round(float(ens["P_inline"].values[0]), 3),
                "P_beat": round(float(ens["P_beat"].values[0]), 3),
            }
        else:
            forecast = {"point": round(point, 4)}

        print(f"  Forecast: point={forecast['point']}, consensus={tgt_cfg['consensus']}")
        if "P_miss" in forecast:
            print(f"  P(miss)={forecast['P_miss']:.1%}, P(inline)={forecast['P_inline']:.1%}, P(beat)={forecast['P_beat']:.1%}")

        all_target_results[tgt_key] = {
            "label": tgt_cfg["label"],
            "consensus": tgt_cfg["consensus"],
            "previous": tgt_cfg["previous"],
            "best_model": best_key,
            "test_rmse": best["test_rmse"],
            "test_r2": best["test_r2"],
            "overfit_passed": best["overfit_passed"],
            "clf_accuracy": best["clf_accuracy"],
            "n_models_tested": len(model_results),
            "n_passed_gate": len(passed),
            "coefficients": final_coefs,
            "forecast": forecast,
        }

        # Figure: actual vs predicted
        fig, ax = plt.subplots(figsize=(12, 4))
        dates = pd.to_datetime(best["test_dates"])
        ax.plot(dates, best["test_actual"], "o-", label="Actual", ms=3, lw=1)
        ax.plot(dates, best["test_pred"], "s--", label=f"Predicted ({best_key})", ms=3, lw=1)
        ax.set_ylabel(tgt_cfg["label"])
        ax.set_title(f"{tgt_key}: Actual vs Predicted (Test 2023-2025)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(RESULTS / f"fig_weekly_{tgt_key}.png", dpi=200, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Save
with open(RESULTS / "weekly_results.json", "w") as f:
    json.dump(all_target_results, f, indent=2, default=str)
print(f"\n{'=' * 60}")
print("SAVED weekly_results.json")

# Summary table
if all_target_results:
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Target':15s} | {'Cons':>6s} | {'Point':>7s} | {'R2':>7s} | {'P(miss)':>7s} | {'P(beat)':>7s} | Model")
    print("-" * 85)
    for k, v in all_target_results.items():
        fc = v["forecast"]
        pm = f"{fc.get('P_miss', 0):.0%}" if "P_miss" in fc else "N/A"
        pb = f"{fc.get('P_beat', 0):.0%}" if "P_beat" in fc else "N/A"
        print(f"{k:15s} | {v['consensus']:6.2f} | {fc['point']:7.3f} | {v['test_r2']:+7.3f} | {pm:>7s} | {pb:>7s} | {v['best_model']}")

print("\nDONE")
