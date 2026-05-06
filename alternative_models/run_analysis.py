"""
Full analysis: run all models for all available targets, generate figures and results.
Execute this before building the notebook/report.
"""
import sys
import os
import warnings
# Ensure alternative_models/src is importable (not shadowed by project root src/)
for _base in [os.path.dirname(os.path.abspath(__file__)),
              os.path.abspath("alternative_models"),
              os.path.abspath(".")]:
    if os.path.isfile(os.path.join(_base, "src", "data_macro.py")):
        # Insert alt_models FIRST so its src/ takes priority over project root src/
        # Then project root AFTER for accessing data/
        sys.path.insert(0, os.path.join(_base, ".."))
        sys.path.insert(0, _base)  # This is index 0 → highest priority
        os.chdir(_base)
        break
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from src.data_macro import fetch_all_macro, compute_ar1_consensus
from src.feature_engineering import (
    build_feature_matrix, THRESHOLDS, TARGETS, beta_weights
)
from src.midas_models import MIDASRegression
from src.xgboost_models import MacroXGBRegressor, MacroXGBClassifier, create_surprise_labels
from src.probabilistic import OrderedLogit, MacroQuantileRegression, ensemble_probabilities
from src.evaluation import (
    train_test_split_temporal, ExpandingWindowCV,
    evaluate_regression, evaluate_classification, check_overfitting,
)

sns.set_style("whitegrid")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "model_outputs").mkdir(exist_ok=True)
(RESULTS_DIR / "forecasts").mkdir(exist_ok=True)

# =====================================================================
# 1. Fetch data
# =====================================================================
print("=" * 60)
print("FETCHING DATA")
print("=" * 60)
data = fetch_all_macro(force=False)

# =====================================================================
# 2. Run models for each target that has sufficient data
# =====================================================================

# Focus on targets where we have FRED data available
RUNNABLE_TARGETS = [
    "US_NFP", "US_UNRATE", "US_AHE_MOM", "US_UMCSENT", "US_MICH_INF",
    "BR_IPCA_YOY", "CA_EMPLOYMENT",
]

all_results = {}
all_forecasts = {}

for target_key in RUNNABLE_TARGETS:
    print(f"\n{'=' * 60}")
    print(f"TARGET: {target_key}")
    print(f"{'=' * 60}")

    try:
        # Build features (XGB mode — works for all model types)
        y, X = build_feature_matrix(target_key, data, mode="xgb")
        if len(y) < 80:
            print(f"  Skipping: only {len(y)} observations")
            continue

        print(f"  Data: {len(y)} obs, {X.shape[1]} features")
        print(f"  Range: {y.index.min().strftime('%Y-%m')} to {y.index.max().strftime('%Y-%m')}")

        # Train/test split
        split = train_test_split_temporal(y, X)
        n_train = len(split["y_train"])
        n_test = len(split["y_test"])
        print(f"  Train: {n_train}, Test: {n_test}")

        if n_train < 40 or n_test < 5:
            print(f"  Skipping: insufficient train/test size")
            continue

        # --- STAGE 1: CV on train set ---
        cv = ExpandingWindowCV(min_train_size=min(60, n_train - 10), step=3)
        from xgboost import XGBRegressor

        cv_preds = []
        for train_idx, test_idx in cv.split(split["X_train"].values):
            m = XGBRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                reg_lambda=1.0, random_state=42, verbosity=0
            )
            m.fit(split["X_train"].values[train_idx], split["y_train"].values[train_idx])
            p = m.predict(split["X_train"].values[test_idx])
            for i, idx in enumerate(test_idx):
                cv_preds.append((split["y_train"].values[idx], p[i]))

        cv_df = pd.DataFrame(cv_preds, columns=["actual", "predicted"])
        cv_rmse = np.sqrt(((cv_df["actual"] - cv_df["predicted"]) ** 2).mean())
        print(f"  Stage 1 CV RMSE: {cv_rmse:.3f}")

        # --- STAGE 2: Test set ---
        xgb_reg = MacroXGBRegressor(params={
            "n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
            "reg_lambda": 1.0, "random_state": 42,
        })
        xgb_reg.fit(split["X_train"], split["y_train"])
        test_pred_xgb = xgb_reg.predict(split["X_test"])
        test_metrics_xgb = evaluate_regression(split["y_test"].values, test_pred_xgb)
        overfit = check_overfitting(cv_rmse, test_metrics_xgb["RMSE"])
        print(f"  Stage 2 XGB Test RMSE: {test_metrics_xgb['RMSE']:.3f}, R2: {test_metrics_xgb['R2']:.3f}")
        print(f"  Overfitting: {overfit['message']}")

        # --- Probabilistic models ---
        # Compute AR(1) consensus
        consensus_df = compute_ar1_consensus(y, window=36)
        threshold = THRESHOLDS.get(target_key, 1.0)

        # Labels for train and test
        consensus_train = consensus_df["forecast"].reindex(split["y_train"].index).ffill()
        consensus_test = consensus_df["forecast"].reindex(split["y_test"].index).ffill()

        # Drop NaN consensus
        valid_train = consensus_train.dropna().index.intersection(split["y_train"].index)
        valid_test = consensus_test.dropna().index.intersection(split["y_test"].index)

        if len(valid_train) > 30 and len(valid_test) > 5:
            labels_train = create_surprise_labels(
                split["y_train"].loc[valid_train],
                consensus_train.loc[valid_train],
                threshold,
            )
            labels_test = create_surprise_labels(
                split["y_test"].loc[valid_test],
                consensus_test.loc[valid_test],
                threshold,
            )

            X_tr_clf = split["X_train"].loc[valid_train]
            X_te_clf = split["X_test"].loc[valid_test]

            # Ordered Logit
            ologit = OrderedLogit(reg_lambda=0.1)
            ologit.fit(X_tr_clf.values, labels_train.values)
            ol_probs = ologit.predict_proba(X_te_clf.values)

            # XGB Classifier
            xgb_clf = MacroXGBClassifier()
            xgb_clf.fit(X_tr_clf.values, labels_train.values)
            xgb_probs = xgb_clf.predict_proba(X_te_clf.values).values

            # Quantile Regression
            qr = MacroQuantileRegression()
            qr.fit(X_tr_clf, split["y_train"].loc[valid_train])
            qr_surprise = qr.surprise_probability(
                X_te_clf, consensus_test.loc[valid_test].values, threshold
            )
            qr_probs = qr_surprise[["P_miss", "P_inline", "P_beat"]].values

            # Ensemble
            ens_probs = ensemble_probabilities(ol_probs, xgb_probs, qr_probs)

            # Evaluate classification
            clf_metrics = evaluate_classification(labels_test.values, ens_probs.values)
            print(f"  Ensemble accuracy: {clf_metrics['accuracy']:.3f}, log-loss: {clf_metrics['log_loss']:.3f}")
            print(f"  Label distribution (test): {labels_test.value_counts().sort_index().to_dict()}")
        else:
            ol_probs = xgb_probs = qr_probs = ens_probs = None
            clf_metrics = {}
            labels_test = None

        # --- Store results ---
        all_results[target_key] = {
            "cv_rmse": cv_rmse,
            "test_rmse": test_metrics_xgb["RMSE"],
            "test_r2": test_metrics_xgb["R2"],
            "overfit_passed": overfit["passed"],
            "overfit_ratio": overfit["ratio"],
            "clf_accuracy": clf_metrics.get("accuracy"),
            "clf_logloss": clf_metrics.get("log_loss"),
            "n_train": n_train,
            "n_test": n_test,
            "n_features": X.shape[1],
        }

        # Feature importance
        fi = xgb_reg.feature_importance()
        all_results[target_key]["top_features"] = fi.head(5).to_dict("records")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

# =====================================================================
# 3. Generate summary figures
# =====================================================================
print(f"\n{'=' * 60}")
print("GENERATING FIGURES")
print(f"{'=' * 60}")

# --- Figure 1: Beta weight profiles ---
fig, ax = plt.subplots(figsize=(10, 5))
K = 22
for (t1, t2), label in [
    ((1.0, 5.0), r"$\theta_1$=1, $\theta_2$=5 (decaying)"),
    ((2.0, 8.0), r"$\theta_1$=2, $\theta_2$=8 (hump, recent-weighted)"),
    ((1.0, 1.0), r"$\theta_1$=1, $\theta_2$=1 (uniform)"),
    ((5.0, 2.0), r"$\theta_1$=5, $\theta_2$=2 (distant-weighted)"),
]:
    w = beta_weights(K, t1, t2)
    ax.plot(range(K), w, label=label, lw=2)
ax.set_xlabel("Lag (business days, 0=most recent)")
ax.set_ylabel("Weight")
ax.set_title("MIDAS Beta Polynomial Weight Profiles (K=22, daily→monthly)")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fig_beta_weights.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved fig_beta_weights.png")

# --- Figure 2: Model comparison table ---
if all_results:
    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = "Target"
    summary_df.to_csv(RESULTS_DIR / "model_summary.csv")
    print(f"\n  Model Summary:")
    print(summary_df[["cv_rmse", "test_rmse", "test_r2", "overfit_passed", "clf_accuracy"]].round(3).to_string())

    # --- Figure 3: Test RMSE comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    targets = list(all_results.keys())
    cv_vals = [all_results[t]["cv_rmse"] for t in targets]
    test_vals = [all_results[t]["test_rmse"] for t in targets]
    x = np.arange(len(targets))
    width = 0.35
    ax.bar(x - width / 2, cv_vals, width, label="Train CV RMSE", color="#1f77b4")
    ax.bar(x + width / 2, test_vals, width, label="Test RMSE", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("Model Performance: Train CV vs Test RMSE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig_cv_vs_test.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved fig_cv_vs_test.png")

    # --- Figure 4: Feature importance (top target) ---
    best_target = max(all_results, key=lambda t: all_results[t].get("test_r2", -99))
    if "top_features" in all_results[best_target]:
        fi_df = pd.DataFrame(all_results[best_target]["top_features"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi_df["feature"], fi_df["importance"], color="#2ca02c")
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top Features: {best_target} (XGBoost)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig_feature_importance.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Saved fig_feature_importance.png")

# Save results as JSON
results_json = {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
                     for kk, vv in v.items() if kk != "top_features"}
                for k, v in all_results.items()}
with open(RESULTS_DIR / "all_results.json", "w") as f:
    json.dump(results_json, f, indent=2, default=str)

print(f"\n{'=' * 60}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 60}")
