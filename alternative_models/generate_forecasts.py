"""Generate next-period forecasts for models that passed validation."""
import sys, os, warnings, json

# Path setup — ensure alternative_models/src takes priority over project root src/
_script_dir = os.path.dirname(os.path.abspath(sys.argv[0] or "."))
if os.path.isfile(os.path.join(_script_dir, "src", "data_macro.py")):
    _dir = _script_dir
elif os.path.isfile(os.path.join(os.path.abspath("alternative_models"), "src", "data_macro.py")):
    _dir = os.path.abspath("alternative_models")
else:
    _dir = os.path.abspath(".")

# Remove any existing src paths that might shadow
sys.path = [p for p in sys.path if not (os.path.isfile(os.path.join(p, "src", "__init__.py"))
            and not os.path.isfile(os.path.join(p, "src", "data_macro.py")))]
# Insert our paths (alternative_models first, then project root)
sys.path.insert(0, os.path.join(_dir, ".."))
sys.path.insert(0, _dir)
os.chdir(_dir)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from src.data_macro import fetch_all_macro, compute_ar1_consensus
from src.feature_engineering import build_feature_matrix, THRESHOLDS
from src.xgboost_models import MacroXGBRegressor, MacroXGBClassifier, create_surprise_labels
from src.probabilistic import OrderedLogit, MacroQuantileRegression, ensemble_probabilities
from src.evaluation import train_test_split_temporal

data = fetch_all_macro(force=False)

# Only models that PASSED overfitting gate AND beat random accuracy
TARGETS_RUN = {
    "US_UNRATE": {"name": "US Unemployment Rate (Apr)", "unit": "%", "consensus": 4.3},
    "US_AHE_MOM": {"name": "US Avg Hourly Earnings MoM (Apr)", "unit": "pp", "consensus": 0.2},
    "BR_IPCA_YOY": {"name": "Brazil IPCA YoY (Apr)", "unit": "%", "consensus": 4.7},
}

results = {}

for target_key, meta in TARGETS_RUN.items():
    print(f"\n{'='*50}")
    print(f"  {meta['name']}")
    print(f"{'='*50}")

    y, X = build_feature_matrix(target_key, data, mode="xgb")
    split = train_test_split_temporal(y, X)
    threshold = THRESHOLDS[target_key]

    # Refit on ALL available data (train + test) for inference
    X_full = pd.concat([split["X_train"], split["X_test"]]).sort_index()
    y_full = pd.concat([split["y_train"], split["y_test"]]).sort_index()

    # XGBoost Regressor
    xgb_reg = MacroXGBRegressor(params={
        "n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
        "reg_lambda": 1.0, "random_state": 42,
    })
    xgb_reg.fit(X_full, y_full)

    # Build classification labels
    cons_df = compute_ar1_consensus(y, window=36)
    cons_full = cons_df["forecast"].reindex(y_full.index).ffill()
    valid = cons_full.dropna().index.intersection(y_full.index)
    labels_full = create_surprise_labels(y_full.loc[valid], cons_full.loc[valid], threshold)
    X_clf = X_full.loc[valid]

    # Ordered Logit
    ol = OrderedLogit(reg_lambda=0.1)
    ol.fit(X_clf.values, labels_full.values)

    # XGBoost Classifier
    xgb_clf = MacroXGBClassifier()
    xgb_clf.fit(X_clf.values, labels_full.values)

    # Quantile Regression
    qr = MacroQuantileRegression()
    qr.fit(X_clf, y_full.loc[valid])

    # Inference: use the most recent feature row
    X_latest = X.iloc[[-1]]
    latest_date = X.index[-1]
    # Align to classification feature columns (same cols as X_clf)
    X_latest_clf = X_latest[X_clf.columns]

    # Point forecast (uses full feature set)
    point_pred = float(xgb_reg.predict(X_latest)[0])

    # Probabilities from each model (use clf-aligned features)
    ol_p = ol.predict_proba(X_latest_clf.values)
    xgb_p = xgb_clf.predict_proba(X_latest_clf.values).values
    qr_q = qr.predict_quantiles(X_latest_clf)
    qr_s = qr.surprise_probability(X_latest_clf, meta["consensus"], threshold)
    qr_p = qr_s[["P_miss", "P_inline", "P_beat"]].values

    # Ensemble (25% OL + 50% XGB + 25% QR)
    ens = ensemble_probabilities(ol_p, xgb_p, qr_p)

    r = {
        "target": meta["name"],
        "feature_date": latest_date.strftime("%Y-%m-%d"),
        "consensus": meta["consensus"],
        "unit": meta["unit"],
        "xgb_point_forecast": round(point_pred, 4),
        "quantile_median": round(float(qr_q["q50"].values[0]), 4),
        "ci_80_lower": round(float(qr_q["q10"].values[0]), 4),
        "ci_80_upper": round(float(qr_q["q90"].values[0]), 4),
        "ci_50_lower": round(float(qr_q["q25"].values[0]), 4),
        "ci_50_upper": round(float(qr_q["q75"].values[0]), 4),
        "ensemble_P_miss": round(float(ens["P_miss"].values[0]), 4),
        "ensemble_P_inline": round(float(ens["P_inline"].values[0]), 4),
        "ensemble_P_beat": round(float(ens["P_beat"].values[0]), 4),
        "ordered_logit": {"miss": round(float(ol_p[0, 0]), 4), "inline": round(float(ol_p[0, 1]), 4), "beat": round(float(ol_p[0, 2]), 4)},
        "xgb_classifier": {"miss": round(float(xgb_p[0, 0]), 4), "inline": round(float(xgb_p[0, 1]), 4), "beat": round(float(xgb_p[0, 2]), 4)},
        "quantile_reg": {"miss": round(float(qr_p[0, 0]), 4), "inline": round(float(qr_p[0, 1]), 4), "beat": round(float(qr_p[0, 2]), 4)},
    }
    results[target_key] = r

    print(f"  Consensus: {meta['consensus']}{meta['unit']}")
    print(f"  XGB Point Forecast: {point_pred:.4f}")
    print(f"  QR Median: {r['quantile_median']}")
    print(f"  80% CI: [{r['ci_80_lower']}, {r['ci_80_upper']}]")
    print(f"  50% CI: [{r['ci_50_lower']}, {r['ci_50_upper']}]")
    print(f"  Ensemble: P(miss)={r['ensemble_P_miss']:.1%}, P(inline)={r['ensemble_P_inline']:.1%}, P(beat)={r['ensemble_P_beat']:.1%}")

# Save
os.makedirs("results/forecasts", exist_ok=True)
with open("results/forecasts/next_period_forecasts.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results/forecasts/next_period_forecasts.json")
