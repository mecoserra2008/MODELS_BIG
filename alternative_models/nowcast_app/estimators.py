"""
Multi-method estimation wrapper for the nowcasting app.

All fit functions return a FitResult dict:
    method     : str
    coef_df    : DataFrame with columns [feature, coef, std_err, t_stat, p_value, ci_low, ci_high]
    r2         : float
    rmse       : float
    aic        : float | None
    bic        : float | None
    fitted     : np.ndarray
    residuals  : np.ndarray
    y_index    : Index (dates)
    y_train    : np.ndarray (in-sample)
    y_test     : np.ndarray | None
    extra      : dict (method-specific: feature_importance, probabilities, quantile, etc.)
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

_APP_DIR = Path(__file__).parent
_ALT_DIR = _APP_DIR.parent
sys.path.insert(0, str(_ALT_DIR))

from src.xgboost_models import MacroXGBRegressor
from src.probabilistic import OrderedLogit, MacroQuantileRegression

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred) -> tuple[float, float]:
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(r2), rmse


def _empty_coef_df(feature_names: list) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": feature_names,
        "coef":    [np.nan] * len(feature_names),
        "std_err": [np.nan] * len(feature_names),
        "t_stat":  [np.nan] * len(feature_names),
        "p_value": [np.nan] * len(feature_names),
        "ci_low":  [np.nan] * len(feature_names),
        "ci_high": [np.nan] * len(feature_names),
    })


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def fit_ols(X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
    X_sm = sm.add_constant(X.values, has_constant="add")
    model = sm.OLS(y.values, X_sm).fit()

    names = ["const"] + list(X.columns)
    ci_raw = model.conf_int()
    ci = ci_raw.values if hasattr(ci_raw, "values") else ci_raw  # handle ndarray or DataFrame
    coef_df = pd.DataFrame({
        "feature": names,
        "coef":    list(model.params),
        "std_err": list(model.bse),
        "t_stat":  list(model.tvalues),
        "p_value": list(model.pvalues),
        "ci_low":  list(ci[:, 0]),
        "ci_high": list(ci[:, 1]),
    })

    r2, rmse = _metrics(y.values, model.fittedvalues)
    return {
        "method":    "OLS",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       float(model.aic),
        "bic":       float(model.bic),
        "fitted":    model.fittedvalues,
        "residuals": model.resid,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"sm_result": model},
    }


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def fit_ridge(X: pd.DataFrame, y: pd.Series, alphas=None, **kwargs) -> dict:
    if alphas is None:
        alphas = np.logspace(-3, 4, 50)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X.values, y.values)
    fitted = model.predict(X.values)
    residuals = y.values - fitted
    r2, rmse = _metrics(y.values, fitted)

    coef_df = _empty_coef_df(list(X.columns))
    coef_df["coef"] = model.coef_

    return {
        "method":    "Ridge",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"alpha": float(model.alpha_)},
    }


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------

def fit_lasso(X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
    model = LassoCV(cv=5, max_iter=10000, random_state=42)
    model.fit(X.values, y.values)
    fitted = model.predict(X.values)
    residuals = y.values - fitted
    r2, rmse = _metrics(y.values, fitted)

    coef_df = _empty_coef_df(list(X.columns))
    coef_df["coef"] = model.coef_

    return {
        "method":    "Lasso",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"alpha": float(model.alpha_),
                      "n_nonzero": int((model.coef_ != 0).sum())},
    }


# ---------------------------------------------------------------------------
# ElasticNet
# ---------------------------------------------------------------------------

def fit_elasticnet(X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
    model = ElasticNetCV(cv=5, max_iter=10000, random_state=42,
                         l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
    model.fit(X.values, y.values)
    fitted = model.predict(X.values)
    residuals = y.values - fitted
    r2, rmse = _metrics(y.values, fitted)

    coef_df = _empty_coef_df(list(X.columns))
    coef_df["coef"] = model.coef_

    return {
        "method":    "ElasticNet",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"alpha": float(model.alpha_), "l1_ratio": float(model.l1_ratio_),
                      "n_nonzero": int((model.coef_ != 0).sum())},
    }


# ---------------------------------------------------------------------------
# OrderedLogit (3-class classification: miss / inline / beat)
# ---------------------------------------------------------------------------

def fit_ordered_logit(
    X: pd.DataFrame,
    y: pd.Series,
    sigma_threshold: float = 0.5,
    **kwargs,
) -> dict:
    """
    Converts continuous y to 3-class labels using ±sigma_threshold * std(y),
    then fits the custom OrderedLogit model from src/probabilistic.py.
    """
    sigma = y.std()
    y_cat = np.where(y.values < -sigma_threshold * sigma, 0,
                     np.where(y.values > sigma_threshold * sigma, 2, 1))

    if len(np.unique(y_cat)) < 3:
        raise ValueError(
            f"Target does not have all 3 classes at threshold ±{sigma_threshold:.2f}σ. "
            "Try reducing the threshold."
        )

    model = OrderedLogit(reg_lambda=0.01)
    model.fit(X.values, y_cat)
    probs = model.predict_proba(X.values)  # (T, 3)
    predicted_class = np.argmax(probs, axis=1)

    # Pseudo-residuals: y_cat - predicted_class
    residuals = (y_cat - predicted_class).astype(float)
    accuracy = float((predicted_class == y_cat).mean())

    coef_df = _empty_coef_df(list(X.columns))
    if model.beta_ is not None:
        coef_df["coef"] = model.beta_

    return {
        "method":    "OrderedLogit",
        "coef_df":   coef_df,
        "r2":        accuracy,       # using accuracy as goodness-of-fit for display
        "rmse":      float(np.mean(np.abs(residuals))),
        "aic":       None,
        "bic":       None,
        "fitted":    predicted_class.astype(float),
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y_cat,
        "y_test":    None,
        "extra":     {
            "probabilities": probs,
            "y_cat":         y_cat,
            "cutpoints":     model.cutpoints_,
            "accuracy":      accuracy,
            "sigma_threshold": sigma_threshold,
            "class_labels":  ["Miss", "Inline", "Beat"],
        },
    }


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def fit_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str | None = None,
    params: dict | None = None,
    **kwargs,
) -> dict:
    model = MacroXGBRegressor(params=params)

    # Optionally split for early stopping
    X_tr, y_tr = X, y
    X_val, y_val = None, None
    if train_end is not None:
        mask = X.index <= pd.Timestamp(train_end)
        if mask.sum() > 20:
            X_tr, y_tr = X[mask], y[mask]
            X_val, y_val = X[~mask], y[~mask]

    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    fitted = model.predict(X)
    residuals = y.values - fitted
    r2, rmse = _metrics(y.values, fitted)
    importance = model.feature_importance(X)

    coef_df = _empty_coef_df(list(X.columns))
    imp_dict = dict(zip(importance["feature"], importance["importance"]))
    coef_df["coef"] = [imp_dict.get(f, 0.0) for f in X.columns]
    coef_df.rename(columns={"coef": "importance"}, inplace=True)

    return {
        "method":    "XGBoost",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y_tr.values,
        "y_test":    None,
        "extra":     {"feature_importance": importance},
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 5,
    **kwargs,
) -> dict:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X.values, y.values)
    fitted = model.predict(X.values)
    residuals = y.values - fitted
    r2, rmse = _metrics(y.values, fitted)

    imp = pd.DataFrame({
        "feature": list(X.columns),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    coef_df = _empty_coef_df(list(X.columns))
    imp_dict = dict(zip(imp["feature"], imp["importance"]))
    coef_df["coef"] = [imp_dict.get(f, 0.0) for f in X.columns]
    coef_df.rename(columns={"coef": "importance"}, inplace=True)

    return {
        "method":    "Random Forest",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"feature_importance": imp},
    }


# ---------------------------------------------------------------------------
# Quantile Regression
# ---------------------------------------------------------------------------

def fit_quantile(
    X: pd.DataFrame,
    y: pd.Series,
    quantile: float = 0.5,
    **kwargs,
) -> dict:
    X_sm = sm.add_constant(X.values, has_constant="add")
    model = sm.QuantReg(y.values, X_sm).fit(q=quantile)

    names = ["const"] + list(X.columns)
    ci_q_raw = model.conf_int()
    ci_q = ci_q_raw.values if hasattr(ci_q_raw, "values") else ci_q_raw
    coef_df = pd.DataFrame({
        "feature": names,
        "coef":    list(model.params),
        "std_err": list(model.bse),
        "t_stat":  list(model.tvalues),
        "p_value": list(model.pvalues),
        "ci_low":  list(ci_q[:, 0]),
        "ci_high": list(ci_q[:, 1]),
    })

    fitted    = model.fittedvalues
    residuals = y.values - fitted
    r2, rmse  = _metrics(y.values, fitted)

    return {
        "method":    f"Quantile ({quantile:.2f})",
        "coef_df":   coef_df,
        "r2":        r2,
        "rmse":      rmse,
        "aic":       None,
        "bic":       None,
        "fitted":    fitted,
        "residuals": residuals,
        "y_index":   y.index,
        "y_train":   y.values,
        "y_test":    None,
        "extra":     {"quantile": quantile},
    }


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

ESTIMATORS = {
    "OLS":           fit_ols,
    "Ridge":         fit_ridge,
    "Lasso":         fit_lasso,
    "ElasticNet":    fit_elasticnet,
    "OrderedLogit":  fit_ordered_logit,
    "XGBoost":       fit_xgboost,
    "Random Forest": fit_random_forest,
    "Quantile":      fit_quantile,
}


def fit_model(method: str, X: pd.DataFrame, y: pd.Series, **kwargs) -> dict:
    """Unified dispatch to any estimation method."""
    if method not in ESTIMATORS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(ESTIMATORS)}")
    return ESTIMATORS[method](X, y, **kwargs)
