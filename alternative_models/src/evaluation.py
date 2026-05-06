"""
Evaluation framework: train/test split, expanding-window CV, and metrics.

Three-stage workflow:
1. Train-set CV (model selection + tuning)
2. Test-set evaluation (overfitting check)
3. Refit on full sample → inference
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# =====================================================================
# Train/Test Split
# =====================================================================

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"
INFERENCE_START = "2026-01-01"


def train_test_split_temporal(
    y: pd.Series,
    X: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> dict:
    """
    Hard temporal train/test split. No data leakage.

    Returns dict with keys:
    X_train, y_train, X_test, y_test, X_inference, y_inference
    """
    train_mask = y.index <= train_end
    test_mask = (y.index >= test_start) & (y.index <= test_end)
    inference_mask = y.index >= INFERENCE_START

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "X_inference": X[inference_mask],
        "y_inference": y[inference_mask],
    }


# =====================================================================
# Expanding Window CV (for use within train set only)
# =====================================================================

class ExpandingWindowCV:
    """
    Time-series cross-validation with expanding training window.

    Used ONLY within the train set for model selection and hyperparameter tuning.
    Never touches the test set.
    """

    def __init__(
        self,
        min_train_size: int = 120,
        step: int = 1,
        n_test: int = 1,
    ):
        self.min_train_size = min_train_size
        self.step = step
        self.n_test = n_test

    def split(self, X):
        """
        Yield (train_indices, test_indices) tuples.

        Parameters
        ----------
        X : array-like of shape (n_samples, ...)

        Yields
        ------
        train_idx, test_idx : np.ndarray
        """
        n = len(X)
        for start in range(self.min_train_size, n - self.n_test + 1, self.step):
            train_idx = np.arange(0, start)
            test_idx = np.arange(start, min(start + self.n_test, n))
            yield train_idx, test_idx

    def get_n_splits(self, X):
        n = len(X)
        return max(0, (n - self.n_test - self.min_train_size) // self.step + 1)


# =====================================================================
# Regression Metrics
# =====================================================================

def evaluate_regression(y_true, y_pred, naive_pred=None) -> dict:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true : array-like, actual values
    y_pred : array-like, model predictions
    naive_pred : array-like or None, naive forecast (e.g., AR(1) or last value)

    Returns
    -------
    dict with RMSE, MAE, R2, and optionally DM test result
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "n_obs": len(y_true),
    }

    if naive_pred is not None:
        naive_pred = np.array(naive_pred)[mask]
        metrics["RMSE_naive"] = np.sqrt(mean_squared_error(y_true, naive_pred))
        metrics["RMSE_ratio"] = metrics["RMSE"] / (metrics["RMSE_naive"] + 1e-10)

        # Diebold-Mariano test
        dm = diebold_mariano_test(
            y_true - y_pred,
            y_true - naive_pred,
        )
        metrics["DM_stat"] = dm["statistic"]
        metrics["DM_pvalue"] = dm["p_value"]

    return metrics


def diebold_mariano_test(e1, e2, h: int = 1) -> dict:
    """
    Diebold-Mariano test for comparing two forecast error series.

    H0: equal predictive accuracy (MSE)
    H1: e1 has different MSE than e2

    Parameters
    ----------
    e1, e2 : forecast errors (actual - predicted)
    h : forecast horizon (for HAC variance adjustment)

    Returns
    -------
    dict with 'statistic' and 'p_value'
    """
    from scipy.stats import norm

    d = e1**2 - e2**2
    T = len(d)
    d_bar = d.mean()

    # HAC variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / T

    if var_d <= 0:
        return {"statistic": 0.0, "p_value": 1.0}

    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return {"statistic": dm_stat, "p_value": p_value}


# =====================================================================
# Classification Metrics
# =====================================================================

def evaluate_classification(y_true, y_pred_proba, class_names=None) -> dict:
    """
    Compute classification metrics for 3-class beat/inline/miss.

    Parameters
    ----------
    y_true : integer labels (0=miss, 1=inline, 2=beat)
    y_pred_proba : (n, 3) probability array

    Returns
    -------
    dict with log_loss, brier_score, accuracy, macro_f1, confusion_matrix
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.argmax(y_pred_proba, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        metrics["log_loss"] = log_loss(y_true, y_pred_proba, labels=[0, 1, 2])
    except Exception:
        metrics["log_loss"] = np.nan

    # Brier score (multi-class)
    one_hot = np.zeros_like(y_pred_proba)
    for i, label in enumerate(y_true):
        one_hot[i, label] = 1.0
    metrics["brier_score"] = np.mean(np.sum((y_pred_proba - one_hot) ** 2, axis=1))

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    metrics["n_obs"] = len(y_true)

    # Per-class accuracy
    if class_names is None:
        class_names = ["miss", "inline", "beat"]
    for j, name in enumerate(class_names):
        mask = y_true == j
        if mask.sum() > 0:
            metrics[f"recall_{name}"] = accuracy_score(y_true[mask], y_pred[mask])
        else:
            metrics[f"recall_{name}"] = np.nan

    return metrics


# =====================================================================
# Overfitting Check
# =====================================================================

def check_overfitting(
    train_cv_rmse: float,
    test_rmse: float,
    threshold: float = 1.5,
) -> dict:
    """
    Check if model is overfitting by comparing train-CV and test RMSE.

    Parameters
    ----------
    train_cv_rmse : average RMSE from expanding-window CV on train set
    test_rmse : RMSE on held-out test set
    threshold : max acceptable ratio of test/train RMSE

    Returns
    -------
    dict with ratio and pass/fail flag
    """
    ratio = test_rmse / (train_cv_rmse + 1e-10)
    return {
        "train_cv_rmse": train_cv_rmse,
        "test_rmse": test_rmse,
        "ratio": ratio,
        "passed": ratio < threshold,
        "message": (
            f"PASS (ratio={ratio:.2f})"
            if ratio < threshold
            else f"FAIL: overfitting detected (ratio={ratio:.2f} > {threshold})"
        ),
    }


# =====================================================================
# Full Backtest Runner
# =====================================================================

def backtest_regression(
    model_class,
    model_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: ExpandingWindowCV,
) -> pd.DataFrame:
    """
    Run expanding-window backtest on the train set.

    Returns DataFrame with columns: date, actual, predicted.
    """
    results = []
    X_arr = X_train.values
    y_arr = y_train.values
    dates = y_train.index

    for train_idx, test_idx in cv.split(X_arr):
        model = model_class(**model_params)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[test_idx])

        for i, idx in enumerate(test_idx):
            results.append({
                "date": dates[idx],
                "actual": y_arr[idx],
                "predicted": pred[i] if hasattr(pred, "__len__") else pred,
            })

    return pd.DataFrame(results).set_index("date")


def run_three_stage_evaluation(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    naive_forecasts: pd.Series | None = None,
    cv_min_train: int = 120,
) -> dict:
    """
    Execute the full 3-stage evaluation workflow.

    Stage 1: Expanding-window CV on train set
    Stage 2: Test-set evaluation + overfitting check
    Stage 3: Refit on full sample (returned as fitted model)

    Returns dict with all results and the final refitted model.
    """
    # Split
    split = train_test_split_temporal(y, X)

    # Stage 1: Train-set CV
    cv = ExpandingWindowCV(min_train_size=cv_min_train)
    if len(split["y_train"]) < cv_min_train + 5:
        # Not enough data for CV — skip stage 1
        cv_results = None
        cv_rmse = np.nan
    else:
        cv_results = backtest_regression(
            model_class, model_params,
            split["X_train"], split["y_train"], cv,
        )
        cv_rmse = np.sqrt(mean_squared_error(cv_results["actual"], cv_results["predicted"]))

    # Stage 2: Test-set evaluation
    model_train = model_class(**model_params)
    model_train.fit(split["X_train"].values, split["y_train"].values)

    if len(split["y_test"]) > 0:
        test_pred = model_train.predict(split["X_test"].values)
        naive = naive_forecasts.reindex(split["y_test"].index) if naive_forecasts is not None else None
        test_metrics = evaluate_regression(
            split["y_test"].values, test_pred,
            naive_pred=naive.values if naive is not None else None,
        )
        overfit = check_overfitting(cv_rmse, test_metrics["RMSE"])
    else:
        test_metrics = {}
        overfit = {"passed": True, "message": "No test data available"}

    # Stage 3: Refit on all data (train + test) for inference
    X_full = pd.concat([split["X_train"], split["X_test"]]).sort_index()
    y_full = pd.concat([split["y_train"], split["y_test"]]).sort_index()
    model_final = model_class(**model_params)
    model_final.fit(X_full.values, y_full.values)

    return {
        "cv_results": cv_results,
        "cv_rmse": cv_rmse,
        "test_metrics": test_metrics,
        "overfitting_check": overfit,
        "model_final": model_final,
        "split": split,
    }
