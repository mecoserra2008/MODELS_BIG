"""
XGBoost regression and classification wrappers for macro forecasting.

- MacroXGBRegressor: point forecasts
- MacroXGBClassifier: P(beat), P(inline), P(miss) probabilities
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier


class MacroXGBRegressor:
    """XGBoost regressor with time-series-safe defaults."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        self.model = XGBRegressor(**self.params)
        self.feature_names_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit with optional early stopping on validation set."""
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        fit_params = {}
        if X_val is not None and y_val is not None:
            self.model.set_params(early_stopping_rounds=20)
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, X=None) -> pd.DataFrame:
        """Return feature importance (gain-based)."""
        imp = self.model.feature_importances_
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def tune(self, X, y, cv_splitter, n_trials: int = 30):
        """Tune hyperparameters using Optuna with time-series CV."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed. Using default params.")
            return self.params

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "random_state": 42,
            }
            model = XGBRegressor(**params)
            scores = []
            X_arr = np.array(X)
            y_arr = np.array(y)
            for train_idx, test_idx in cv_splitter.split(X_arr):
                model.fit(X_arr[train_idx], y_arr[train_idx])
                pred = model.predict(X_arr[test_idx])
                rmse = np.sqrt(np.mean((y_arr[test_idx] - pred) ** 2))
                scores.append(rmse)
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        best["random_state"] = 42
        self.params = best
        self.model = XGBRegressor(**best)
        return best


class MacroXGBClassifier:
    """XGBoost 3-class classifier for beat/inline/miss."""

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
        }
        self.model = XGBClassifier(**self.params)
        self.class_names_ = ["miss", "inline", "beat"]

    def fit(self, X, y_labels):
        """
        Fit classifier.
        y_labels: integer labels 0=miss, 1=inline, 2=beat.
        """
        # Handle class imbalance via sample weights
        class_counts = np.bincount(y_labels.astype(int), minlength=3)
        total = len(y_labels)
        weights = np.array([total / (3 * (c + 1)) for c in class_counts])
        sample_weights = weights[y_labels.astype(int)]

        self.model.fit(X, y_labels, sample_weight=sample_weights)
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        """Return DataFrame with columns: miss, inline, beat."""
        probs = self.model.predict_proba(X)
        return pd.DataFrame(probs, columns=self.class_names_)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def tune(self, X, y_labels, cv_splitter, n_trials: int = 30):
        """Tune using Optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return self.params

        from sklearn.metrics import log_loss

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "objective": "multi:softprob",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "random_state": 42,
            }
            model = XGBClassifier(**params)
            scores = []
            X_arr = np.array(X)
            y_arr = np.array(y_labels)
            for train_idx, test_idx in cv_splitter.split(X_arr):
                model.fit(X_arr[train_idx], y_arr[train_idx])
                probs = model.predict_proba(X_arr[test_idx])
                scores.append(log_loss(y_arr[test_idx], probs, labels=[0, 1, 2]))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        best.update({"objective": "multi:softprob", "num_class": 3,
                      "eval_metric": "mlogloss", "random_state": 42})
        self.params = best
        self.model = XGBClassifier(**best)
        return best


def create_surprise_labels(
    y: pd.Series,
    forecasts: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Create 3-class labels: 0=miss, 1=inline, 2=beat.

    miss: actual < forecast - threshold
    inline: |actual - forecast| <= threshold
    beat: actual > forecast + threshold
    """
    surprise = y - forecasts
    labels = pd.Series(1, index=y.index, dtype=int)  # default inline
    labels[surprise < -threshold] = 0  # miss
    labels[surprise > threshold] = 2   # beat
    return labels
