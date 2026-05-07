"""
Probabilistic models for macro surprise classification.

- OrderedLogit: custom MLE implementation
- MacroQuantileRegression: statsmodels QuantReg wrapper
- Ensemble combiner
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # logistic sigmoid
from statsmodels.regression.quantile_regression import QuantReg


# =====================================================================
# Custom Ordered Logit (Proportional Odds Model)
# =====================================================================

class OrderedLogit:
    """
    Ordered logistic regression for 3-class macro surprises.

    P(Y <= j | X) = sigma(alpha_j - X @ beta)

    Classes: 0=miss, 1=inline, 2=beat
    Cutpoints: alpha_0 < alpha_1 (2 thresholds for 3 classes)

    P(Y=0) = sigma(alpha_0 - X @ beta)
    P(Y=1) = sigma(alpha_1 - X @ beta) - sigma(alpha_0 - X @ beta)
    P(Y=2) = 1 - sigma(alpha_1 - X @ beta)
    """

    def __init__(self, reg_lambda: float = 0.01):
        self.reg_lambda = reg_lambda
        self.beta_ = None
        self.cutpoints_ = None
        self.n_classes_ = 3

    def _unpack_params(self, params, p):
        """Unpack flat parameter vector into cutpoints and beta."""
        # Cutpoints: alpha_0 free, alpha_1 = alpha_0 + exp(delta) to enforce ordering
        alpha_0 = params[0]
        delta = params[1]
        alpha_1 = alpha_0 + np.exp(delta)
        cutpoints = np.array([alpha_0, alpha_1])
        beta = params[2:]
        return cutpoints, beta

    def _neg_log_likelihood(self, params, X, y):
        """Negative log-likelihood with L2 regularization."""
        p = X.shape[1]
        cutpoints, beta = self._unpack_params(params, p)
        xb = X @ beta

        # Cumulative probabilities
        cum_prob = np.zeros((len(y), self.n_classes_ + 1))
        cum_prob[:, 0] = 0.0
        for j in range(self.n_classes_ - 1):
            cum_prob[:, j + 1] = expit(cutpoints[j] - xb)
        cum_prob[:, self.n_classes_] = 1.0

        # Class probabilities
        probs = np.diff(cum_prob, axis=1)
        probs = np.clip(probs, 1e-10, 1.0)

        # Log-likelihood
        ll = 0.0
        for j in range(self.n_classes_):
            mask = y == j
            ll += np.sum(np.log(probs[mask, j]))

        # L2 regularization on beta
        penalty = 0.5 * self.reg_lambda * np.sum(beta**2)

        return -ll + penalty

    def fit(self, X, y):
        """
        Fit ordered logit by MLE.

        Parameters
        ----------
        X : array of shape (n, p)
        y : array of integer labels {0, 1, 2}
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        n, p = X.shape

        # Initialize: cutpoints from class boundaries, beta from logistic regression
        alpha_0_init = 0.0
        delta_init = 1.0
        beta_init = np.zeros(p)
        params0 = np.concatenate([[alpha_0_init, delta_init], beta_init])

        result = minimize(
            self._neg_log_likelihood,
            params0,
            args=(X, y),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        self.cutpoints_, self.beta_ = self._unpack_params(result.x, p)
        self._opt_result = result
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.

        Returns array of shape (n, 3): [P(miss), P(inline), P(beat)]
        """
        X = np.array(X, dtype=float)
        xb = X @ self.beta_

        cum_prob = np.zeros((len(X), self.n_classes_ + 1))
        cum_prob[:, 0] = 0.0
        for j in range(self.n_classes_ - 1):
            cum_prob[:, j + 1] = expit(self.cutpoints_[j] - xb)
        cum_prob[:, self.n_classes_] = 1.0

        probs = np.diff(cum_prob, axis=1)
        probs = np.clip(probs, 0.0, 1.0)
        # Renormalize
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X) -> np.ndarray:
        """Predict most likely class."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# =====================================================================
# Quantile Regression wrapper
# =====================================================================

class MacroQuantileRegression:
    """
    Quantile regression for probabilistic macro forecasts.

    Fits separate linear quantile regressions at each tau level.
    Converts to beat/miss probabilities by interpolating the predicted CDF.
    """

    def __init__(
        self,
        quantiles: tuple = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ):
        self.quantiles = quantiles
        self.models_ = {}

    def fit(self, X, y):
        """Fit separate QuantReg at each quantile."""
        import statsmodels.api as sm
        X_const = sm.add_constant(np.array(X, dtype=float))
        y_arr = np.array(y, dtype=float)

        for q in self.quantiles:
            model = QuantReg(y_arr, X_const)
            try:
                res = model.fit(q=q, max_iter=1000)
                self.models_[q] = res
            except Exception:
                self.models_[q] = None
        return self

    def predict_quantiles(self, X) -> pd.DataFrame:
        """Predict at all quantile levels. Returns DataFrame."""
        X_arr = np.array(X, dtype=float)
        # Manually add constant column (sm.add_constant fails on single rows)
        X_const = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        preds = {}
        for q in self.quantiles:
            if self.models_.get(q) is not None:
                preds[f"q{int(q*100):02d}"] = self.models_[q].predict(X_const)
            else:
                preds[f"q{int(q*100):02d}"] = np.full(len(X), np.nan)

        df = pd.DataFrame(preds)

        # Monotonic rearrangement: sort quantiles per row
        values = df.values.copy()
        for i in range(len(values)):
            values[i] = np.sort(values[i])
        df = pd.DataFrame(values, columns=df.columns, index=df.index)

        return df

    def surprise_probability(
        self,
        X,
        consensus: float | np.ndarray,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Compute P(beat), P(inline), P(miss) from quantile predictions.

        Uses linear interpolation of the empirical CDF defined by quantile predictions.
        """
        qpreds = self.predict_quantiles(X)
        q_vals = np.array(self.quantiles)

        if isinstance(consensus, (int, float)):
            consensus = np.full(len(X), consensus)

        results = []
        for i in range(len(X)):
            row_q = qpreds.iloc[i].values
            lower = consensus[i] - threshold
            upper = consensus[i] + threshold

            # Interpolate CDF at lower and upper thresholds
            p_below_lower = np.interp(lower, row_q, q_vals, left=0.0, right=1.0)
            p_below_upper = np.interp(upper, row_q, q_vals, left=0.0, right=1.0)

            p_miss = p_below_lower
            p_beat = 1.0 - p_below_upper
            p_inline = 1.0 - p_miss - p_beat

            results.append({
                "P_miss": max(0, p_miss),
                "P_inline": max(0, p_inline),
                "P_beat": max(0, p_beat),
                "median_forecast": np.interp(0.5, q_vals, row_q),
            })

        return pd.DataFrame(results)


# =====================================================================
# Ensemble combiner
# =====================================================================

def ensemble_probabilities(
    ordered_logit_probs: np.ndarray,
    xgb_probs: np.ndarray,
    quantreg_probs: np.ndarray | None = None,
    weights: tuple = (0.25, 0.50, 0.25),
) -> pd.DataFrame:
    """
    Weighted average of probability models.

    Parameters
    ----------
    ordered_logit_probs : (n, 3) array [P(miss), P(inline), P(beat)]
    xgb_probs : (n, 3) array
    quantreg_probs : (n, 3) array or None
    weights : tuple of 3 weights (ordered_logit, xgb, quantreg)

    Returns
    -------
    DataFrame with columns: P_miss, P_inline, P_beat
    """
    if quantreg_probs is None:
        # Only 2 models
        w1, w2 = weights[0] / (weights[0] + weights[1]), weights[1] / (weights[0] + weights[1])
        combined = w1 * ordered_logit_probs + w2 * xgb_probs
    else:
        combined = (
            weights[0] * ordered_logit_probs
            + weights[1] * xgb_probs
            + weights[2] * quantreg_probs
        )

    # Renormalize
    combined = combined / combined.sum(axis=1, keepdims=True)

    return pd.DataFrame(
        combined,
        columns=["P_miss", "P_inline", "P_beat"],
    )
