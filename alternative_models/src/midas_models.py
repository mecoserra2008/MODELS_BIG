"""
Custom MIDAS regression with Beta polynomial weights.

Estimation via concentrating NLS:
- Inner step: OLS for linear parameters given theta
- Outer step: scipy.optimize for theta (beta weight params)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass, field

from .feature_engineering import beta_weights, align_hf_to_lf


@dataclass
class MIDASResult:
    """Container for MIDAS regression results."""
    alpha: float = 0.0
    betas_hf: np.ndarray = field(default_factory=lambda: np.array([]))
    gammas_sf: np.ndarray = field(default_factory=lambda: np.array([]))
    thetas: dict = field(default_factory=dict)
    weight_profiles: dict = field(default_factory=dict)
    fitted: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    r_squared: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_obs: int = 0
    n_params: int = 0
    rmse: float = 0.0


class MIDASRegression:
    """
    Multi-regressor MIDAS regression with Beta polynomial weights.

    y_t = alpha + sum_j beta_j * sum_k w(k; theta_j) * x_{j,t-k} + gamma' Z_t + eps_t

    Estimation: concentrating NLS
    - Given theta -> MIDAS features are fixed -> OLS for (alpha, beta, gamma)
    - Optimize theta by minimizing RSS from the conditional OLS
    """

    def __init__(
        self,
        theta_bounds: tuple = (1.01, 30.0),
        n_restarts: int = 5,
        seed: int = 42,
    ):
        self.theta_bounds = theta_bounds
        self.n_restarts = n_restarts
        self.seed = seed
        self.result_ = None

    def fit(
        self,
        y: np.ndarray,
        hf_lags: dict[str, np.ndarray],
        K_map: dict[str, int],
        Z: np.ndarray | None = None,
    ) -> MIDASResult:
        """
        Fit MIDAS regression.

        Parameters
        ----------
        y : array of shape (T,), target variable
        hf_lags : dict mapping regressor name -> array of shape (T, K_j)
            Raw high-frequency lags for each regressor.
        K_map : dict mapping regressor name -> K (number of HF lags)
        Z : array of shape (T, p) or None, same-frequency controls

        Returns
        -------
        MIDASResult
        """
        T = len(y)
        hf_names = list(hf_lags.keys())
        n_hf = len(hf_names)
        n_sf = Z.shape[1] if Z is not None else 0

        rng = np.random.RandomState(self.seed)
        lo, hi = self.theta_bounds

        def _build_X(theta_flat):
            """Build design matrix given theta parameters."""
            cols = []
            idx = 0
            for name in hf_names:
                t1, t2 = theta_flat[idx], theta_flat[idx + 1]
                K = K_map[name]
                w = beta_weights(K, t1, t2)
                # Weighted sum of HF lags
                weighted = (hf_lags[name] * w).sum(axis=1)
                cols.append(weighted)
                idx += 2
            X = np.column_stack([np.ones(T)] + cols)
            if Z is not None:
                X = np.column_stack([X, Z])
            return X

        def _objective(theta_flat):
            """RSS from conditional OLS given theta."""
            try:
                X = _build_X(theta_flat)
                # Check for NaN/Inf
                if np.any(~np.isfinite(X)):
                    return 1e20
                coef, rss, _, _ = np.linalg.lstsq(X, y, rcond=None)
                fitted = X @ coef
                return np.sum((y - fitted) ** 2)
            except Exception:
                return 1e20

        # Initial theta values: multiple random restarts
        best_rss = np.inf
        best_theta = None

        for restart in range(self.n_restarts):
            if restart == 0:
                # Start with default (1.0, 5.0) for each regressor
                theta0 = np.array([1.5, 5.0] * n_hf)
            else:
                theta0 = rng.uniform(lo, hi, size=2 * n_hf)

            bounds = [(lo, hi)] * (2 * n_hf)

            try:
                res = minimize(
                    _objective,
                    theta0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-8},
                )
                if res.fun < best_rss:
                    best_rss = res.fun
                    best_theta = res.x
            except Exception:
                continue

        if best_theta is None:
            # Fallback: use initial theta
            best_theta = np.array([1.5, 5.0] * n_hf)

        # Final OLS with best theta
        X_final = _build_X(best_theta)
        coef, _, _, _ = np.linalg.lstsq(X_final, y, rcond=None)
        fitted = X_final @ coef
        residuals = y - fitted

        # Extract results
        alpha = coef[0]
        betas_hf = coef[1:1 + n_hf]
        gammas_sf = coef[1 + n_hf:] if n_sf > 0 else np.array([])

        # Theta and weight profiles
        thetas = {}
        weight_profiles = {}
        idx = 0
        for name in hf_names:
            t1, t2 = best_theta[idx], best_theta[idx + 1]
            K = K_map[name]
            thetas[name] = (t1, t2)
            weight_profiles[name] = beta_weights(K, t1, t2)
            idx += 2

        # Fit statistics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_sq = 1 - ss_res / (ss_tot + 1e-10)
        n_params = len(coef) + 2 * n_hf  # linear params + theta params
        aic = T * np.log(ss_res / T + 1e-10) + 2 * n_params
        bic = T * np.log(ss_res / T + 1e-10) + n_params * np.log(T)
        rmse = np.sqrt(ss_res / T)

        self.result_ = MIDASResult(
            alpha=alpha,
            betas_hf=betas_hf,
            gammas_sf=gammas_sf,
            thetas=thetas,
            weight_profiles=weight_profiles,
            fitted=fitted,
            residuals=residuals,
            r_squared=r_sq,
            aic=aic,
            bic=bic,
            n_obs=T,
            n_params=n_params,
            rmse=rmse,
        )
        self._X_final = X_final
        self._coef = coef
        self._best_theta = best_theta
        self._hf_names = hf_names
        self._K_map = K_map
        return self.result_

    def predict(
        self,
        hf_lags_new: dict[str, np.ndarray],
        Z_new: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict using fitted theta and coefficients."""
        T = next(iter(hf_lags_new.values())).shape[0]
        cols = []
        idx = 0
        for name in self._hf_names:
            t1, t2 = self._best_theta[idx], self._best_theta[idx + 1]
            K = self._K_map[name]
            w = beta_weights(K, t1, t2)
            weighted = (hf_lags_new[name] * w).sum(axis=1)
            cols.append(weighted)
            idx += 2
        X = np.column_stack([np.ones(T)] + cols)
        if Z_new is not None:
            X = np.column_stack([X, Z_new])
        return X @ self._coef
