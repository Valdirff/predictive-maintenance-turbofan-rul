"""
stochastic_model.py
===================
Stochastic power-function degradation model for C-MAPSS RUL estimation.

Implementation
--------------
Faithfully implements the framework from:

  artigo_exp_1 — Wen et al. (2021), RESS 205:107241
    "A generalized RUL prediction method for complex systems
     based on composite health indicator"

Core approach
~~~~~~~~~~~~~
1. **Log-linearisation** (Eq. 1-2 of the paper):
   The degradation model z(t) = φ + α·t^β·exp(ε) is log-linearised:
       y(t) = ln(z(t) − φ) = θ⁽⁰⁾ + θ⁽¹⁾·ln(t) + ε̃
   This is a weighted linear regression in x = ln(t), y = ln(HI − φ).

2. **WLS parameter estimation** (Section 2.2):
   Weights are a geometric series {c_k} with ratio q > 1, giving more
   influence to recent cycles — closer to failure = more informative.

3. **Empirical Failure Threshold**:
   Instead of a hard-coded config value, the threshold is estimated as
   the mean HI over the last cycle of every training engine. This is
   statistically grounded in the observed failure boundary.

4. **Parameter reconstruction for test engines** (Section 2.3):
   θ̂_T = Σ wᵢ θᵢ  s.t. wᵢ≥0, Σwᵢ=1
   Weights wᵢ are found by minimising the WLS fitting error between
   the reconstructed trajectory and the test observations (SQP-like).

5. **RUL estimation** (Section 2.4):
   Solve θ̂_T(t) = ln(FT − φ) for t = t_fail → RUL = t_fail − t_current.

6. **Uncertainty quantification**:
   Bootstrap over reconstructed parameter sets to produce a 90 % CI.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (
    METADATA_DIR,
    RUL_CAP,
    STOCHASTIC_MAX_EXTRAPOLATION,
    STOCHASTIC_N_BOOTSTRAP,
    STOCHASTIC_WEIGHT_Q,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _geometric_weights(n: int, q: float = STOCHASTIC_WEIGHT_Q) -> np.ndarray:
    """
    Build an increasing geometric weight sequence of length n with ratio q.

    c_k = c_1 · q^(k-1),  k=1..n,   with Σ sqrt(c_k) = 1  (Eq. 4 of paper).
    For simplicity we normalise so that the weights sum to 1.
    """
    if n == 1:
        return np.array([1.0])
    k = np.arange(1, n + 1, dtype=float)
    raw = q ** (k - 1)
    return raw / raw.sum()


def _wls_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """
    Weighted Least Squares (WLS) for the model y = θ⁽⁰⁾ + θ⁽¹⁾·x.

    Returns (θ⁽⁰⁾, θ⁽¹⁾).  Identical to the WLS derivation in Section 2.2.
    """
    W = np.diag(weights)
    X = np.column_stack([np.ones_like(x), x])   # design matrix [1, xₖ]
    # θ = (XᵀWX)⁻¹ XᵀWy
    XTW = X.T @ W
    try:
        theta = np.linalg.solve(XTW @ X, XTW @ y)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(XTW @ X, XTW @ y, rcond=None)[0]
    return float(theta[0]), float(theta[1])


def _trajectory(t: np.ndarray, theta0: float, theta1: float, phi: float) -> np.ndarray:
    """
    Convert log-linear parameters back to the HI space.

    HI(t) = φ + exp(θ⁽⁰⁾ + θ⁽¹⁾·ln(t))  = φ + exp(θ⁽⁰⁾)·t^θ⁽¹⁾
    """
    return phi + np.exp(theta0 + theta1 * np.log(np.maximum(t, 1e-6)))


def _extrapolate_failure_cycle(
    theta0: float,
    theta1: float,
    phi: float,
    failure_threshold: float,
    last_cycle: float,
    max_extra: float = STOCHASTIC_MAX_EXTRAPOLATION,
) -> float:
    """
    Solve HI(t) = failure_threshold analytically for t.

    ln(FT − φ) = θ⁽⁰⁾ + θ⁽¹⁾·ln(t)
    → t = exp( (ln(FT − φ) − θ⁽⁰⁾) / θ⁽¹⁾ )
    """
    inner = failure_threshold - phi
    if inner <= 0 or theta1 == 0:
        return last_cycle + max_extra

    try:
        t_fail = math.exp((math.log(inner) - theta0) / theta1)
        if t_fail < last_cycle or not math.isfinite(t_fail):
            return last_cycle
        return min(t_fail, last_cycle + max_extra)
    except (ValueError, ZeroDivisionError, OverflowError):
        return last_cycle + max_extra


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class StochasticDegradationRUL:
    """
    Stochastic power-function degradation model for turbofan RUL.

    Usage
    -----
    >>> model = StochasticDegradationRUL()
    >>> model.fit(train_df_with_hi)
    >>> rul_point = model.predict_test(test_df_with_hi)
    >>> rul_dist  = model.predict_with_uncertainty(test_df_with_hi)

    Parameters
    ----------
    weight_q         : geometric weight ratio q for WLS (default from config)
    max_extrapolation: hard cap on forward extrapolation in cycles
    n_bootstrap      : Monte Carlo samples for uncertainty quantification
    """

    def __init__(
        self,
        weight_q: float = STOCHASTIC_WEIGHT_Q,
        max_extrapolation: float = STOCHASTIC_MAX_EXTRAPOLATION,
        n_bootstrap: int = STOCHASTIC_N_BOOTSTRAP,
    ) -> None:
        self.weight_q = weight_q
        self.max_extrapolation = max_extrapolation
        self.n_bootstrap = n_bootstrap

        # Set after fit()
        self.engine_params_: dict[int, dict] = {}
        self.phi_: float = 0.0              # baseline HI floor (intercept)
        self.failure_threshold_: float = 0.1

    # ──────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, df_with_hi: pd.DataFrame) -> "StochasticDegradationRUL":
        """
        1. Estimate φ (global HI floor) from the earliest cycles of all engines.
        2. Estimate empirical failure threshold from each engine's last HI.
        3. For each training engine, log-linearise and WLS-fit (θ⁽⁰⁾, θ⁽¹⁾).

        Parameters
        ----------
        df_with_hi : DataFrame with columns [unit_id, cycle, HI]

        Returns
        -------
        self
        """
        # 1. Estimate φ: small offset below the minimum observed HI
        global_min_hi = df_with_hi["HI"].min()
        self.phi_ = max(0.0, global_min_hi - 0.02)

        # 2. Empirical failure threshold
        # Use percentile 75 of final-cycle HI: more conservative than mean,
        # avoids pushing the threshold too close to phi which makes the
        # log-transform blow up.
        last_hi_values = (
            df_with_hi
            .sort_values("cycle")
            .groupby("unit_id")["HI"]
            .last()
            .values
        )
        # Use the 75th percentile so the threshold is well above the noise floor
        self.failure_threshold_ = float(np.percentile(last_hi_values, 75))
        ft_std = float(np.std(last_hi_values))
        print(
            f"[StochasticDegradationRUL] Failure threshold estimated (p75): "
            f"{self.failure_threshold_:.4f} ± {ft_std:.4f}  (φ={self.phi_:.4f})"
        )

        # 3. Per-engine WLS fit
        n_ok = n_fail = 0
        for engine_id, group in df_with_hi.groupby("unit_id"):
            group = group.sort_values("cycle")
            cycles = group["cycle"].values.astype(float)
            hi = group["HI"].values.astype(float)

            # Log-linearise: y = ln(HI − φ)
            hi_shifted = hi - self.phi_
            valid = hi_shifted > 0
            if valid.sum() < 3:
                n_fail += 1
                self.engine_params_[int(engine_id)] = self._fallback_params(cycles, hi)
                continue

            x = np.log(cycles[valid])
            y = np.log(hi_shifted[valid])
            w = _geometric_weights(valid.sum(), self.weight_q)

            theta0, theta1 = _wls_fit(x, y, w)

            # Require declining trend (theta1 < 0 for HI-decreasing degradation)
            if theta1 >= 0:
                n_fail += 1
                self.engine_params_[int(engine_id)] = self._fallback_params(cycles, hi)
                continue

            n_ok += 1
            self.engine_params_[int(engine_id)] = {
                "theta0": theta0,
                "theta1": theta1,
                "phi": self.phi_,
                "last_cycle": float(cycles[-1]),
                "last_hi": float(hi[-1]),
                "fit_ok": True,
            }

        print(
            f"[StochasticDegradationRUL] Fitted {n_ok} engines OK, "
            f"{n_fail} fallbacks."
        )
        return self

    def _fallback_params(self, cycles: np.ndarray, hi: np.ndarray) -> dict:
        """Minimal fallback when WLS fails — linear slope as pseudo-theta."""
        slope = (hi[-1] - hi[0]) / max(cycles[-1] - cycles[0], 1.0)
        # Encode as a minimal degradation model
        return {
            "theta0": float(np.log(max(hi[0] - self.phi_, 1e-6))),
            "theta1": min(slope * 2, -1e-4),  # force negative
            "phi": self.phi_,
            "last_cycle": float(cycles[-1]),
            "last_hi": float(hi[-1]),
            "fit_ok": False,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Parameter reconstruction  (Section 2.3 of artigo_exp_1)
    # ──────────────────────────────────────────────────────────────────────

    def _reconstruct_params(
        self,
        test_cycles: np.ndarray,
        test_hi: np.ndarray,
    ) -> dict:
        """
        Given the observed trajectory of a (new) test engine, find the
        convex combination of training-engine parameters that best
        reproduces the observation in WLS sense.

        Optimisation (mirror of Eq. 16 in the paper):
            min_w  ‖ y_T − X_T · (Σ wᵢ θᵢ) ‖²_W
            s.t.   wᵢ ≥ 0,  Σ wᵢ = 1

        Falls back to the simple WLS fit on the test engine if the
        reconstruction gives a worse fit or there are too few points.
        """
        m = len(self.engine_params_)
        if m == 0 or len(test_cycles) < 3:
            return self._direct_wls(test_cycles, test_hi)

        # Prepare test log-linear system
        hi_shifted = test_hi - self.phi_
        valid = hi_shifted > 0
        if valid.sum() < 3:
            return self._direct_wls(test_cycles, test_hi)

        x_T = np.log(test_cycles[valid])
        y_T = np.log(hi_shifted[valid])
        w_T = _geometric_weights(valid.sum(), self.weight_q)

        # Training parameter matrix: each row = [θ⁽⁰⁾ᵢ, θ⁽¹⁾ᵢ]
        param_keys = list(self.engine_params_.keys())
        Theta = np.array(
            [[self.engine_params_[k]["theta0"],
              self.engine_params_[k]["theta1"]]
             for k in param_keys]
        )   # shape (m, 2)

        # Design matrix for convex combination:
        # y_T ≈ X_T @ Theta.T @ w   where X_T = [1, xₖ]
        X_T = np.column_stack([np.ones_like(x_T), x_T])   # (n_valid, 2)
        A = X_T @ Theta.T   # (n_valid, m)

        # Weighted residual: minimise Σ w_T * (y_T - A @ w_mix)²
        W_diag = np.diag(w_T)

        def objective(w_mix: np.ndarray) -> float:
            residual = y_T - A @ w_mix
            return float(residual @ W_diag @ residual)

        def gradient(w_mix: np.ndarray) -> np.ndarray:
            residual = y_T - A @ w_mix
            return -2.0 * A.T @ W_diag @ residual

        # Constraints: Σ wᵢ = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * m
        w0 = np.ones(m) / m

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                w0,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-10},
            )

        if not result.success:
            return self._direct_wls(test_cycles, test_hi)

        w_opt = result.x
        # Reconstruct composite parameters
        theta0_rec = float(w_opt @ Theta[:, 0])
        theta1_rec = float(w_opt @ Theta[:, 1])

        if theta1_rec >= 0:
            return self._direct_wls(test_cycles, test_hi)

        return {
            "theta0": theta0_rec,
            "theta1": theta1_rec,
            "phi": self.phi_,
            "last_cycle": float(test_cycles[-1]),
            "last_hi": float(test_hi[-1]),
            "fit_ok": True,
            "weights": w_opt,          # expose for uncertainty propagation
            "param_keys": param_keys,
        }

    def _direct_wls(self, cycles: np.ndarray, hi: np.ndarray) -> dict:
        """Fallback: fit WLS directly on the test engine trajectory."""
        hi_shifted = hi - self.phi_
        valid = hi_shifted > 0
        if valid.sum() < 2:
            return {
                "theta0": 0.0, "theta1": -0.01, "phi": self.phi_,
                "last_cycle": float(cycles[-1]), "last_hi": float(hi[-1]),
                "fit_ok": False,
            }
        x = np.log(cycles[valid])
        y = np.log(hi_shifted[valid])
        w = _geometric_weights(valid.sum(), self.weight_q)
        theta0, theta1 = _wls_fit(x, y, w)
        theta1 = min(theta1, -1e-4)   # enforce decline
        return {
            "theta0": theta0, "theta1": theta1, "phi": self.phi_,
            "last_cycle": float(cycles[-1]), "last_hi": float(hi[-1]),
            "fit_ok": True,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────

    def _rul_from_params(self, params: dict) -> float:
        """Compute point RUL from a parameter dict."""
        t_fail = _extrapolate_failure_cycle(
            params["theta0"],
            params["theta1"],
            params["phi"],
            self.failure_threshold_,
            params["last_cycle"],
            self.max_extrapolation,
        )
        rul = max(0.0, t_fail - params["last_cycle"])
        if RUL_CAP is not None:
            return min(float(rul), float(RUL_CAP))
        return float(rul)

    def predict_test(self, test_df_with_hi: pd.DataFrame) -> np.ndarray:
        """
        Predict a point RUL estimate for each test engine.

        For each engine the observed HI trajectory is used to reconstruct
        a mixture of training-engine parameters, then RUL is extrapolated.

        Parameters
        ----------
        test_df_with_hi : DataFrame with [unit_id, cycle, HI]

        Returns
        -------
        np.ndarray of shape (n_test_engines,) with RUL estimates.
        """
        preds: list[float] = []
        for engine_id, group in test_df_with_hi.groupby("unit_id"):
            group = group.sort_values("cycle")
            cycles = group["cycle"].values.astype(float)
            hi = group["HI"].values.astype(float)

            params = self._reconstruct_params(cycles, hi)
            preds.append(self._rul_from_params(params))

        return np.array(preds)

    def predict_with_uncertainty(
        self,
        test_df_with_hi: pd.DataFrame,
        ci_level: float = 0.90,
    ) -> pd.DataFrame:
        """
        Predict RUL with uncertainty quantification (Monte Carlo bootstrap).

        For each test engine, perturbs the reconstructed parameter set by
        bootstrapping the residuals from fit to obtain a distribution of
        possible RUL values, then reports the mean and CI bounds.

        Parameters
        ----------
        test_df_with_hi : DataFrame with [unit_id, cycle, HI]
        ci_level        : confidence interval level (default 90%)

        Returns
        -------
        DataFrame with columns [unit_id, rul_mean, ci_lower, ci_upper]
        """
        alpha = (1.0 - ci_level) / 2.0
        records: list[dict] = []

        for engine_id, group in test_df_with_hi.groupby("unit_id"):
            group = group.sort_values("cycle")
            cycles = group["cycle"].values.astype(float)
            hi = group["HI"].values.astype(float)

            params = self._reconstruct_params(cycles, hi)
            rul_point = self._rul_from_params(params)

            # Bootstrap: perturb theta0/theta1 with their WLS residual noise
            hi_shifted = hi - self.phi_
            valid = hi_shifted > 0
            if valid.sum() >= 3:
                y_obs = np.log(hi_shifted[valid])
                t_valid = cycles[valid]
                y_hat = params["theta0"] + params["theta1"] * np.log(t_valid)
                residuals = y_obs - y_hat
                sigma_res = np.std(residuals) if len(residuals) > 1 else 0.1

                rng = np.random.default_rng(42)
                sample_ruls: list[float] = []
                for _ in range(self.n_bootstrap):
                    noise = rng.normal(0, sigma_res)
                    perturbed = {
                        **params,
                        "theta0": params["theta0"] + noise,
                    }
                    sample_ruls.append(self._rul_from_params(perturbed))

                ci_lo = float(np.quantile(sample_ruls, alpha))
                ci_hi = float(np.quantile(sample_ruls, 1.0 - alpha))
            else:
                ci_lo = max(0.0, rul_point * 0.7)
                ci_hi = rul_point * 1.3

            records.append({
                "unit_id": int(engine_id),
                "rul_mean": round(rul_point, 2),
                "ci_lower": round(ci_lo, 2),
                "ci_upper": round(ci_hi, 2),
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_params(self, path: Path | None = None) -> Path:
        """Serialise fitted engine parameters to JSON."""
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        path = path or METADATA_DIR / "stochastic_params.json"
        payload = {
            "phi": self.phi_,
            "failure_threshold": self.failure_threshold_,
            "weight_q": self.weight_q,
            "engine_params": {
                str(k): {
                    kk: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for kk, v in vv.items()
                    if kk not in ("weights", "param_keys")
                }
                for k, vv in self.engine_params_.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[StochasticDegradationRUL] Parameters saved → {path}")
        return path

    def load_params(self, path: Path) -> "StochasticDegradationRUL":
        """Load fitted engine parameters from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.phi_ = data["phi"]
        self.failure_threshold_ = data["failure_threshold"]
        self.weight_q = data.get("weight_q", self.weight_q)
        self.engine_params_ = {
            int(k): v for k, v in data["engine_params"].items()
        }
        return self
