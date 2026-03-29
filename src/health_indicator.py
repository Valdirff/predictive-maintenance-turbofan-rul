"""
health_indicator.py
===================
Construct a scalar Health Indicator (HI) from multi-sensor C-MAPSS data.

The HI aggregates the most degradation-informative sensors into a single
monotonic signal scaled [0, 1] (1 = healthy, 0 = failed). This signal is
consumed directly by both the old ExponentialRUL model and the new
StochasticDegradationRUL model.

Construction methods:
  - 'pca'      : first principal component of selected sensors (sign-corrected)
  - 'weighted' : variance-weighted average of selected sensors
  - 'logistic' : fleet-level logistic regression on healthy/failure samples
                 (Maulana et al., 2023 style) — use LogisticHIBuilder class
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.config import HI_METHOD, HI_SMOOTHING_WINDOW
from src.preprocessing import get_sensor_columns


# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------

def select_monotonic_sensors(
    train_df: pd.DataFrame,
    top_k: int = 9,
) -> list[str]:
    """
    Rank sensors by |Spearman correlation with cycle| and return the top-k.

    Higher |ρ| means the sensor trends monotonically with degradation,
    making it more useful for health indicator construction.
    """
    sensor_cols = get_sensor_columns(train_df)
    correlations: dict[str, float] = {}

    for col in sensor_cols:
        rho, _ = spearmanr(train_df["cycle"], train_df[col])
        correlations[col] = abs(rho)

    ranked = sorted(correlations, key=correlations.get, reverse=True)  # type: ignore[arg-type]
    selected = ranked[:top_k]

    return selected


# ---------------------------------------------------------------------------
# HI construction — PCA / Weighted (legacy methods, kept for compatibility)
# ---------------------------------------------------------------------------

def _smooth(series: pd.Series, window: int) -> pd.Series:
    """Apply a rolling mean to reduce sensor noise."""
    return series.rolling(window=window, min_periods=1, center=False).mean()


def build_hi(
    df: pd.DataFrame,
    sensors: list[str],
    method: str = HI_METHOD,
    smoothing_window: int = HI_SMOOTHING_WINDOW,
) -> pd.Series:
    """
    Compute a scalar Health Indicator (HI) for every row in *df*.

    HI ≈ 1  →  healthy (beginning of life)
    HI ≈ 0  →  degraded (near failure)

    For method='logistic', use LogisticHIBuilder instead.
    """
    if method == "logistic":
        raise ValueError(
            "method='logistic' requires a fitted LogisticHIBuilder. "
            "Use LogisticHIBuilder.fit().transform() instead."
        )

    data = df[sensors].copy()
    for col in sensors:
        data[col] = _smooth(data[col], smoothing_window)

    if method == "pca":
        pca = PCA(n_components=1)
        raw_hi = pca.fit_transform(data.values).ravel()
    elif method == "weighted":
        weights = data.std(axis=0).values
        weights = weights / (weights.sum() + 1e-8)
        raw_hi = data.values @ weights
    else:
        raise ValueError(
            f"Unknown HI method: {method!r}. Choose 'pca', 'weighted', or 'logistic'."
        )

    scaler = MinMaxScaler()
    hi_scaled = scaler.fit_transform(raw_hi.reshape(-1, 1)).ravel()

    first_cycle_mask = df["cycle"] <= df["cycle"].quantile(0.05)
    if hi_scaled[first_cycle_mask].mean() < 0.5:
        hi_scaled = 1.0 - hi_scaled

    return pd.Series(hi_scaled, index=df.index, name="HI")


def add_hi_column(
    df: pd.DataFrame,
    sensors: list[str],
    method: str = HI_METHOD,
    smoothing_window: int = HI_SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """Convenience wrapper: add 'HI' column to a copy of *df*."""
    df = df.copy()
    df["HI"] = build_hi(df, sensors, method, smoothing_window)
    return df


# ---------------------------------------------------------------------------
# HI construction — Logistic (Maulana et al., 2023 style)
# ---------------------------------------------------------------------------

class LogisticHIBuilder:
    """
    Constructs a calibrated Health Indicator using logistic regression
    trained on per-engine healthy-vs-failure state samples.

    Inspired by Maulana et al. (2023, Machines).

    The logit function:
        g(F) = α + Σ(βⱼ Fⱼ)
    is fitted per engine and then averaged over the fleet, so that:
        HI(t) = sigmoid(g(F(t))) ∈ [0, 1]
    with 1 ≈ healthy and 0 ≈ failed.

    Parameters
    ----------
    sensors   : list of sensor column names used as features
    n_samples : number of healthy / failure cycles sampled per engine
    """

    def __init__(self, sensors: list[str], n_samples: int = 5) -> None:
        self.sensors = sensors
        self.n_samples = n_samples
        self.alpha_: float | None = None          # fleet-mean intercept
        self.betas_: np.ndarray | None = None     # fleet-mean coefficients
        self._alpha_per_engine: dict[int, float] = {}
        self._beta_per_engine: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "LogisticHIBuilder":
        """
        Fit logistic regression from per-engine healthy/failure samples.

        Parameters
        ----------
        train_df : training DataFrame with [unit_id, cycle, *sensors]

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression

        alphas: list[float] = []
        betas: list[np.ndarray] = []

        for engine_id, group in train_df.groupby("unit_id"):
            group = group.sort_values("cycle")
            n = self.n_samples

            healthy_rows = group.head(n)[self.sensors].values
            failure_rows = group.tail(n)[self.sensors].values

            X_eng = np.vstack([healthy_rows, failure_rows])
            y_eng = np.array([1] * n + [0] * n)

            if np.std(X_eng) < 1e-9:
                continue

            try:
                lr = LogisticRegression(
                    solver="lbfgs", max_iter=2000, C=1.0, random_state=42
                )
                lr.fit(X_eng, y_eng)
                # Ensure class order [0, 1]
                if list(lr.classes_) != [0, 1]:
                    continue
                a = float(lr.intercept_[0])
                b = lr.coef_[0].copy()
                alphas.append(a)
                betas.append(b)
                self._alpha_per_engine[int(engine_id)] = a
                self._beta_per_engine[int(engine_id)] = b
            except Exception:
                continue

        if not alphas:
            raise RuntimeError(
                "[LogisticHIBuilder] No engines produced valid fits. "
                "Verify sensor variance after normalization."
            )

        self.alpha_ = float(np.mean(alphas))
        self.betas_ = np.mean(np.vstack(betas), axis=0)


        return self

    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the fleet logistic model to compute HI for every row.

        HI = sigmoid(α + F @ β)

        Parameters
        ----------
        df : DataFrame with sensor columns (normalized)

        Returns
        -------
        pd.Series of HI values ∈ [0, 1], same index as df.
        """
        if self.alpha_ is None or self.betas_ is None:
            raise RuntimeError("Call fit() before transform().")

        X = df[self.sensors].values
        logit = self.alpha_ + X @ self.betas_
        hi = 1.0 / (1.0 + np.exp(-logit))

        # Safety clamp to strict (0,1) range
        hi = np.clip(hi, 1e-6, 1.0 - 1e-6)

        # Ensure HI=1 early, HI→0 near failure
        first_mask = df["cycle"] <= df["cycle"].quantile(0.05)
        if hi[first_mask].mean() < 0.5:
            hi = 1.0 - hi

        return pd.Series(hi, index=df.index, name="HI")

    # ------------------------------------------------------------------

    def fit_transform(self, train_df: pd.DataFrame) -> pd.Series:
        """Convenience: fit on train_df and return transformed HI."""
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------------

    def get_fleet_params(self) -> dict:
        """Return fleet-level logistic parameters as dict."""
        if self.alpha_ is None:
            raise RuntimeError("Call fit() first.")
        return {
            "alpha": self.alpha_,
            "betas": self.betas_.tolist(),
            "n_engines_fitted": len(self._alpha_per_engine),
        }
