"""
test_stochastic_model.py
========================
Unit and integration tests for StochasticDegradationRUL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.health_indicator import LogisticHIBuilder
from src.models.stochastic_model import (
    StochasticDegradationRUL,
    _geometric_weights,
    _wls_fit,
    _trajectory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_train_df(n_engines: int = 8, max_cycles: int = 120) -> pd.DataFrame:
    """
    Create a synthetic training DataFrame with deterministic degradation.
    HI(t) ≈ 1.0 − 0.008*t  (linear decay from 1 to ~0.04)
    """
    records = []
    rng = np.random.default_rng(0)
    for eid in range(1, n_engines + 1):
        n = rng.integers(80, max_cycles + 1)
        for c in range(1, int(n) + 1):
            hi = max(0.05, 1.0 - 0.008 * c + rng.normal(0, 0.01))
            records.append({"unit_id": eid, "cycle": c, "HI": hi})
    return pd.DataFrame(records)


def _make_test_df(n_engines: int = 5, max_cycles: int = 60) -> pd.DataFrame:
    """Shorter trajectories for test engines (truncated mid-life)."""
    records = []
    rng = np.random.default_rng(1)
    for eid in range(1, n_engines + 1):
        n = rng.integers(30, max_cycles + 1)
        for c in range(1, int(n) + 1):
            hi = max(0.05, 1.0 - 0.008 * c + rng.normal(0, 0.01))
            records.append({"unit_id": eid, "cycle": c, "HI": hi})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Helper function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_geometric_weights_sum_to_one(self):
        for n in [1, 5, 20]:
            w = _geometric_weights(n, q=1.2)
            assert abs(w.sum() - 1.0) < 1e-9, f"Weights don't sum to 1 for n={n}"

    def test_geometric_weights_increasing(self):
        w = _geometric_weights(10, q=1.3)
        assert all(w[i] <= w[i + 1] for i in range(len(w) - 1)), \
            "Weights should be non-decreasing with q > 1"

    def test_wls_fit_recovers_known_line(self):
        """WLS should exactly recover a noiseless linear model."""
        x = np.linspace(0.1, 5.0, 50)
        y = 2.0 + 0.5 * x   # known: theta0=2, theta1=0.5
        w = np.ones(50) / 50
        theta0, theta1 = _wls_fit(x, y, w)
        assert abs(theta0 - 2.0) < 1e-6
        assert abs(theta1 - 0.5) < 1e-6

    def test_trajectory_shape(self):
        t = np.array([1.0, 10.0, 50.0, 100.0])
        hi = _trajectory(t, theta0=0.5, theta1=-0.3, phi=0.02)
        assert hi.shape == (4,)
        # Trajectory should be decreasing (theta1 < 0)
        assert hi[0] > hi[-1], "Trajectory with negative theta1 should decline"


# ─────────────────────────────────────────────────────────────────────────────
# StochasticDegradationRUL tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStochasticDegradationRUL:
    @pytest.fixture(scope="class")
    def train_df(self):
        return _make_train_df()

    @pytest.fixture(scope="class")
    def test_df(self):
        return _make_test_df()

    @pytest.fixture(scope="class")
    def fitted_model(self, train_df):
        model = StochasticDegradationRUL(n_bootstrap=50)
        model.fit(train_df)
        return model

    # ── fit() ────────────────────────────────────────────────────────────

    def test_fit_does_not_raise(self, train_df):
        StochasticDegradationRUL().fit(train_df)

    def test_fit_sets_phi(self, fitted_model):
        assert 0.0 <= fitted_model.phi_ <= 0.5

    def test_fit_estimates_failure_threshold(self, fitted_model):
        ft = fitted_model.failure_threshold_
        assert 0.0 < ft < 1.0, f"Failure threshold {ft} out of (0,1)"

    def test_fit_populates_engine_params(self, fitted_model, train_df):
        n_engines = train_df["unit_id"].nunique()
        assert len(fitted_model.engine_params_) == n_engines

    def test_theta1_negative_for_ok_engines(self, fitted_model):
        """All 'fit_ok' engines must have a declining trajectory (theta1 < 0)."""
        for params in fitted_model.engine_params_.values():
            if params["fit_ok"]:
                assert params["theta1"] < 0, (
                    f"theta1={params['theta1']} should be < 0 for a declining HI"
                )

    # ── predict_test() ────────────────────────────────────────────────────

    def test_predict_test_returns_correct_shape(self, fitted_model, test_df):
        preds = fitted_model.predict_test(test_df)
        n_test = test_df["unit_id"].nunique()
        assert preds.shape == (n_test,), \
            f"Expected shape ({n_test},), got {preds.shape}"

    def test_predict_test_non_negative(self, fitted_model, test_df):
        preds = fitted_model.predict_test(test_df)
        assert (preds >= 0).all(), "RUL predictions must be non-negative"

    def test_predict_test_within_reasonable_bounds(self, fitted_model, test_df):
        preds = fitted_model.predict_test(test_df)
        # Engines seen for at most 60 cycles — RUL should not exceed 600
        assert preds.max() < 600, f"Unreasonably large RUL: {preds.max():.1f}"

    # ── predict_with_uncertainty() ────────────────────────────────────────

    def test_uncertainty_dataframe_columns(self, fitted_model, test_df):
        result = fitted_model.predict_with_uncertainty(test_df)
        assert set(result.columns) >= {"unit_id", "rul_mean", "ci_lower", "ci_upper"}

    def test_uncertainty_ci_ordering(self, fitted_model, test_df):
        result = fitted_model.predict_with_uncertainty(test_df)
        assert (result["ci_lower"] <= result["rul_mean"]).all(), \
            "ci_lower should be ≤ rul_mean"
        assert (result["rul_mean"] <= result["ci_upper"]).all(), \
            "rul_mean should be ≤ ci_upper"

    def test_uncertainty_non_negative(self, fitted_model, test_df):
        result = fitted_model.predict_with_uncertainty(test_df)
        assert (result["ci_lower"] >= 0).all()
        assert (result["rul_mean"] >= 0).all()

    # ── save / load ────────────────────────────────────────────────────────

    def test_save_and_load(self, fitted_model, test_df, tmp_path):
        path = tmp_path / "stochastic_params.json"
        fitted_model.save_params(path)

        model2 = StochasticDegradationRUL()
        model2.load_params(path)

        preds_orig = fitted_model.predict_test(test_df)
        preds_loaded = model2.predict_test(test_df)
        np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# LogisticHIBuilder tests (integration with stochastic model)
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticHIBuilder:
    @pytest.fixture(scope="class")
    def sensor_df(self):
        """Synthetic sensor DataFrame for n engines."""
        rng = np.random.default_rng(42)
        records = []
        for eid in range(1, 10):
            for c in range(1, 80):
                records.append({
                    "unit_id": eid,
                    "cycle": c,
                    "sensor_2":  rng.normal(0.5 - 0.004 * c, 0.02),
                    "sensor_3":  rng.normal(0.3 - 0.003 * c, 0.01),
                    "sensor_4":  rng.normal(0.7 - 0.006 * c, 0.03),
                })
        return pd.DataFrame(records)

    def test_fit_returns_self(self, sensor_df):
        builder = LogisticHIBuilder(
            sensors=["sensor_2", "sensor_3", "sensor_4"],
            n_samples=5,
        )
        result = builder.fit(sensor_df)
        assert result is builder

    def test_transform_range(self, sensor_df):
        builder = LogisticHIBuilder(
            sensors=["sensor_2", "sensor_3", "sensor_4"],
            n_samples=5,
        )
        hi = builder.fit_transform(sensor_df)
        assert hi.min() >= 0.0 and hi.max() <= 1.0, \
            "HI must be in [0, 1] after logistic transform"

    def test_hi_column_added(self, sensor_df):
        builder = LogisticHIBuilder(
            sensors=["sensor_2", "sensor_3", "sensor_4"],
            n_samples=5,
        )
        hi = builder.fit_transform(sensor_df)
        assert len(hi) == len(sensor_df)

    def test_stochastic_model_accepts_logistic_hi(self, sensor_df):
        """End-to-end: LogisticHIBuilder → StochasticDegradationRUL."""
        sensors = ["sensor_2", "sensor_3", "sensor_4"]
        builder = LogisticHIBuilder(sensors=sensors, n_samples=5)
        df = sensor_df.copy()
        df["HI"] = builder.fit_transform(df)

        model = StochasticDegradationRUL(n_bootstrap=20)
        model.fit(df)

        # Use some engines as pseudo-test
        test_df = df[df["unit_id"].isin([1, 2])].copy()
        preds = model.predict_test(test_df)
        assert preds.shape == (2,)
        assert (preds >= 0).all()
