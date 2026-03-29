"""Unit tests for the XGBoostRUL wrapper model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.xgboost_model import XGBoostRUL


def test_xgboost_smoke():
    """Smoke test: train with synthetic data and verify prediction format."""
    np.random.seed(42)
    # 50 rows, 3 manually engineered features
    X = pd.DataFrame({
        "unit_id": [1]*50,
        "cycle": list(range(1, 51)),
        "feat_mean": np.random.randn(50),
        "feat_slope": np.random.randn(50) * 0.1,
    })
    # Target linearly dependent on cycle and noise
    y = pd.Series(130 - X["cycle"] + np.random.randn(50))
    feature_cols = ["feat_mean", "feat_slope"]
    
    # 1. Initialize
    model = XGBoostRUL(random_state=42)
    
    # 2. Fit without grid search (fast path for smoke tests or predefined parameters)
    model.fit(X, y, feature_cols=feature_cols, search=False)
    
    # 3. Predict
    preds = model.predict(X)
    
    # Assertions
    assert len(preds) == len(X)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()
    
    # Check feature importance extraction
    fi = model.feature_importance()
    assert len(fi) == len(feature_cols)
    assert "feature" in fi.columns
    assert "importance" in fi.columns
