"""
xgboost_model.py
================
XGBoost regressor for RUL estimation using engineered rolling features.

The model operates on a *flat* feature representation: for each engine,
features are extracted from the last `window` cycles (mean, std, slope, etc.)
and the model predicts the RUL at that snapshot.

This approach is highly effective in practice because:
- It captures temporal trends through window statistics without requiring
  sequence modelling.
- XGBoost is robust to feature scale, handles nonlinear interactions, and
  trains in seconds.
- The feature importance output provides interpretability similar to a
  traditional model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

from src.config import MODELS_DIR, XGBOOST_CV_FOLDS, XGBOOST_PARAM_GRID


# ---------------------------------------------------------------------------
# XGBoostRUL
# ---------------------------------------------------------------------------

class XGBoostRUL:
    """
    XGBoost-based RUL regressor.

    Parameters
    ----------
    param_grid     : hyperparameter grid for GridSearchCV
    cv_folds       : number of cross-validation folds
    random_state   : seed for reproducibility
    """

    def __init__(
        self,
        param_grid: dict | None = None,
        cv_folds: int = XGBOOST_CV_FOLDS,
        random_state: int = 42,
    ) -> None:
        self.param_grid = param_grid or XGBOOST_PARAM_GRID
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model_: XGBRegressor | None = None
        self.feature_cols_: list[str] = []
        self.best_params_: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        feature_cols: list[str],
        search: bool = True,
        n_iter: int = 15,
    ) -> "XGBoostRUL":
        """
        Train the XGBoost model.

        Parameters
        ----------
        X_train     : feature DataFrame
        y_train     : RUL target array
        feature_cols: columns to use for training
        search      : if True, run RandomizedSearchCV
        n_iter      : number of iterations for RandomizedSearchCV

        Returns
        -------
        self
        """
        from sklearn.model_selection import KFold, RandomizedSearchCV
        
        self.feature_cols_ = feature_cols
        X = X_train[feature_cols].values
        y = np.asarray(y_train, dtype=float)

        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            n_estimators=400,  # Default if no search
            max_depth=6,
            learning_rate=0.05,
        )

        if search:
            print(f"[XGBoostRUL] Running RandomizedSearchCV (iter={n_iter}, cv={self.cv_folds}) …")
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Ensure the grid is compatible with RandomizedSearchCV
            # (Standard GridSearchCV dict is usually fine)
            rs = RandomizedSearchCV(
                base_model,
                self.param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state,
            )
            rs.fit(X, y)
            self.best_params_ = rs.best_params_
            self.model_ = rs.best_estimator_
            print(f"[XGBoostRUL] Best params: {self.best_params_}")
        else:
            self.model_ = base_model
            self.model_.fit(X, y)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict RUL values. Returns array clipped to [0, ∞)."""
        assert self.model_ is not None, "Model not fitted. Call .fit() first."
        preds = self.model_.predict(X[self.feature_cols_].values)
        return np.clip(preds, 0, None)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame with feature importances sorted descending."""
        assert self.model_ is not None, "Model not fitted."
        importances = self.model_.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Pickle the fitted model to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or MODELS_DIR / "xgboost_rul.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[XGBoostRUL] Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "XGBoostRUL":
        """Load a pickled XGBoostRUL from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
