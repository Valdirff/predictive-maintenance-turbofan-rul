"""
train_xgboost.py
================
End-to-end pipeline: load FD001 → preprocess → feature engineering →
train XGBoost → evaluate on test set → save metrics, feature importance
plot, and prediction scatter.

Usage
-----
    python -m src.pipelines.train_xgboost
    # or
    python src/pipelines/train_xgboost.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, SUBSET, XGBOOST_WINDOW
from src.data_loader import load_cmapss
from src.evaluation import Timer, evaluation_report, save_metrics
from src.feature_engineering import build_rolling_features, get_feature_cols
from src.models.xgboost_model import XGBoostRUL
from src.preprocessing import (
    add_rul_target,
    drop_constant_sensors,
    fit_scaler,
    get_sensor_columns,
    split_by_engine,
    transform_features,
)
from src.visualization import plot_real_vs_predicted, plot_residuals


def _plot_feature_importance(model: XGBoostRUL, top_n: int = 15) -> None:
    """Save a horizontal bar chart of the top-N feature importances."""
    fi = model.feature_importance().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="magma", ax=ax)
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_ylabel("")
    fig.tight_layout()
    
    out_dir = FIGURES_DIR / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "xgboost_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_xgboost] Feature importance saved → {path}")


def run(subset: str = SUBSET) -> dict:
    print(f"\n{'='*60}")
    print(f"  XGBoost Regressor — {subset}")
    print(f"  (Methodology: Feature-Engineered Gradient Boosting)")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────
    train_df, test_df, rul_true = load_cmapss(subset)

    # ── 2. Clean ──────────────────────────────────────────────────
    train_df = drop_constant_sensors(train_df)
    test_df  = drop_constant_sensors(test_df)

    # ── 3. Target ─────────────────────────────────────────────────
    # Piecewise linear RUL (Threshold=130 from config)
    train_df = add_rul_target(train_df)

    # ── 4. Train/val split by engine (70:30) ──────────────────────
    df_tr, df_va = split_by_engine(train_df, val_ratio=0.3)

    # ── 5. Normalize (fit on train split only) ────────────────────
    sensor_cols = get_sensor_columns(train_df)
    scaler = fit_scaler(
        df_tr, sensor_cols,
        save_path=Path("artifacts/scalers/scaler_xgboost.pkl"),
    )
    df_tr   = transform_features(df_tr,   scaler, sensor_cols)
    df_va   = transform_features(df_va,   scaler, sensor_cols)
    test_df = transform_features(test_df, scaler, sensor_cols)

    # ── 6. Feature engineering ────────────────────────────────────
    # Use ALL CYCLES for training/validation to maximize data
    feat_tr   = build_rolling_features(df_tr,  window=XGBOOST_WINDOW, all_cycles=True)
    feat_va   = build_rolling_features(df_va,  window=XGBOOST_WINDOW, all_cycles=True)
    
    # Use LAST CYCLE only for test evaluation (NASA standard)
    feat_test = build_rolling_features(test_df, window=XGBOOST_WINDOW, all_cycles=False)

    feature_cols = get_feature_cols(feat_tr)
    print(f"[train_xgboost] Using {len(feature_cols)} features for model training.")

    # ── 7. Train XGBoost ──────────────────────────────────────────
    model = XGBoostRUL(random_state=42)
    with Timer() as t_train:
        # RandomizedSearchCV is now default in .fit()
        model.fit(feat_tr, feat_tr["RUL"], feature_cols=feature_cols, search=True, n_iter=20)

    # ── 8. Predict ────────────────────────────────────────────────
    with Timer() as t_inf:
        y_pred = model.predict(feat_test)

    print(f"[train_xgboost] First 5 predictions: {y_pred[:5]}")
    y_true = rul_true.values

    # ── 9. Evaluate ───────────────────────────────────────────────
    report = evaluation_report(
        model_name="XGBoost",
        y_true=y_true,
        y_pred=y_pred,
        train_time=t_train.elapsed,
        inference_time=t_inf.elapsed,
        extra={
            "interpretability": "High (Feature Importance)",
            "computational_cost": "Low-Medium",
            "n_features": len(feature_cols),
            "best_params": model.best_params_,
            "training_samples": len(feat_tr),
        },
    )
    save_metrics(report)
    model.save()
    print(f"\n[Results] RMSE={report['rmse']:.2f}  NASA Score={report['nasa_score']:.0f}")

    # ── 10. Figures ───────────────────────────────────────────────
    _plot_feature_importance(model)
    
    # Corrected: Use 'filename' with relative subfolder path
    plot_real_vs_predicted(y_true, y_pred, "XGBoost", filename="xgboost/real_vs_pred_xgboost.png")
    plot_residuals(y_true, y_pred, "XGBoost", filename="xgboost/residuals_xgboost.png")

    return report


if __name__ == "__main__":
    run()
