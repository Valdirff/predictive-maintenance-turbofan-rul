"""
train_stochastic.py
===================
End-to-end pipeline:
    load FD001
    → preprocess
    → select sensors (monotonicity)
    → build HI via logistic regression (LogisticHIBuilder)
    → fit StochasticDegradationRUL
    → evaluate on test set (point + uncertainty)
    → save metrics and figures

Usage
-----
    python -m src.pipelines.train_stochastic
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.config import HI_LOGISTIC_N_SAMPLES, SUBSET
from src.data_loader import load_cmapss
from src.evaluation import Timer, evaluation_report, save_metrics
from src.health_indicator import LogisticHIBuilder, select_monotonic_sensors
from src.models.stochastic_model import StochasticDegradationRUL
from src.preprocessing import (
    add_rul_target,
    drop_constant_sensors,
    fit_scaler,
    get_sensor_columns,
    transform_features,
)
from src.visualization import plot_real_vs_predicted, plot_residuals


def run(subset: str = SUBSET) -> dict:
    print(f"\n{'='*60}")
    print(f"  Stochastic Degradation Model — {subset}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    train_df, test_df, rul_true = load_cmapss(subset)

    # ── 2. Clean sensors ──────────────────────────────────────────────────
    train_df = drop_constant_sensors(train_df)
    test_df  = drop_constant_sensors(test_df)

    # ── 3. Add RUL target (for training set evaluation only) ──────────────
    train_df = add_rul_target(train_df)

    # ── 4. Normalize sensors (z-score using training statistics) ──────────
    sensor_cols = get_sensor_columns(train_df)
    scaler = fit_scaler(
        train_df, sensor_cols,
        save_path=Path("artifacts/scalers/scaler_stochastic.pkl"),
    )
    train_df = transform_features(train_df, scaler, sensor_cols)
    test_df  = transform_features(test_df,  scaler, sensor_cols)

    # ── 5. Select monotonic sensors ───────────────────────────────────────
    top_sensors = select_monotonic_sensors(train_df, top_k=9)

    # ── 6. Build Health Indicator via logistic regression ──────────────────
    print("\n[Pipeline] Building Health Indicator (logistic)...")
    hi_builder = LogisticHIBuilder(sensors=top_sensors, n_samples=HI_LOGISTIC_N_SAMPLES)
    train_df["HI"] = hi_builder.fit_transform(train_df)
    test_df["HI"]  = hi_builder.transform(test_df)

    hi_stats = train_df.groupby("unit_id")["HI"].agg(["mean", "min", "max"])
    print(f"[Pipeline] Train HI stats (per-engine mean):\n{hi_stats.describe().round(3)}")

    # ── 7. Fit model ───────────────────────────────────────────────────────
    print("\n[Pipeline] Fitting StochasticDegradationRUL...")
    model = StochasticDegradationRUL()
    with Timer() as t_train:
        model.fit(train_df)
    model.save_params()

    # ── 8. Predict on test set ─────────────────────────────────────────────
    print("\n[Pipeline] Predicting RUL on test engines...")
    with Timer() as t_inf:
        y_pred = model.predict_test(test_df)

    y_true = rul_true.values

    # ── 9. Uncertainty quantification ─────────────────────────────────────
    print("[Pipeline] Computing uncertainty (90% CI)...")
    rul_uncertainty = model.predict_with_uncertainty(test_df, ci_level=0.90)
    mean_ci_width = (
        rul_uncertainty["ci_upper"] - rul_uncertainty["ci_lower"]
    ).mean()
    print(f"[Pipeline] Mean 90% CI width: {mean_ci_width:.1f} cycles")

    # ── 10. Evaluate ───────────────────────────────────────────────────────
    report = evaluation_report(
        model_name="Stochastic",
        y_true=y_true,
        y_pred=y_pred,
        train_time=t_train.elapsed,
        inference_time=t_inf.elapsed,
        extra={
            "interpretability": "High",
            "computational_cost": "Low-Medium",
            "uncertainty_quantification": True,
            "mean_ci_width_cycles": round(mean_ci_width, 2),
            "failure_threshold": round(model.failure_threshold_, 4),
            "phi": round(model.phi_, 4),
        },
    )
    save_metrics(report, filename="stochastic.json")
    print(f"\n[Results] RMSE={report['rmse']:.2f}  NASA Score={report['nasa_score']:.0f}")

    # ── 11. Figures ────────────────────────────────────────────────────────
    plot_real_vs_predicted(y_true, y_pred, "Stochastic")
    plot_residuals(y_true, y_pred, "Stochastic")

    return report


if __name__ == "__main__":
    run()
