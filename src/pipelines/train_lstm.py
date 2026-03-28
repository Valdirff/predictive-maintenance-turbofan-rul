"""
train_lstm.py
=============
End-to-end pipeline: load FD001 → preprocess → build sequences → train LSTM
→ evaluate on test set → save metrics, learning curves, and prediction scatter.

Usage
-----
    python -m src.pipelines.train_lstm
    # or
    python src/pipelines/train_lstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from src.config import LSTM_WINDOW, SUBSET
from src.data_loader import load_cmapss
from src.evaluation import Timer, evaluation_report, save_metrics
from src.models.lstm_model import LSTMTrainer, build_sequences, build_test_sequences
from src.preprocessing import (
    add_rul_target,
    apply_median_filter,
    drop_constant_sensors,
    fit_scaler,
    get_sensor_columns,
    select_correlated_sensors,
    split_by_engine,
    transform_features,
)
from src.visualization import plot_learning_curves, plot_real_vs_predicted, plot_residuals


def run(subset: str = SUBSET) -> dict:
    print(f"\n{'='*60}")
    print(f"  OPTIMIZED Attention-LSTM Model — {subset}")
    print(f"{'='*60}\n")

    # ── 1. Load data ──────────────────────────────────────────────
    train_df, test_df, rul_true = load_cmapss(subset)

    # ── 2. Clean & Pre-process ────────────────────────────────────
    train_df = drop_constant_sensors(train_df)
    test_df  = drop_constant_sensors(test_df)

    sensor_cols = get_sensor_columns(train_df)

    # Apply Median Filter (Artigo 1: Noise reduction)
    print(f"[train_lstm] Applying Moving Median Filter (window=5) …")
    train_df = apply_median_filter(train_df, sensor_cols, window=5)
    test_df  = apply_median_filter(test_df,  sensor_cols, window=5)

    # ── 3. Target ─────────────────────────────────────────────────
    train_df = add_rul_target(train_df)

    # ── 4. Feature Selection & Augmentation ───────────────────────
    # Filter sensors based on correlation (Artigo 1)
    selected_sensors = select_correlated_sensors(train_df, sensor_cols, threshold=0.1)
    
    # Add Hybrid Rolling Features (Hints for trend, like XGBoost but sequential)
    print(f"[train_lstm] Adding rolling Mean, Std, and Slope (window={LSTM_WINDOW}) …")
    
    def get_slope(y):
        if len(y) < 2: return 0.0
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    for col in selected_sensors:
        # Mean (Trend center)
        roll = lambda df: df.groupby("unit_id")[col].rolling(window=LSTM_WINDOW, min_periods=1)
        train_df[f"{col}_mean"] = train_df.groupby("unit_id")[col].transform(lambda x: x.rolling(LSTM_WINDOW, 1).mean())
        test_df[f"{col}_mean"]  = test_df.groupby("unit_id")[col].transform(lambda x: x.rolling(LSTM_WINDOW, 1).mean())
        
        # Std (Volatility/Noise)
        train_df[f"{col}_std"] = train_df.groupby("unit_id")[col].transform(lambda x: x.rolling(LSTM_WINDOW, 1).std().fillna(0))
        test_df[f"{col}_std"]  = test_df.groupby("unit_id")[col].transform(lambda x: x.rolling(LSTM_WINDOW, 1).std().fillna(0))
        
        # Slope (Rate of degradation) - using smaller window for reactivity
        train_df[f"{col}_slope"] = train_df.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=10, min_periods=2).apply(get_slope, raw=True).fillna(0)
        )
        test_df[f"{col}_slope"] = test_df.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=10, min_periods=2).apply(get_slope, raw=True).fillna(0)
        )

    # Re-evaluate feature column list (Raw selected + Mean + Std + Slope)
    all_features = selected_sensors.copy()
    for col in selected_sensors:
        all_features.extend([f"{col}_mean", f"{col}_std", f"{col}_slope"])
        
    print(f"[train_lstm] Total features in sequence: {len(all_features)}")

    # ── 5. Train/val split ────────────────────────────────────────
    df_tr, df_va = split_by_engine(train_df)

    # ── 6. Normalize ──────────────────────────────────────────────
    scaler = fit_scaler(
        df_tr, all_features,
        save_path=Path("artifacts/scalers/scaler_lstm_opt.pkl"),
    )
    df_tr   = transform_features(df_tr,   scaler, all_features)
    df_va   = transform_features(df_va,   scaler, all_features)
    test_df = transform_features(test_df, scaler, all_features)

    # ── 7. Build sequences ────────────────────────────────────────
    print(f"[train_lstm] Building sequences (window={LSTM_WINDOW}) …")
    X_tr, y_tr = build_sequences(df_tr, window_size=LSTM_WINDOW, feature_cols=all_features)
    X_va, y_va = build_sequences(df_va, window_size=LSTM_WINDOW, feature_cols=all_features)
    X_te       = build_test_sequences(test_df, window_size=LSTM_WINDOW, feature_cols=all_features)

    print(f"[train_lstm] X_train: {X_tr.shape}  X_val: {X_va.shape}  X_test: {X_te.shape}")

    # ── 8. Train Attention-LSTM ───────────────────────────────────
    n_features = X_tr.shape[2]
    trainer = LSTMTrainer(input_size=n_features)

    with Timer() as t_train:
        trainer.fit(X_tr, y_tr, X_va, y_va)

    # ── 9. Predict ────────────────────────────────────────────────
    with Timer() as t_inf:
        y_pred = trainer.predict(X_te)

    y_true = rul_true.values

    # ── 10. Evaluate ──────────────────────────────────────────────
    report = evaluation_report(
        model_name="LSTM_Optimized",
        y_true=y_true,
        y_pred=y_pred,
        train_time=t_train.elapsed,
        inference_time=t_inf.elapsed,
        extra={
            "interpretability": "Medium (Attention)",
            "computational_cost": "High",
            "window_size": LSTM_WINDOW,
            "features": "Raw + Rolling Mean",
            "sensor_selection": "Correlation > 0.1",
        },
    )
    save_metrics(report)
    trainer.save()
    print(f"\n[Results] RMSE={report['rmse']:.2f}  NASA Score={report['nasa_score']:.0f}")

    # ── 11. Figures ───────────────────────────────────────────────
    plot_learning_curves(trainer.history, filename="lstm/learning_curves.png")
    plot_real_vs_predicted(y_true, y_pred, "LSTM_Optimized", filename="lstm/real_vs_pred.png")
    plot_residuals(y_true, y_pred, "LSTM_Optimized", filename="lstm/residuals.png")

    return report


if __name__ == "__main__":
    run()
