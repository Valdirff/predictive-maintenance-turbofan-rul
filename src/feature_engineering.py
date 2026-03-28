"""
feature_engineering.py
=======================
Build temporal and rolling features from C-MAPSS sensor data for use by the
XGBoost model (and optionally other tabular learners).

The primary output is a *flat* feature DataFrame where each row represents the
last observed cycle of a given engine (i.e. one row per engine for training /
prediction), with features derived from a rolling window over the engine's
full history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import XGBOOST_WINDOW
from src.preprocessing import get_sensor_columns


# ---------------------------------------------------------------------------
# Rolling feature helpers
# ---------------------------------------------------------------------------

def _slope(series: pd.Series) -> float:
    """Least-squares slope of a 1-D series over equally-spaced time steps."""
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = series.values.astype(float)
    y -= y.mean()
    denom = (x ** 2).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def _ewma_last(series: pd.Series, span: int = 10) -> float:
    """Last value of an exponential weighted moving average."""
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Per-engine feature extraction
# ---------------------------------------------------------------------------

def _extract_engine_features(
    group: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
    all_cycles: bool = False,
    target_col: str | None = "RUL",
) -> list[dict]:
    """
    Extract features from an engine's timeline.
    If all_cycles is False, returns only the last cycle's features.
    If all_cycles is True, returns features for all cycles >= window.
    """
    records = []
    total_len = len(group)
    
    # Cycles to predict/train on
    if all_cycles:
        indices = range(window, total_len + 1)
    else:
        indices = [total_len]
        
    for i in indices:
        # Window of rows up to current index 'i'
        chunk = group.iloc[max(0, i-window):i]
        
        feats: dict[str, float] = {}
        feats["unit_id"] = int(group["unit_id"].iloc[0])
        feats["cycle"] = float(group["cycle"].iloc[i-1])
        feats["n_cycles"] = float(i)
        
        for col in sensor_cols:
            s = chunk[col]
            feats[f"{col}_mean"]  = float(s.mean())
            feats[f"{col}_std"]   = float(s.std(ddof=0))
            feats[f"{col}_min"]   = float(s.min())
            feats[f"{col}_max"]   = float(s.max())
            feats[f"{col}_slope"] = _slope(s)
            feats[f"{col}_ewma"]  = _ewma_last(s)
            feats[f"{col}_delta"] = float(s.iloc[-1] - s.iloc[0])
            
        if target_col and target_col in group.columns:
            feats[target_col] = float(group[target_col].iloc[i-1])
            
        records.append(feats)
        
    return records


def build_rolling_features(
    df: pd.DataFrame,
    window: int = XGBOOST_WINDOW,
    target_col: str | None = "RUL",
    all_cycles: bool = False,
) -> pd.DataFrame:
    """
    Build a flat feature matrix.
    
    Parameters
    ----------
    df         : preprocessed DataFrame
    window     : rolling window size
    target_col : if present, include in output
    all_cycles : if True, generate one row per valid cycle; 
                 if False, only one row per engine (last cycle)
                 
    Returns
    -------
    DataFrame with engineered features.
    """
    sensor_cols = get_sensor_columns(df)
    all_records: list[dict] = []
    
    print(f"[feature_engineering] Building features (window={window}, all_cycles={all_cycles}) ...")
    
    for engine_id, group in df.groupby("unit_id"):
        records = _extract_engine_features(group, sensor_cols, window, all_cycles, target_col)
        all_records.extend(records)
        
    result = pd.DataFrame(all_records)
    # Use (unit_id, cycle) as a unique index if all_cycles is True, 
    # but we will return it with unit_id in columns for easier grouping later
    return result


def get_feature_cols(feature_df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return feature column names, excluding target and metadata columns."""
    skip = set(exclude or []) | {"RUL", "unit_id", "cycle"}
    return [str(c) for c in feature_df.columns if c not in skip]
