"""
preprocessing.py
================
Clean, normalize, and prepare C-MAPSS data for modelling.

Key functions
-------------
drop_constant_sensors   : remove near-zero-variance sensors
add_rul_target          : compute RUL column (with optional cap)
split_by_engine         : train/validation split on engine ids
fit_scaler              : fit StandardScaler on training features
transform_features      : apply a fitted scaler to features
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    RUL_CAP,
    SCALERS_DIR,
    SENSORS_TO_DROP,
    VALIDATION_ENGINE_RATIO,
)


# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------

def drop_constant_sensors(
    df: pd.DataFrame,
    extra_drops: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove low-variance sensor columns.

    Parameters
    ----------
    df          : input DataFrame (train or test)
    extra_drops : additional column names to drop beyond the config defaults

    Returns
    -------
    DataFrame with uninformative sensor columns removed (in-place copy).
    """
    to_drop = list(SENSORS_TO_DROP)
    if extra_drops:
        to_drop += extra_drops
    # Only drop columns that actually exist
    to_drop = [c for c in to_drop if c in df.columns]
    return df.drop(columns=to_drop)


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of sensor column names present in *df*."""
    return [c for c in df.columns if c.startswith("sensor_")]


# ---------------------------------------------------------------------------
# RUL target generation
# ---------------------------------------------------------------------------

def add_rul_target(
    train_df: pd.DataFrame,
    rul_cap: int | None = RUL_CAP,
) -> pd.DataFrame:
    """
    Add a 'RUL' column to the training set.

    RUL = max_cycle_for_engine − current_cycle.
    Optionally capped at *rul_cap* to reduce target imbalance in early life.

    Parameters
    ----------
    train_df : training DataFrame (must contain `unit_id` and `cycle`)
    rul_cap  : upper cap on RUL values (None = no cap)

    Returns
    -------
    DataFrame with new 'RUL' column added.
    """
    df = train_df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    if rul_cap is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_cap)

    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_by_engine(
    train_df: pd.DataFrame,
    val_ratio: float = VALIDATION_ENGINE_RATIO,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train/validation sets by engine id.

    Splitting by engine (not by row) prevents temporal leakage.

    Parameters
    ----------
    train_df     : full training DataFrame
    val_ratio    : fraction of engine ids to reserve for validation
    random_state : random seed for reproducibility

    Returns
    -------
    (df_train, df_val) : two DataFrames
    """
    rng = np.random.default_rng(random_state)
    engine_ids = train_df["unit_id"].unique()
    rng.shuffle(engine_ids)
    n_val = max(1, int(len(engine_ids) * val_ratio))
    val_ids = set(engine_ids[:n_val])
    train_ids = set(engine_ids[n_val:])
    df_tr = train_df[train_df["unit_id"].isin(train_ids)].copy()
    df_va = train_df[train_df["unit_id"].isin(val_ids)].copy()
    print(
        f"[preprocessing] Split → train engines: {len(train_ids)}, "
        f"val engines: {len(val_ids)}"
    )
    return df_tr, df_va


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def apply_median_filter(df: pd.DataFrame, sensor_cols: list[str], window: int = 5) -> pd.DataFrame:
    """
    Apply a moving median filter to sensor columns per engine.
    This helps remove outliers/spikes while preserving trends.
    """
    df = df.copy()
    for col in sensor_cols:
        df[col] = df.groupby("unit_id")[col].transform(
            lambda x: x.rolling(window=window, min_periods=1, center=True).median()
        )
    return df


def select_correlated_sensors(
    df: pd.DataFrame,
    sensor_cols: list[str],
    target_col: str = "rul",
    threshold: float = 0.1,
) -> list[str]:
    """
    Filter sensor columns based on their absolute correlation with the target.
    This helps remove sensors that don't capture degradation patterns.
    """
    if target_col not in df.columns:
        return sensor_cols

    correlations = df[sensor_cols + [target_col]].corr()[target_col].abs()
    selected = [c for c in sensor_cols if correlations.get(c, 0) > threshold]

    # Log selection
    dropped = set(sensor_cols) - set(selected)
    if dropped:
        print(f"[preprocessing] Dropped weakly correlated sensors: {sorted(list(dropped))}")
    print(f"[preprocessing] Selected {len(selected)} sensors with corr > {threshold}")

    return selected


def fit_scaler(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    save_path: Path | None = None,
) -> StandardScaler:
    """
    Fit a StandardScaler on training features.

    Parameters
    ----------
    train_df     : training DataFrame
    feature_cols : columns to scale
    save_path    : if provided, serialize the scaler to disk

    Returns
    -------
    Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[preprocessing] Scaler saved → {save_path}")

    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Apply a fitted scaler to *feature_cols* in *df* (returns a copy).
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def load_scaler(path: Path) -> StandardScaler:
    """Load a serialized StandardScaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
