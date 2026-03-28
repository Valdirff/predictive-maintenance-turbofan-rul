"""Unit tests for preprocessing.py"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_cmapss
from src.preprocessing import (
    add_rul_target,
    drop_constant_sensors,
    fit_scaler,
    get_sensor_columns,
    load_scaler,
    split_by_engine,
    transform_features,
)


@pytest.fixture(scope="module")
def train_df():
    df, _, _ = load_cmapss("FD001")
    return df


class TestDropConstantSensors:
    def test_removes_configured_sensors(self, train_df):
        cleaned = drop_constant_sensors(train_df)
        for col in ["sensor_1", "sensor_5", "sensor_6"]:
            assert col not in cleaned.columns

    def test_no_error_if_already_absent(self, train_df):
        # Should not raise even if extras don't exist
        cleaned = drop_constant_sensors(train_df, extra_drops=["nonexistent_column"])
        assert "unit_id" in cleaned.columns


class TestAddRulTarget:
    def test_rul_column_exists(self, train_df):
        result = add_rul_target(train_df)
        assert "RUL" in result.columns

    def test_rul_non_negative(self, train_df):
        result = add_rul_target(train_df)
        assert (result["RUL"] >= 0).all()

    def test_rul_cap_respected(self, train_df):
        cap = 125
        result = add_rul_target(train_df, rul_cap=cap)
        assert result["RUL"].max() <= cap

    def test_no_cap(self, train_df):
        result = add_rul_target(train_df, rul_cap=None)
        # Without cap, max RUL > 125 is possible
        assert result["RUL"].max() > 0

    def test_last_cycle_rul_is_zero(self, train_df):
        result = add_rul_target(train_df, rul_cap=None)
        for _, group in result.groupby("unit_id"):
            assert group["RUL"].iloc[-1] == 0


class TestSplitByEngine:
    def test_no_engine_overlap(self, train_df):
        df = add_rul_target(train_df)
        df_tr, df_va = split_by_engine(df)
        tr_ids = set(df_tr["unit_id"].unique())
        va_ids = set(df_va["unit_id"].unique())
        assert tr_ids.isdisjoint(va_ids)

    def test_combined_covers_all(self, train_df):
        df = add_rul_target(train_df)
        df_tr, df_va = split_by_engine(df)
        all_ids = set(df["unit_id"].unique())
        combined = set(df_tr["unit_id"].unique()) | set(df_va["unit_id"].unique())
        assert combined == all_ids


class TestScaler:
    def test_fit_and_transform(self, train_df, tmp_path):
        df = drop_constant_sensors(train_df)
        sensor_cols = get_sensor_columns(df)
        scaler = fit_scaler(df, sensor_cols, save_path=tmp_path / "scaler.pkl")
        transformed = transform_features(df, scaler, sensor_cols)
        means = transformed[sensor_cols].mean()
        assert (means.abs() < 0.1).all(), "Transformed features should be near zero mean."

    def test_scaler_save_and_load(self, train_df, tmp_path):
        df = drop_constant_sensors(train_df)
        sensor_cols = get_sensor_columns(df)
        path = tmp_path / "scaler.pkl"
        fit_scaler(df, sensor_cols, save_path=path)
        loaded = load_scaler(path)
        assert isinstance(loaded, StandardScaler)
