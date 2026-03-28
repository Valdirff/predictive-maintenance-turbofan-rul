"""Unit tests for data_loader.py"""

from __future__ import annotations

import pytest
import pandas as pd

from src.data_loader import load_cmapss
from src.config import COLUMN_NAMES


class TestLoadCmapss:
    def test_returns_three_objects(self):
        train_df, test_df, rul_true = load_cmapss("FD001")
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert isinstance(rul_true, pd.Series)

    def test_column_count(self):
        train_df, test_df, _ = load_cmapss("FD001")
        assert train_df.shape[1] == 26, f"Expected 26 columns, got {train_df.shape[1]}"
        assert test_df.shape[1] == 26

    def test_column_names(self):
        train_df, _, _ = load_cmapss("FD001")
        assert list(train_df.columns) == COLUMN_NAMES

    def test_sorted_by_unit_and_cycle(self):
        train_df, _, _ = load_cmapss("FD001")
        for _, group in train_df.groupby("unit_id"):
            cycles = group["cycle"].values
            assert all(cycles[i] <= cycles[i + 1] for i in range(len(cycles) - 1)), \
                "Cycles not sorted within engine"

    def test_no_nan(self):
        train_df, test_df, _ = load_cmapss("FD001")
        assert not train_df.isnull().values.any()
        assert not test_df.isnull().values.any()

    def test_rul_length_matches_test_engines(self):
        _, test_df, rul_true = load_cmapss("FD001")
        assert len(rul_true) == test_df["unit_id"].nunique()

    def test_fd001_engine_count(self):
        train_df, test_df, _ = load_cmapss("FD001")
        assert train_df["unit_id"].nunique() == 100
        assert test_df["unit_id"].nunique() == 100
