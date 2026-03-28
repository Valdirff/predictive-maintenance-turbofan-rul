"""Unit tests for evaluation.py"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import mae, nasa_score, rmse, evaluation_report, save_metrics


class TestRmse:
    def test_perfect_prediction(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert rmse(y_true, y_pred) == pytest.approx(expected)


class TestNasaScore:
    def test_zero_error(self):
        y = np.array([50.0, 100.0])
        assert nasa_score(y, y) == pytest.approx(0.0)

    def test_late_prediction_penalised_more(self):
        """Late predictions (positive d) should score higher than symmetric early ones."""
        y_true = np.array([100.0])
        # 10 cycles late
        late  = nasa_score(y_true, np.array([110.0]))
        # 10 cycles early
        early = nasa_score(y_true, np.array([ 90.0]))
        assert late > early, "Late predictions must be penalised more than early ones."

    def test_positive_for_nonzero_errors(self):
        y_true = np.array([50.0, 50.0])
        y_pred = np.array([60.0, 40.0])
        assert nasa_score(y_true, y_pred) > 0


class TestMae:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == pytest.approx(2.0)


class TestEvaluationReport:
    def test_keys_present(self):
        y = np.array([10.0, 20.0])
        report = evaluation_report("TestModel", y, y, 1.0, 0.1)
        for key in ["model", "rmse", "nasa_score", "mae", "n_samples", "train_time_s", "inference_time_s"]:
            assert key in report

    def test_save_creates_file(self, tmp_path, monkeypatch):
        from src import evaluation as ev
        monkeypatch.setattr(ev, "METRICS_DIR", tmp_path)
        y = np.array([10.0, 20.0])
        report = evaluation_report("UnitTest", y, y, 0.5, 0.01)
        path = save_metrics(report)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model"] == "UnitTest"
