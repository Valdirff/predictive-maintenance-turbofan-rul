"""
evaluation.py
=============
Metrics and reporting utilities for RUL prognostics.

Implements:
  - RMSE (standard regression error)
  - NASA asymmetric scoring function (penalises late predictions more)
  - Timing utilities
  - evaluation_report: returns a dict ready for JSON export
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.config import METRICS_DIR


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    Defined in the original prognostics competition:
        d = y_pred - y_true
        s = sum(exp(-d/13) - 1) for d < 0   (early prediction)
          + sum(exp( d/10) - 1) for d >= 0  (late prediction)

    Late predictions are penalised more harshly than early ones.
    Lower is better.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum())


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager to measure wall-clock elapsed time (seconds)."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def evaluation_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    inference_time: float,
    extra: dict | None = None,
) -> dict:
    """
    Build a standardised evaluation report dict.

    Parameters
    ----------
    model_name      : human-readable model identifier
    y_true          : ground-truth RUL values
    y_pred          : predicted RUL values
    train_time      : wall-clock training time in seconds
    inference_time  : wall-clock inference time in seconds
    extra           : any additional fields to include

    Returns
    -------
    dict with keys: model, rmse, nasa_score, mae, n_samples, train_time_s,
                    inference_time_s, and any extra fields.
    """
    report = {
        "model": model_name,
        "rmse": round(rmse(y_true, y_pred), 4),
        "nasa_score": round(nasa_score(y_true, y_pred), 2),
        "mae": round(mae(y_true, y_pred), 4),
        "n_samples": int(len(y_true)),
        "train_time_s": round(train_time, 3),
        "inference_time_s": round(inference_time, 4),
    }
    if extra:
        report.update(extra)
    return report


def save_metrics(report: dict, filename: str | None = None) -> Path:
    """
    Persist an evaluation report to results/metrics/<model>.json.

    Parameters
    ----------
    report   : dict produced by evaluation_report()
    filename : override filename (default: <report['model']>.json)

    Returns
    -------
    Path to the saved file.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fname = filename or f"{report['model'].replace(' ', '_').lower()}.json"
    path = METRICS_DIR / fname
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[evaluation] Metrics saved → {path}")
    return path


def load_all_metrics() -> pd.DataFrame:
    """Load all JSON metric files from results/metrics/ into a DataFrame."""
    records = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return pd.DataFrame(records)
