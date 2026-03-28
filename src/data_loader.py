"""
data_loader.py
==============
Load NASA C-MAPSS train, test, and RUL files for any subset (FD001–FD004).
Columns are named, records sorted by unit_id and cycle, and a basic integrity
check is performed on load.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.config import COLUMN_NAMES, RAW_DIR, ROOT_DIR


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_extracted() -> None:
    """Extract CMAPSSData.zip into data/raw/CMAPSSData/ if files are missing."""
    expected = RAW_DIR / "train_FD001.txt"
    if not expected.exists():
        zip_path = ROOT_DIR / "CMAPSSData.zip"
        if not zip_path.exists():
            raise FileNotFoundError(
                f"Neither {expected} nor {zip_path} found. "
                "Please place CMAPSSData.zip in the project root."
            )
        print(f"[data_loader] Extracting {zip_path} → {RAW_DIR.parent} …")
        RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR.parent)
        print("[data_loader] Extraction complete.")


def _load_txt(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS file, strip trailing NaN columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # The raw files sometimes have trailing whitespace → extra NaN columns
    df = df.dropna(axis=1, how="all")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cmapss(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load one C-MAPSS subset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.

    Returns
    -------
    train_df : pd.DataFrame
        Training sequences (run-to-failure). 26 columns.
    test_df : pd.DataFrame
        Test sequences (truncated before failure). 26 columns.
    rul_true : pd.Series
        Ground-truth RUL for each test engine (index = engine id starting at 1).
    """
    _ensure_extracted()

    train_path = RAW_DIR / f"train_{subset}.txt"
    test_path  = RAW_DIR / f"test_{subset}.txt"
    rul_path   = RAW_DIR / f"RUL_{subset}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    train_df = _load_txt(train_path)
    test_df  = _load_txt(test_path)
    rul_raw  = _load_txt(rul_path)

    # Assign column names
    train_df.columns = COLUMN_NAMES[: len(train_df.columns)]
    test_df.columns  = COLUMN_NAMES[: len(test_df.columns)]

    # Sort by engine and cycle
    for df in (train_df, test_df):
        df.sort_values(["unit_id", "cycle"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # RUL ground truth — one scalar per test engine
    rul_true = rul_raw.iloc[:, 0].astype(float)
    rul_true.index = range(1, len(rul_true) + 1)
    rul_true.index.name = "unit_id"

    _validate(train_df, test_df, rul_true, subset)
    return train_df, test_df, rul_true


def _validate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_true: pd.Series,
    subset: str,
) -> None:
    """Run lightweight integrity checks and print a summary."""
    assert train_df.shape[1] == 26, "Expected 26 columns in train set."
    assert not train_df.isnull().values.any(), "NaN detected in train set."
    assert not test_df.isnull().values.any(), "NaN detected in test set."
    n_train = train_df["unit_id"].nunique()
    n_test  = test_df["unit_id"].nunique()
    assert len(rul_true) == n_test, (
        f"RUL vector length {len(rul_true)} ≠ n_test_engines {n_test}."
    )
    print(
        f"[data_loader] {subset} loaded — "
        f"train engines: {n_train}, test engines: {n_test}, "
        f"train rows: {len(train_df):,}"
    )
