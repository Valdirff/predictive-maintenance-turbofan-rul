"""
lstm_model.py
=============
LSTM sequence model for RUL estimation.

Architecture
------------
Input  : (batch, window_size, n_features)
LSTM   : 2 stacked layers with dropout
Dense  : fully-connected output → scalar RUL prediction

The LSTM consumes *fixed-length sliding windows* of sensor readings and
predicts the RUL at the last cycle of each window. Windows are constructed
with stride=1 during training; for test engines the last *window_size* cycles
are used for a single prediction per engine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    LSTM_PATIENCE,
    LSTM_WINDOW,
    MODELS_DIR,
)
from src.preprocessing import get_sensor_columns


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RULWindowDataset(Dataset):
    """Sliding-window dataset for LSTM training."""

    def __init__(
        self,
        sequences: np.ndarray,   # shape: (n_windows, window_size, n_features)
        targets: np.ndarray,     # shape: (n_windows,)
    ) -> None:
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    window_size: int = LSTM_WINDOW,
    feature_cols: list[str] | None = None,
    target_col: str | None = "RUL",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a preprocessed DataFrame into overlapping fixed-length windows.

    Each window covers `window_size` consecutive cycles of one engine.
    The target for that window is the RUL at the *last* cycle in the window.
    For engines shorter than `window_size`, zero-padding is applied at the start.

    Parameters
    ----------
    df           : preprocessed DataFrame (sorted by unit_id, cycle)
    window_size  : number of cycles per window
    feature_cols : columns to use as features (defaults to sensor columns)
    target_col   : target column (None for test-set inference)

    Returns
    -------
    (X, y) : numpy arrays of shape (n_windows, window_size, n_features)
             and (n_windows,) respectively. y is all zeros if target_col=None.
    """
    if feature_cols is None:
        feature_cols = get_sensor_columns(df)

    X_list, y_list = [], []

    for _, group in df.groupby("unit_id"):
        data = group[feature_cols].values.astype(np.float32)
        n = len(data)

        if target_col and target_col in df.columns:
            targets = group[target_col].values.astype(np.float32)
        else:
            targets = np.zeros(n, dtype=np.float32)

        # Zero-pad if the engine has fewer cycles than window_size
        if n < window_size:
            pad = np.zeros((window_size - n, data.shape[1]), dtype=np.float32)
            data = np.vstack([pad, data])
            targets = np.concatenate([np.zeros(window_size - n, dtype=np.float32), targets])
            n = window_size

        for i in range(n - window_size + 1):
            X_list.append(data[i: i + window_size])
            y_list.append(targets[i + window_size - 1])

    return np.array(X_list), np.array(y_list)


def build_test_sequences(
    df: pd.DataFrame,
    window_size: int = LSTM_WINDOW,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Build one window per test engine (last *window_size* cycles, with padding).
    Returns shape (n_engines, window_size, n_features).
    """
    if feature_cols is None:
        feature_cols = get_sensor_columns(df)

    X_list = []
    for _, group in df.groupby("unit_id"):
        data = group[feature_cols].values.astype(np.float32)
        n = len(data)
        if n >= window_size:
            window = data[-window_size:]
        else:
            pad = np.zeros((window_size - n, data.shape[1]), dtype=np.float32)
            window = np.vstack([pad, data])
        X_list.append(window)
    return np.array(X_list)


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    """
    Self-Attention for LSTM hidden states.
    Calculates a learnable weight for each timestep 't'.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq, hidden)
        # scores: (batch, seq, 1)
        scores = self.attn(lstm_output)
        weights = torch.softmax(scores, dim=1)
        # context: (batch, hidden)
        context = torch.sum(weights * lstm_output, dim=1)
        return context, weights


class LSTMRegressor(nn.Module):
    """
    Advanced LSTM + Attention model for RUL regression.

    Architecture:
    1. LSTM (Stacked Layers)
    2. Attention Layer (Context vector extraction)
    3. Dropout
    4. FC Layer (Output)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out, _ = self.lstm(x)             # out: (batch, seq, hidden)
        context, _ = self.attention(out)  # context: (batch, hidden)
        context = self.dropout(context)
        return self.fc(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMTrainer:
    """
    Wrapper around LSTMRegressor with a full training loop, early stopping,
    and history tracking.

    Parameters
    ----------
    input_size  : number of sensor features (automatically detected from data)
    device      : 'cpu', 'cuda', or 'mps' (auto-detected if None)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        lr: float = LSTM_LR,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self.model = LSTMRegressor(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        patience: int = LSTM_PATIENCE,
    ) -> "LSTMTrainer":
        """
        Train with mini-batch gradient descent + early stopping.

        Parameters
        ----------
        X_train, y_train : training windows and targets
        X_val, y_val     : validation windows and targets
        epochs, batch_size, patience : training controls

        Returns
        -------
        self
        """
        train_ds = RULWindowDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        val_X_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        val_y_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        best_val_loss = math.inf
        patience_counter = 0
        best_state = None

        print(f"[LSTMTrainer] Training on {self.device} | {len(X_train):,} windows | "
              f"patience={patience}")

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(y_batch)

            avg_train = epoch_loss / len(train_ds)

            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(val_X_t)
                avg_val = self.criterion(val_preds, val_y_t).item()

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{epochs} | "
                      f"train_loss={avg_train:.2f} | val_loss={avg_val:.2f}")

            # --- Early stopping ---
            if avg_val < best_val_loss - 1e-4:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val loss={best_val_loss:.2f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference; returns clipped RUL array."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()
        return np.clip(preds, 0, None)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        """Save model state dict and training history."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or MODELS_DIR / "lstm_rul.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "history": self.history,
                "model_config": {
                    "input_size": self.model.lstm.input_size,
                    "hidden_size": self.model.lstm.hidden_size,
                    "num_layers": self.model.lstm.num_layers,
                },
            },
            path,
        )
        print(f"[LSTMTrainer] Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "LSTMTrainer":
        """Load a saved LSTMTrainer from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        cfg = checkpoint["model_config"]
        trainer = cls(input_size=cfg["input_size"], device=device)
        trainer.model = LSTMRegressor(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.history = checkpoint.get("history", {})
        if device:
            trainer.device = torch.device(device)
            trainer.model = trainer.model.to(trainer.device)
        return trainer
