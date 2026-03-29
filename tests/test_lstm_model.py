"""Unit tests for LSTM sequence building and model architecture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm_model import LSTMRegressor, build_sequences, build_test_sequences


def test_build_sequences_padding():
    """Verify that training sequence generation handles short engines correctly."""
    # Synthetic dataframe with one engine containing only 10 cycles
    df_short = pd.DataFrame({
        "unit_id": [1] * 10,
        "cycle": list(range(1, 11)),
        "sensor_1": np.ones(10),
        "RUL": np.arange(10, 0, -1)
    })
    
    window_size = 30
    
    # Generate sliding window sequences
    X, y = build_sequences(
        df_short, 
        window_size=window_size, 
        feature_cols=["sensor_1"], 
        target_col="RUL"
    )
    
    # Should only create 1 window containing all 10 cycles, left-padded with 20 zeros
    # Wait: actually the sequence logic loops `for i in range(n - window_size + 1)`
    # Since n < window_size, `n` gets padded up to `window_size`. So we should get 1 window.
    assert X.shape == (1, window_size, 1)
    assert y.shape == (1,)
    
    # Check padding logic: first 20 elements should be 0.0, last 10 elements should be 1.0
    assert np.allclose(X[0, :20, 0], 0.0)
    assert np.allclose(X[0, 20:, 0], 1.0)
    
    # Target RUL drops exactly on the last real prediction point
    assert y[0] == 1.0  


def test_build_test_sequences_padding():
    """Verify inference sequence builder handles short engines correctly."""
    df_test = pd.DataFrame({
        "unit_id": [99] * 5,  # only 5 cycles
        "cycle": list(range(50, 55)),
        "sensor_1": np.full(5, 7.5)
    })
    
    window_size = 15
    X_test = build_test_sequences(
        df_test, 
        window_size=window_size, 
        feature_cols=["sensor_1"]
    )
    
    # Test builder strictly returns exactly 1 window per engine
    assert X_test.shape == (1, window_size, 1)
    
    # Padding validation
    assert np.allclose(X_test[0, :10, 0], 0.0)
    assert np.allclose(X_test[0, 10:, 0], 7.5)


def test_lstm_architecture_forward():
    """Verify the Attention-LSTM accepts the sequences and returns a scalar RUL."""
    batch_size = 4
    seq_len = 30
    features = 10
    
    model = LSTMRegressor(
        input_size=features,
        hidden_size=32,
        num_layers=1,
        dropout=0.0
    )
    
    # Mock data tensor matching the sequences
    X_mock = torch.rand((batch_size, seq_len, features))
    
    # Forward pass must return dimension (batch_size, 1) securely
    model.eval()
    with torch.no_grad():
        preds = model(X_mock)
        
    assert preds.shape == (batch_size,)
