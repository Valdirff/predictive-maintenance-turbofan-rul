"""
visualization.py
================
Polished plotting utilities for the turbofan RUL prognostics project.

All figures are saved as PNG to results/figures/. Functions return the
matplotlib Figure object so they can also be displayed inline in notebooks.

Figure catalogue
----------------
plot_degradation_trajectories : sensor trends for selected engines
plot_health_indicator         : HI trajectories coloured by cycle fraction
plot_real_vs_predicted        : scatter plot with error bands per model
plot_residuals                : residual distribution histogram
plot_model_comparison         : grouped bar chart (RMSE, NASA score)
plot_learning_curves          : training/validation loss over epochs (LSTM)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import COLOR_PALETTE, FIGURE_DPI, FIGURE_STYLE, FIGURES_DIR


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

plt.style.use(FIGURE_STYLE)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_FONT_TITLE = {"fontsize": 13, "fontweight": "bold"}
_FONT_LABEL = {"fontsize": 11}


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    # Ensure parent directory (e.g. results/figures/xgboost/) exists
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"[visualization] Figure saved → {path}")
    return path


def _engine_sample(df: pd.DataFrame, n: int = 6, random_state: int = 0) -> list[int]:
    """Return up to *n* representative engine ids."""
    ids = df["unit_id"].unique().tolist()
    rng = np.random.default_rng(random_state)
    return list(rng.choice(ids, size=min(n, len(ids)), replace=False))


# ---------------------------------------------------------------------------
# Degradation trajectories
# ---------------------------------------------------------------------------

def plot_degradation_trajectories(
    df: pd.DataFrame,
    sensors: list[str],
    n_engines: int = 6,
    filename: str = "degradation_trajectories.png",
) -> plt.Figure:
    """
    Plot raw sensor trends for a sample of engines to visualise degradation.
    """
    engine_ids = _engine_sample(df, n_engines)
    n_sensors = len(sensors)
    ncols = 3
    nrows = (n_sensors + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes_flat = axes.ravel() if nrows > 1 else [axes] if ncols == 1 else axes.ravel()

    palette = sns.color_palette("tab10", n_colors=n_engines)

    for ax_idx, sensor in enumerate(sensors):
        ax = axes_flat[ax_idx]
        for eng_idx, eid in enumerate(engine_ids):
            sub = df[df["unit_id"] == eid]
            ax.plot(
                sub["cycle"], sub[sensor],
                color=palette[eng_idx], alpha=0.7, linewidth=1,
                label=f"Engine {eid}" if ax_idx == 0 else None,
            )
        ax.set_title(sensor, **_FONT_TITLE)
        ax.set_xlabel("Cycle", **_FONT_LABEL)
        ax.set_ylabel("Value (normalised)", **_FONT_LABEL)

    # Hide unused axes
    for i in range(n_sensors, len(axes_flat)):
        axes_flat[i].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=3, fontsize=9)
    fig.suptitle("Sensor Degradation Trajectories — FD001", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Health Indicator
# ---------------------------------------------------------------------------

def plot_health_indicator(
    df: pd.DataFrame,
    n_engines: int = 8,
    filename: str = "health_indicator.png",
) -> plt.Figure:
    """
    Plot HI trajectories for a sample of engines, coloured by degradation stage.
    """
    assert "HI" in df.columns, "DataFrame must contain an 'HI' column."
    engine_ids = _engine_sample(df, n_engines)

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = sns.color_palette("coolwarm_r", n_colors=n_engines)

    for idx, eid in enumerate(engine_ids):
        sub = df[df["unit_id"] == eid]
        max_c = sub["cycle"].max()
        ax.plot(
            sub["cycle"] / max_c,    # normalised lifecycle
            sub["HI"],
            color=palette[idx], alpha=0.8, linewidth=1.5,
            label=f"Engine {eid}",
        )

    ax.axhline(0.1, color="crimson", linestyle="--", linewidth=1.2, label="Failure threshold")
    ax.set_xlabel("Normalised Lifecycle (0=BoL, 1=EoL)", **_FONT_LABEL)
    ax.set_ylabel("Health Indicator", **_FONT_LABEL)
    ax.set_title("Health Indicator Trajectories — FD001", **_FONT_TITLE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Real vs. Predicted
# ---------------------------------------------------------------------------

def plot_real_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of actual vs. predicted RUL with identity line.
    """
    color = COLOR_PALETTE.get(model_name.lower().split()[0], "#3B82F6")
    filename = filename or f"real_vs_pred_{model_name.lower().replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, color=color, alpha=0.6, edgecolors="white", linewidths=0.4, s=40)

    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Perfect prediction")
    # ±20% error bands
    ax.fill_between([0, lim], [0, lim * 0.8], [0, lim * 1.2],
                    alpha=0.08, color="gray", label="±20 % band")

    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.set_xlabel("Actual RUL (cycles)", **_FONT_LABEL)
    ax.set_ylabel("Predicted RUL (cycles)", **_FONT_LABEL)
    ax.set_title(f"{model_name} — Actual vs. Predicted RUL\nRMSE = {rmse_val:.2f}", **_FONT_TITLE)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    filename: str | None = None,
) -> plt.Figure:
    """Residual distribution histogram and KDE."""
    residuals = y_pred - y_true
    color = COLOR_PALETTE.get(model_name.lower().split()[0], "#3B82F6")
    filename = filename or f"residuals_{model_name.lower().replace(' ', '_')}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=30, kde=True, color=color, ax=ax, alpha=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (Predicted − Actual) [cycles]", **_FONT_LABEL)
    ax.set_ylabel("Count", **_FONT_LABEL)
    ax.set_title(f"{model_name} — Residual Distribution", **_FONT_TITLE)
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    filename: str = "model_comparison.png",
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE and NASA score across models.

    Parameters
    ----------
    metrics_df : DataFrame with columns ['model', 'rmse', 'nasa_score']
    """
    models = metrics_df["model"].tolist()
    rmse_vals = metrics_df["rmse"].tolist()
    score_vals = metrics_df["nasa_score"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [
        COLOR_PALETTE.get(m.lower().split()[0], "#6B7280") for m in models
    ]

    bars1 = ax1.bar(x, rmse_vals, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("RMSE (cycles)", **_FONT_LABEL)
    ax1.set_title("RMSE Comparison", **_FONT_TITLE)
    ax1.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    ax1.set_ylim(0, max(rmse_vals) * 1.25)

    bars2 = ax2.bar(x, score_vals, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel("NASA Asymmetric Score (lower = better)", **_FONT_LABEL)
    ax2.set_title("NASA Score Comparison", **_FONT_TITLE)
    ax2.bar_label(bars2, fmt="%.0f", padding=3, fontsize=9)
    ax2.set_ylim(0, max(score_vals) * 1.25)

    fig.suptitle("Model Comparison — FD001 Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return fig


# ---------------------------------------------------------------------------
# Learning curves (LSTM)
# ---------------------------------------------------------------------------

def plot_learning_curves(
    history: dict[str, list[float]],
    filename: str = "lstm_learning_curves.png",
) -> plt.Figure:
    """
    Plot LSTM training and validation loss over epochs.

    Parameters
    ----------
    history : dict with keys 'train_loss' and 'val_loss'
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color=COLOR_PALETTE["lstm"], linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss",
            color=COLOR_PALETTE["lstm"], linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", **_FONT_LABEL)
    ax.set_ylabel("MSE Loss", **_FONT_LABEL)
    ax.set_title("LSTM — Training & Validation Loss", **_FONT_TITLE)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, filename)
    return fig
