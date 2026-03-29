"""
generate_stochastic_plots.py
============================
Gera todos os plots diagnósticos e analíticos do modelo estocástico
de degradação de motores turbofan (NASA C-MAPSS).

Plots produzidos
----------------
1. hi_fleet_trajectories.png      — Trajetórias de HI de toda a frota de treino
2. hi_degradation_detail.png      — HI com curva WLS ajustada + threshold por motor
3. rul_actual_vs_predicted.png    — Scatter real vs. predito com banda de erro
4. rul_per_engine_with_ci.png     — RUL previsto ± IC 90% por motor de teste
5. residuals_analysis.png         — Distribuição de resíduos + QQ-plot
6. wls_log_linear_fit.png         — Espaço log-log: ajuste linear dos parâmetros θ

Usage
-----
    python -m src.pipelines.generate_stochastic_plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import (
    FIGURES_DIR,
    FIGURE_DPI,
    HI_LOGISTIC_N_SAMPLES,
    SUBSET,
)
from src.data_loader import load_cmapss
from src.health_indicator import LogisticHIBuilder, select_monotonic_sensors
from src.models.stochastic_model import (
    StochasticDegradationRUL,
    _geometric_weights,
    _trajectory,
    _wls_fit,
)
from src.preprocessing import (
    add_rul_target,
    drop_constant_sensors,
    fit_scaler,
    get_sensor_columns,
    transform_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 120,
})

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Palette
C_VIOLET   = "#7C3AED"
C_AMBER    = "#F59E0B"
C_SLATE    = "#475569"
C_RED      = "#EF4444"
C_GREEN    = "#10B981"
C_BLUE     = "#3B82F6"
C_PINK     = "#EC4899"
C_ORANGE   = "#F97316"

CMAP_FLEET = "plasma"


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  ✓  {name}")
    return path


def _engine_sample(
    df: pd.DataFrame, n: int = 10, random_state: int = 7
) -> list[int]:
    ids = sorted(df["unit_id"].unique().tolist())
    rng = np.random.default_rng(random_state)
    return sorted(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Fleet HI trajectories (Maulana et al., 2023 Fig 3 style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hi_fleet_trajectories(
    train_df: pd.DataFrame,
    failure_threshold: float,
    n_engines: int = 18,
) -> plt.Figure:
    """
    All HI trajectories of the training fleet plotted over cycle,
    coloured by their total life length (short = red, long = green).
    Shows how the logistic HI naturally separates healthy vs degraded state.
    """
    engine_ids = _engine_sample(train_df, n_engines, random_state=5)
    max_lives = {
        eid: train_df[train_df["unit_id"] == eid]["cycle"].max()
        for eid in engine_ids
    }
    life_vals = np.array([max_lives[e] for e in engine_ids], dtype=float)
    norm = plt.Normalize(life_vals.min(), life_vals.max())
    cmap = plt.get_cmap("plasma_r")

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for eid in engine_ids:
        sub = train_df[train_df["unit_id"] == eid].sort_values("cycle")
        color = cmap(norm(max_lives[eid]))
        ax.plot(sub["cycle"], sub["HI"], color=color, alpha=0.72, linewidth=1.4)

    # Failure threshold
    ax.axhline(
        failure_threshold, color=C_RED, linestyle="--", linewidth=1.8,
        label=f"Failure threshold  (HI = {failure_threshold:.3f})",
        zorder=5,
    )

    sm = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Engine total life (cycles)", fontsize=10)

    ax.set_xlabel("Operational Cycle", fontsize=12)
    ax.set_ylabel("Health Indicator (HI)", fontsize=12)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(
        "Fleet Health Indicator Trajectories — FD001\n"
        r"$\mathit{HI}(t)$ constructed via Logistic Regression (Maulana et al., 2023 style)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    _save(fig, "hi_fleet_trajectories.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — WLS degradation fit overlay (Wen et al. 2021 Fig 5 style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hi_degradation_detail(
    train_df: pd.DataFrame,
    model: StochasticDegradationRUL,
    n_engines: int = 6,
) -> plt.Figure:
    """
    Shows the observed HI plus the WLS power-function fit for individual
    engines — mirrors the paper's 'fitted degradation curves' figure.
    """
    engine_ids = _engine_sample(train_df, n_engines, random_state=99)
    ncols = 3
    nrows = (n_engines + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharey=True)
    axes_flat = axes.ravel()

    for ax_idx, eid in enumerate(engine_ids):
        ax = axes_flat[ax_idx]
        sub = train_df[train_df["unit_id"] == eid].sort_values("cycle")
        cycles = sub["cycle"].values.astype(float)
        hi = sub["HI"].values.astype(float)

        # Observed HI
        ax.plot(cycles, hi, color=C_SLATE, linewidth=1.3, alpha=0.8, label="Observed HI")

        # WLS fitted curve
        params = model.engine_params_.get(int(eid))
        if params and params.get("fit_ok"):
            t_fit = np.linspace(cycles[0], cycles[-1], 200)
            hi_fit = _trajectory(
                t_fit, params["theta0"], params["theta1"], params["phi"]
            )
            ax.plot(t_fit, hi_fit, color=C_VIOLET, linewidth=2.0,
                    linestyle="--", label="WLS fit  $\\phi + e^{\\theta^{(0)}} t^{\\theta^{(1)}}$")

        # Failure threshold line
        ax.axhline(model.failure_threshold_, color=C_RED, linewidth=1.2,
                   linestyle=":", alpha=0.8, label="Failure threshold")

        ax.set_title(f"Engine {eid}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=10)
        ax.set_ylabel("HI", fontsize=10)
        ax.set_ylim(-0.05, 1.1)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.03))

    # Hide empty axes
    for i in range(n_engines, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "WLS Power-Function Degradation Fit per Engine\n"
        r"Log-linear model:  $\ln(\mathrm{HI}(t)-\phi) = \theta^{(0)} + \theta^{(1)}\ln(t)$",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "hi_degradation_detail.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Actual vs Predicted RUL scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_rul_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted RUL with identity line, ±20% band,
    and ±10% band. Points are colour-coded by absolute error magnitude.
    """
    abs_err = np.abs(y_true - y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(abs_err))

    fig, ax = plt.subplots(figsize=(8, 7))

    # Background error bands
    lim = max(y_true.max(), y_pred.max()) * 1.08
    ax.fill_between([0, lim], [0, lim * 0.8],  [0, lim * 1.2],
                    alpha=0.07, color="gray", label="±20% band")
    ax.fill_between([0, lim], [0, lim * 0.9],  [0, lim * 1.1],
                    alpha=0.12, color=C_VIOLET, label="±10% band")

    # Identity line
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.5, label="Perfect prediction", zorder=3)

    # Scatter coloured by error
    sc = ax.scatter(
        y_true, y_pred,
        c=abs_err, cmap="plasma_r",
        vmin=0, vmax=min(abs_err.max(), 120),
        s=55, alpha=0.85, edgecolors="white", linewidths=0.4, zorder=4,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Absolute Error (cycles)", fontsize=10)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual RUL (cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=12)
    ax.set_title(
        f"Stochastic Model — Actual vs. Predicted RUL\n"
        f"RMSE = {rmse:.1f} cycles   |   MAE = {mae:.1f} cycles",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "rul_actual_vs_predicted.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — RUL per test engine with 90% CI
# ─────────────────────────────────────────────────────────────────────────────

def plot_rul_per_engine_with_ci(
    y_true: np.ndarray,
    rul_df: pd.DataFrame,
    n_show: int = 30,
) -> plt.Figure:
    """
    Bar chart showing predicted RUL with 90% CI shading alongside true RUL
    for the first n_show test engines — inspired by Maulana et al. 2023 prognosis plot.
    """
    df = rul_df.copy().head(n_show)
    x = np.arange(len(df))
    y_true_sub = y_true[:n_show]

    fig, ax = plt.subplots(figsize=(15, 5.5))

    # CI ribbon
    ax.fill_between(
        x, df["ci_lower"], df["ci_upper"],
        alpha=0.25, color=C_VIOLET, label="90% Confidence Interval",
    )

    # Predicted RUL line
    ax.plot(x, df["rul_mean"], color=C_VIOLET, linewidth=2.2,
            marker="o", markersize=4.5, label="Predicted RUL (mean)", zorder=4)

    # Actual RUL
    ax.plot(x, y_true_sub, color=C_AMBER, linewidth=2.0,
            marker="s", markersize=4.5, linestyle="--", label="Actual RUL", zorder=5)

    ax.set_xlabel("Test Engine Index", fontsize=12)
    ax.set_ylabel("RUL (cycles)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["unit_id"].astype(int).tolist(), fontsize=7, rotation=45)
    ax.set_title(
        f"Predicted vs. Actual RUL — First {n_show} Test Engines\n"
        "Shaded region: 90% CI via Monte Carlo Bootstrap",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    _save(fig, "rul_per_engine_with_ci.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Residual analysis (histogram + QQ-plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Two-panel residual analysis:
      Left  — histogram + KDE of residuals (ŷ − y)
      Right — Normal Q-Q plot to assess residual distribution
    """
    residuals = y_pred - y_true
    mean_r = residuals.mean()
    std_r  = residuals.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ─── Left: Histogram + KDE curve ───
    n_obs = len(residuals)
    hist_vals, bin_edges = np.histogram(residuals, bins=18)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax1.bar(bin_centers, hist_vals, width=bin_width * 0.9,
            color=C_VIOLET, alpha=0.75, label="Histogram")
    # KDE overlaid on count axis (scaled by n_obs * bin_width)
    kde_x = np.linspace(residuals.min() - 10, residuals.max() + 10, 300)
    kde_y = stats.gaussian_kde(residuals)(kde_x) * n_obs * bin_width
    ax1.plot(kde_x, kde_y, color=C_VIOLET, linewidth=2.5, label="KDE")
    ax1.axvline(0,      color="black", linestyle="--", linewidth=1.5, label="Zero error")
    ax1.axvline(mean_r, color=C_AMBER, linestyle="-",  linewidth=1.5,
                label=f"Mean = {mean_r:.1f}")
    ax1.set_xlabel("Residual  (Predicted - Actual)  [cycles]", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title(
        f"Residual Distribution\nmean = {mean_r:.1f}  |  std = {std_r:.1f} cycles",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=9)

    # ─── Right: Q-Q plot ───
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, color=C_VIOLET, alpha=0.7, s=30, label="Sample quantiles")
    x_line = np.array([osm.min(), osm.max()])
    ax2.plot(x_line, slope * x_line + intercept, color=C_AMBER,
             linewidth=2, label=f"Normal line  (R2={r**2:.3f})")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Stochastic Degradation Model — Residual Analysis",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "residuals_analysis.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — WLS log-log parameter space (Wen et al. Fig 4 style)
# ─────────────────────────────────────────────────────────────────────────────




def plot_wls_loglog_fit(
    train_df,
    model,
    n_engines=6,
):
    engine_ids = _engine_sample(train_df, n_engines, random_state=42)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.ravel()
    phi = model.phi_

    for ax_idx, eid in enumerate(engine_ids):
        ax = axes_flat[ax_idx]
        sub    = train_df[train_df['unit_id'] == eid].sort_values('cycle')
        cycles = sub['cycle'].values.astype(float)
        hi     = sub['HI'].values.astype(float)

        hi_s  = hi - phi
        valid = hi_s > 1e-6
        if valid.sum() < 3:
            ax.set_visible(False)
            continue

        x_log = np.log(cycles[valid])
        y_log = np.log(hi_s[valid])
        w     = _geometric_weights(valid.sum())
        t0, t1 = _wls_fit(x_log, y_log, w)

        ax.scatter(cycles[valid], hi_s[valid], c=w, cmap='plasma',
                   s=20, alpha=0.75, edgecolors='none',
                   label='Observed (colour=WLS weight)')
        c_fit  = np.linspace(cycles[valid][0], cycles[valid][-1], 300)
        hi_fit = np.clip(np.exp(t0) * c_fit ** t1, 1e-6, 2.0)
        ax.plot(c_fit, hi_fit, color=C_VIOLET, linewidth=2.0,
                label=f'WLS  t1={t1:.2f}')
        ax.axhline(model.failure_threshold_, color=C_RED, linestyle=':',
                   linewidth=1.2, alpha=0.7, label='Failure threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Cycle  [log scale]', fontsize=9)
        ax.set_ylabel('HI  [log scale]', fontsize=9)
        ax.set_title(f'Engine {eid}', fontsize=11, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=8)

    for i in range(n_engines, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        'WLS Power-Function Fit in Log-Log Space per Engine\n'
        r'Model: $\mathrm{HI}(t) = e^{\theta^{(0)}} \cdot t^{\theta^{(1)}}$'
        + f'   phi={phi:.4f}',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    _save(fig, 'wls_loglog_fit.png')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(subset: str = SUBSET) -> None:
    print(f"\n{'='*60}")
    print(f"  Generating Stochastic Model Diagnostic Plots — {subset}")
    print(f"{'='*60}\n")

    # ── Load + preprocess ─────────────────────────────────────────────────
    train_df, test_df, rul_true = load_cmapss(subset)
    train_df = drop_constant_sensors(train_df)
    test_df  = drop_constant_sensors(test_df)
    train_df = add_rul_target(train_df)

    sensor_cols = get_sensor_columns(train_df)
    scaler = fit_scaler(train_df, sensor_cols)
    train_df = transform_features(train_df, scaler, sensor_cols)
    test_df  = transform_features(test_df,  scaler, sensor_cols)

    # ── HI via logistic ────────────────────────────────────────────────────
    top_sensors = select_monotonic_sensors(train_df, top_k=9)
    hi_builder  = LogisticHIBuilder(sensors=top_sensors, n_samples=HI_LOGISTIC_N_SAMPLES)
    train_df["HI"] = hi_builder.fit_transform(train_df)
    test_df["HI"]  = hi_builder.transform(test_df)

    # ── Fit stochastic model ───────────────────────────────────────────────
    model = StochasticDegradationRUL()
    model.fit(train_df)

    # ── Predict (point + CI) ───────────────────────────────────────────────
    y_pred    = model.predict_test(test_df)
    rul_df    = model.predict_with_uncertainty(test_df)
    y_true    = rul_true.values

    # ── Generate all plots ─────────────────────────────────────────────────
    print("\nSaving plots to:", FIGURES_DIR, "\n")

    plot_hi_fleet_trajectories(train_df, model.failure_threshold_, n_engines=20)
    plot_hi_degradation_detail(train_df, model, n_engines=6)
    plot_rul_actual_vs_predicted(y_true, y_pred)
    plot_rul_per_engine_with_ci(y_true, rul_df, n_show=30)
    plot_residuals_analysis(y_true, y_pred)
    plot_wls_loglog_fit(train_df, model, n_engines=6)

    print(f"\n{'='*60}")
    print(f"  All 6 plots generated successfully!")
    print(f"  Location: {FIGURES_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
