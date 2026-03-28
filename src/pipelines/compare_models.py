"""
compare_models.py
=================
Load all saved metric JSON files from results/metrics/ and produce:
  - A markdown comparison table  →  results/tables/comparison.md
  - A grouped bar chart           →  results/figures/model_comparison.png

This script is intended to be run *after* all three model pipelines have
been executed and their metrics saved.

Usage
-----
    python -m src.pipelines.compare_models
    # or
    python src/pipelines/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from src.evaluation import load_all_metrics
from src.config import TABLES_DIR
from src.visualization import plot_model_comparison


# Qualitative labels (filled from saved metrics or hardcoded fallback)
QUALITATIVE: dict[str, dict[str, str]] = {
    "Exponential": {
        "interpretability": "⭐⭐⭐ High",
        "computational_cost": "🟢 Low",
        "industrial_deployability": "✅ High",
    },
    "XGBoost": {
        "interpretability": "⭐⭐ Medium",
        "computational_cost": "🟡 Low–Medium",
        "industrial_deployability": "✅✅ Very High",
    },
    "LSTM": {
        "interpretability": "⭐ Low",
        "computational_cost": "🔴 High",
        "industrial_deployability": "⚠️ Moderate",
    },
}


def build_markdown_table(df: pd.DataFrame) -> str:
    """Convert a metrics DataFrame into a clean markdown table."""
    rows = ["| Model | RMSE | NASA Score | Train Time (s) | Interpretability | Cost | Deployability |",
            "|-------|------|------------|----------------|------------------|------|---------------|"]

    for _, row in df.iterrows():
        model = row.get("model", "N/A")
        qual = QUALITATIVE.get(model, {})
        rows.append(
            f"| {model} "
            f"| {row.get('rmse', '-'):.2f} "
            f"| {row.get('nasa_score', '-'):.0f} "
            f"| {row.get('train_time_s', '-'):.1f} "
            f"| {qual.get('interpretability', '-')} "
            f"| {qual.get('computational_cost', '-')} "
            f"| {qual.get('industrial_deployability', '-')} |"
        )
    return "\n".join(rows)


def run() -> None:
    print(f"\n{'='*60}")
    print("  Model Comparison")
    print(f"{'='*60}\n")

    metrics_df = load_all_metrics()

    if metrics_df.empty:
        print("[compare_models] No metric files found. Run the training pipelines first.")
        return

    # Sort by RMSE ascending (best first)
    metrics_df = metrics_df.sort_values("rmse").reset_index(drop=True)

    # Markdown table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_md = build_markdown_table(metrics_df)
    table_path = TABLES_DIR / "comparison.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"[compare_models] Comparison table saved → {table_path}")
    print("\n" + table_md)

    # Bar charts
    plot_model_comparison(metrics_df[["model", "rmse", "nasa_score"]])

    # Best model summary
    best = metrics_df.iloc[0]
    print(
        f"\n📊 Best model by RMSE: {best['model']} "
        f"(RMSE={best['rmse']:.2f}, NASA Score={best['nasa_score']:.0f})"
    )


if __name__ == "__main__":
    run()
