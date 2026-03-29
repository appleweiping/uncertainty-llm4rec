# src/analysis/calibration_plotting.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_before_after_reliability(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    output_path: str | Path
) -> None:
    _ensure_parent_dir(output_path)

    before_plot = before_df.dropna(subset=["avg_confidence", "accuracy"]).copy()
    after_plot = after_df.dropna(subset=["avg_confidence", "accuracy"]).copy()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    plt.plot(before_plot["avg_confidence"], before_plot["accuracy"], marker="o", label="Before")
    plt.plot(after_plot["avg_confidence"], after_plot["accuracy"], marker="o", label="After")
    plt.xlabel("Average Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram: Before vs After Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()