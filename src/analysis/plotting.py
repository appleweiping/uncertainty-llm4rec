# src/analysis/plotting.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.confidence_correctness import prepare_prediction_dataframe


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_confidence_histogram(
    df: pd.DataFrame,
    output_path: str | Path
) -> None:
    df = prepare_prediction_dataframe(df)
    _ensure_parent_dir(output_path)

    correct = df[df["is_correct"] == 1]["confidence"]
    wrong = df[df["is_correct"] == 0]["confidence"]

    plt.figure(figsize=(8, 5))
    plt.hist(correct, bins=10, alpha=0.6, label="Correct")
    plt.hist(wrong, bins=10, alpha=0.6, label="Wrong")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Histogram: Correct vs Wrong")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reliability_diagram(
    reliability_df: pd.DataFrame,
    output_path: str | Path
) -> None:
    _ensure_parent_dir(output_path)

    plot_df = reliability_df.dropna(subset=["avg_confidence", "accuracy"]).copy()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])
    plt.plot(plot_df["avg_confidence"], plot_df["accuracy"], marker="o")
    plt.xlabel("Average Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_popularity_avg_confidence(
    pop_df: pd.DataFrame,
    output_path: str | Path
) -> None:
    _ensure_parent_dir(output_path)

    plt.figure(figsize=(7, 5))
    plt.bar(pop_df["target_popularity_group"], pop_df["avg_confidence"])
    plt.xlabel("Popularity Group")
    plt.ylabel("Average Confidence")
    plt.title("Average Confidence by Popularity Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_popularity_confidence_boxplot(
    df: pd.DataFrame,
    output_path: str | Path
) -> None:
    df = prepare_prediction_dataframe(df)
    _ensure_parent_dir(output_path)

    groups = []
    labels = []
    for group_name in ["head", "mid", "tail", "unknown"]:
        subset = df[df["target_popularity_group"] == group_name]["confidence"]
        if len(subset) > 0:
            groups.append(subset)
            labels.append(group_name)

    if not groups:
        return

    plt.figure(figsize=(7, 5))
    plt.boxplot(groups, labels=labels)
    plt.xlabel("Popularity Group")
    plt.ylabel("Confidence")
    plt.title("Confidence Distribution by Popularity Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_high_confidence_exposure_shift(
    exposure_df: pd.DataFrame,
    output_path: str | Path
) -> None:
    _ensure_parent_dir(output_path)

    plt.figure(figsize=(7, 5))
    plt.bar(exposure_df["target_popularity_group"], exposure_df["exposure_shift"])
    plt.xlabel("Popularity Group")
    plt.ylabel("High-Confidence Exposure Shift")
    plt.title("Exposure Shift in High-Confidence Samples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()