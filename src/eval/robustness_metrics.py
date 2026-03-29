# src/eval/robustness_metrics.py

from __future__ import annotations

import pandas as pd


def compute_degradation(clean_metrics: dict, noisy_metrics: dict):
    result = {}

    for key in clean_metrics:
        if key in noisy_metrics:
            result[key + "_drop"] = clean_metrics[key] - noisy_metrics[key]

    return result


def build_robustness_table(clean_df: pd.DataFrame, noisy_df: pd.DataFrame):
    merged = clean_df.merge(noisy_df, on="method", suffixes=("_clean", "_noisy"))

    rows = []

    for _, row in merged.iterrows():
        entry = {"method": row["method"]}

        for col in clean_df.columns:
            if col == "method":
                continue

            clean_val = row[col + "_clean"]
            noisy_val = row[col + "_noisy"]

            entry[col + "_clean"] = clean_val
            entry[col + "_noisy"] = noisy_val
            entry[col + "_drop"] = clean_val - noisy_val

        rows.append(entry)

    return pd.DataFrame(rows)