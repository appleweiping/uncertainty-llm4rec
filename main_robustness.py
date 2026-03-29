# main_robustness.py

from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.eval.robustness_metrics import build_robustness_table
from src.analysis.noise_analysis import summarize_noise_effect


def main():

    clean_path = "outputs/tables/rerank_results.csv"
    noisy_path = "outputs/tables/rerank_results_noisy.csv"

    clean_df = pd.read_csv(clean_path)
    noisy_df = pd.read_csv(noisy_path)

    robustness_df = build_robustness_table(clean_df, noisy_df)

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    robustness_df.to_csv("outputs/tables/robustness_table.csv", index=False)

    summary = summarize_noise_effect(robustness_df)
    pd.DataFrame([summary]).to_csv("outputs/tables/robustness_summary.csv", index=False)

    print("Saved robustness_table.csv and robustness_summary.csv")


if __name__ == "__main__":
    main()