# main_uncertainty_compare.py

from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.eval.calibration_metrics import compute_calibration_metrics
from src.eval.ranking_metrics import compute_ranking_metrics


def evaluate_estimator(df, score_col):
    temp = df.copy()
    temp["confidence"] = temp[score_col]

    calib = compute_calibration_metrics(temp, "confidence")

    temp = temp.sort_values(["user_id", "confidence"], ascending=[True, False])
    temp["rank"] = temp.groupby("user_id").cumcount() + 1

    ranking = compute_ranking_metrics(temp)

    result = {}
    result.update(calib)
    result.update(ranking)

    return result


def main():
    df = pd.read_json("outputs/calibrated/test_calibrated.jsonl", lines=True)

    estimators = {
        "verbalized_raw": "confidence",
        "verbalized_calibrated": "calibrated_confidence",
        "consistency": "consistency_confidence",
        "fused": "fused_confidence",
    }

    results = []

    for name, col in estimators.items():
        if col not in df.columns:
            continue

        res = evaluate_estimator(df, col)
        res["estimator"] = name
        results.append(res)

    result_df = pd.DataFrame(results)
    result_df.to_csv("outputs/tables/estimator_comparison.csv", index=False)

    print("Saved estimator_comparison.csv")


if __name__ == "__main__":
    main()