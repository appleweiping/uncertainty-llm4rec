from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.aggregate_model_results import aggregate_experiment
from src.utils.io import ensure_dir

RUN_PATTERN = re.compile(r"^(?P<setting>.+)_rep(?P<run_id>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--exp_names", type=str, nargs="*", default=None)
    return parser.parse_args()


def discover_rep_experiments(output_root: Path) -> list[str]:
    exp_names: list[str] = []
    for child in sorted(output_root.iterdir()):
        if child.is_dir() and RUN_PATTERN.match(child.name):
            exp_names.append(child.name)
    return exp_names


def parse_run_metadata(exp_name: str) -> dict[str, Any]:
    match = RUN_PATTERN.match(exp_name)
    if not match:
        raise ValueError(f"Experiment name does not follow *_repN convention: {exp_name}")
    return {
        "setting": match.group("setting"),
        "run_id": int(match.group("run_id")),
    }


def build_check_table(output_root: Path, exp_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for exp_name in exp_names:
        row = aggregate_experiment(output_root / exp_name)
        row.update(parse_run_metadata(exp_name))
        rows.append(row)

    check_df = pd.DataFrame(rows)
    if check_df.empty:
        return check_df

    ordered_columns = [
        "setting",
        "run_id",
        "exp_name",
        "domain",
        "model",
        "lambda",
    ]
    remaining_columns = [col for col in check_df.columns if col not in ordered_columns]
    return check_df[ordered_columns + remaining_columns].sort_values(
        ["setting", "run_id"]
    ).reset_index(drop=True)


def build_delta_table(check_df: pd.DataFrame) -> pd.DataFrame:
    if check_df.empty:
        return pd.DataFrame()

    id_columns = {"setting", "run_id", "exp_name", "domain", "model"}
    numeric_columns = [
        col for col in check_df.columns
        if col not in id_columns and pd.api.types.is_numeric_dtype(check_df[col])
    ]

    rows: list[dict[str, Any]] = []
    for setting, group in check_df.groupby("setting", sort=True):
        group = group.sort_values("run_id").reset_index(drop=True)
        if len(group) < 2:
            continue

        first = group.iloc[0]
        second = group.iloc[1]

        row: dict[str, Any] = {
            "setting": setting,
            "domain": first.get("domain"),
            "model": first.get("model"),
            "run_a": first.get("exp_name"),
            "run_b": second.get("exp_name"),
        }

        abs_deltas: list[float] = []
        for column in numeric_columns:
            first_value = first.get(column)
            second_value = second.get(column)
            if pd.isna(first_value) or pd.isna(second_value):
                row[f"{column}_abs_diff"] = pd.NA
                continue

            abs_diff = abs(float(first_value) - float(second_value))
            row[f"{column}_abs_diff"] = abs_diff
            abs_deltas.append(abs_diff)

        row["max_abs_diff"] = max(abs_deltas) if abs_deltas else pd.NA
        row["mean_abs_diff"] = sum(abs_deltas) / len(abs_deltas) if abs_deltas else pd.NA
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["setting"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    exp_names = args.exp_names or discover_rep_experiments(output_root)
    if not exp_names:
        raise ValueError("No reproducibility experiment folders found.")

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    check_df = build_check_table(output_root, exp_names)
    delta_df = build_delta_table(check_df)

    check_path = summary_dir / "reproducibility_check.csv"
    delta_path = summary_dir / "reproducibility_delta.csv"

    check_df.to_csv(check_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    print(f"Saved reproducibility check rows to: {check_path}")
    print(f"Saved reproducibility delta rows to: {delta_path}")


if __name__ == "__main__":
    main()
