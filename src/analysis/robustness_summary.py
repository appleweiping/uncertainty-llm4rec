from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--compare_names", type=str, nargs="*", default=None)
    return parser.parse_args()


def infer_domain_model(exp_name: str) -> tuple[str, str]:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    known_models = ["deepseek", "qwen", "kimi", "doubao", "glm", "gpt", "openai", "local"]
    model = "unknown"
    for token in reversed(tokens):
        if token in known_models:
            model = token
            break

    ignore_tokens = set(known_models) | {"small", "mini", "large", "clean", "noisy"}
    domain_tokens = [token for token in tokens if token not in ignore_tokens]
    domain = domain_tokens[0] if domain_tokens else exp_name.lower()
    if domain.startswith("movie"):
        domain = "movies"
    elif domain.startswith("book"):
        domain = "books"
    elif domain.startswith("electronic"):
        domain = "electronics"
    return domain, model


def discover_compare_dirs(robustness_root: Path) -> list[Path]:
    compare_dirs: list[Path] = []
    if not robustness_root.exists():
        return compare_dirs
    for child in sorted(robustness_root.iterdir()):
        if (
            child.is_dir()
            and (child / "tables" / "robustness_summary.csv").exists()
            and (child / "tables" / "robustness_diagnostic_table.csv").exists()
            and (child / "tables" / "robustness_calibration_table.csv").exists()
            and (child / "tables" / "robustness_confidence_table.csv").exists()
        ):
            compare_dirs.append(child)
    return compare_dirs


def load_single_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def aggregate_compare_dir(compare_dir: Path) -> dict[str, Any]:
    compare_name = compare_dir.name
    if "_vs_" not in compare_name:
        clean_exp = compare_name
        noisy_exp = compare_name
    else:
        clean_exp, noisy_exp = compare_name.split("_vs_", 1)

    domain, model = infer_domain_model(clean_exp)
    summary_row = load_single_row(compare_dir / "tables" / "robustness_summary.csv")

    row: dict[str, Any] = {
        "compare_name": compare_name,
        "clean_exp": clean_exp,
        "noisy_exp": noisy_exp,
        "domain": domain,
        "model": model,
    }
    row.update(summary_row)
    return row


def build_brief_summary(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "domain",
        "model",
        "clean_exp",
        "noisy_exp",
        "rerank_HR@10_drop",
        "rerank_NDCG@10_drop",
        "rerank_MRR@10_drop",
        "confidence_wrong_high_conf_fraction_drop",
        "calibration_ece_after_drop",
        "calibration_brier_score_after_drop",
    ]
    existing = [col for col in preferred if col in df.columns]
    return df[existing].copy()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    robustness_root = output_root / "robustness"

    if args.compare_names:
        compare_dirs = [robustness_root / name for name in args.compare_names]
    else:
        compare_dirs = discover_compare_dirs(robustness_root)

    rows = [aggregate_compare_dir(compare_dir) for compare_dir in compare_dirs if compare_dir.exists()]
    if not rows:
        raise ValueError(f"No robustness comparison folders found under {robustness_root}")

    summary_df = pd.DataFrame(rows).sort_values(["domain", "model", "compare_name"]).reset_index(drop=True)

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    full_output_path = summary_dir / "robustness_results.csv"
    brief_output_path = summary_dir / "robustness_brief.csv"

    summary_df.to_csv(full_output_path, index=False)
    build_brief_summary(summary_df).to_csv(brief_output_path, index=False)

    print(f"Aggregated {len(summary_df)} robustness rows.")
    print(f"Saved robustness summary to: {full_output_path}")
    print(f"Saved brief robustness summary to: {brief_output_path}")


if __name__ == "__main__":
    main()
