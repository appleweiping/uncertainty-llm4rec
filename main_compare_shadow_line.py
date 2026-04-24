from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _read_first_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return pd.read_csv(path).iloc[0].to_dict()


def _read_rerank_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "method" not in df.columns:
        return df.iloc[-1].to_dict()
    rerank_df = df[df["method"].astype(str).str.contains("uncertainty_aware", na=False)]
    if rerank_df.empty:
        return df.iloc[-1].to_dict()
    return rerank_df.iloc[0].to_dict()


def _float(row: dict[str, Any], key: str) -> float | str:
    value = row.get(key, "")
    if value == "":
        return ""
    try:
        return float(value)
    except Exception:
        return ""


def _status(path: Path) -> str:
    return "ready" if path.exists() else "missing"


def _exp_prefix(domain: str, variant: str, scenario: str) -> str:
    return f"{domain}_qwen3_{variant}_{scenario}"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _to_markdown_without_tabulate(df: pd.DataFrame) -> str:
    if df.empty:
        return "\n"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("\n", " ") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_summary(
    *,
    scenario: str,
    variants: list[str],
    domains: list[str],
    output_root: str = "outputs",
) -> pd.DataFrame:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for variant in variants:
            prefix = _exp_prefix(domain, variant, scenario)
            pointwise_exp = f"{prefix}_pointwise"
            rerank_exp = f"{prefix}_structured_risk"
            noisy_pointwise_exp = f"{pointwise_exp}_noisy_nl10"
            noisy_rerank_exp = f"{rerank_exp}_noisy_nl10"

            pointwise_path = root / pointwise_exp / "tables" / "diagnostic_metrics.csv"
            calib_path = root / pointwise_exp / "tables" / "calibration_comparison.csv"
            rerank_path = root / rerank_exp / "tables" / "rerank_results.csv"
            noisy_pointwise_path = root / noisy_pointwise_exp / "tables" / "diagnostic_metrics.csv"
            noisy_rerank_path = root / noisy_rerank_exp / "tables" / "rerank_results.csv"

            pointwise = _read_first_row(pointwise_path)
            rerank = _read_rerank_row(rerank_path)
            noisy_pointwise = _read_first_row(noisy_pointwise_path)
            noisy_rerank = _read_rerank_row(noisy_rerank_path)

            calib_after_ece = ""
            calib_after_brier = ""
            if calib_path.exists():
                calib_df = pd.read_csv(calib_path)
                test_ece = calib_df[(calib_df["split"] == "test") & (calib_df["metric"] == "ece")]
                test_brier = calib_df[(calib_df["split"] == "test") & (calib_df["metric"] == "brier_score")]
                if not test_ece.empty:
                    calib_after_ece = float(test_ece.iloc[0]["after"])
                if not test_brier.empty:
                    calib_after_brier = float(test_brier.iloc[0]["after"])

            clean_ndcg = _float(rerank, "NDCG@10")
            noisy_ndcg = _float(noisy_rerank, "NDCG@10")
            ndcg_drop = ""
            if clean_ndcg != "" and noisy_ndcg != "":
                ndcg_drop = float(clean_ndcg) - float(noisy_ndcg)

            rows.append(
                {
                    "week_stage": "week7_9_shadow",
                    "scenario": scenario,
                    "domain": domain,
                    "shadow_variant": variant,
                    "pointwise_exp_name": pointwise_exp,
                    "rerank_exp_name": rerank_exp,
                    "noisy_pointwise_exp_name": noisy_pointwise_exp,
                    "noisy_rerank_exp_name": noisy_rerank_exp,
                    "pointwise_status": _status(pointwise_path),
                    "calibration_status": _status(calib_path),
                    "rerank_status": _status(rerank_path),
                    "noisy_pointwise_status": _status(noisy_pointwise_path),
                    "noisy_rerank_status": _status(noisy_rerank_path),
                    "pointwise_auroc": _float(pointwise, "auroc"),
                    "pointwise_ece": _float(pointwise, "ece"),
                    "pointwise_brier": _float(pointwise, "brier_score"),
                    "pointwise_accuracy": _float(pointwise, "accuracy"),
                    "calibrated_ece": calib_after_ece,
                    "calibrated_brier": calib_after_brier,
                    "rerank_ndcg_at_10": clean_ndcg,
                    "rerank_mrr": _float(rerank, "MRR"),
                    "rerank_coverage_at_10": _float(rerank, "coverage@10"),
                    "rerank_head_exposure_ratio_at_10": _float(rerank, "head_exposure_ratio@10"),
                    "rerank_longtail_coverage_at_10": _float(rerank, "longtail_coverage@10"),
                    "noisy_pointwise_auroc": _float(noisy_pointwise, "auroc"),
                    "noisy_rerank_ndcg_at_10": noisy_ndcg,
                    "ndcg_drop_noisy": ndcg_drop,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="small_prior")
    parser.add_argument("--variants", default="shadow_v1,shadow_v2,shadow_v3,shadow_v4,shadow_v5,shadow_v6")
    parser.add_argument("--domains", default="beauty,books,electronics,movies")
    parser.add_argument("--output_root", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    df = build_summary(
        scenario=args.scenario,
        variants=variants,
        domains=domains,
        output_root=args.output_root,
    )
    out_path = Path(args.output_root) / "summary" / f"week7_9_shadow_{args.scenario}_summary.csv"
    _write(df, out_path)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_to_markdown_without_tabulate(df), encoding="utf-8")
    print(f"Saved shadow summary to: {out_path}")
    print(f"Saved shadow markdown to: {md_path}")
    ready = df[["pointwise_status", "calibration_status", "rerank_status"]].eq("ready").all(axis=1).sum()
    print(f"ready_core_rows={ready}/{len(df)}")


if __name__ == "__main__":
    main()
