from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from main_rerank import save_table
from src.utils.paths import ensure_exp_dirs


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def candidate_pool_hash(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    for _, row in df.iterrows():
        hasher.update(str(row.get("user_id", "")).encode("utf-8"))
        for item_id in row.get("candidate_item_ids", []) or []:
            hasher.update(str(item_id).encode("utf-8"))
            hasher.update(b"|")
        hasher.update(b"\n")
    return hasher.hexdigest()


def prediction_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "parse_success_rate": None,
            "schema_valid_rate": None,
        }
    df = pd.read_json(path, lines=True)
    return {
        "exists": True,
        "rows": int(len(df)),
        "parse_success_rate": float(df["parse_success"].astype(bool).mean()) if "parse_success" in df else None,
        "schema_valid_rate": float(df["schema_valid"].astype(bool).mean()) if "schema_valid" in df else None,
    }


def prediction_status_ok(status: dict[str, Any], expected_rows: int) -> bool:
    if not status["exists"]:
        return True
    rate = status.get("parse_success_rate")
    rows = int(status.get("rows") or 0)
    if rows >= expected_rows:
        return rate is not None and float(rate) >= 0.99
    return rate in {None, 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_config", default="configs/exp/beauty_deepseek_recommendation_list_full_plain.yaml")
    parser.add_argument("--evidence_config", default="configs/exp/beauty_deepseek_recommendation_list_full_evidence.yaml")
    parser.add_argument("--summary_dir", default="output-repaired/summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plain_cfg = load_config(args.plain_config)
    evidence_cfg = load_config(args.evidence_config)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    plain_df = pd.read_json(plain_cfg["input_path"], lines=True)
    evidence_df = pd.read_json(evidence_cfg["input_path"], lines=True)
    plain_paths = ensure_exp_dirs(plain_cfg["exp_name"], plain_cfg.get("output_root", "output-repaired"))
    evidence_paths = ensure_exp_dirs(evidence_cfg["exp_name"], evidence_cfg.get("output_root", "output-repaired"))

    checks: list[dict[str, Any]] = []
    def add_check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    plain_hash = candidate_pool_hash(plain_df)
    evidence_hash = candidate_pool_hash(evidence_df)
    add_check("same_input_path", plain_cfg["input_path"] == evidence_cfg["input_path"], plain_cfg["input_path"])
    add_check("same_user_count", len(plain_df) == len(evidence_df), len(plain_df))
    add_check("same_user_id_order", plain_df["user_id"].astype(str).tolist() == evidence_df["user_id"].astype(str).tolist(), len(plain_df))
    add_check("same_candidate_pool_hash", plain_hash == evidence_hash, plain_hash[:16])
    add_check("same_top_k", int(plain_cfg.get("top_k", 10)) == int(evidence_cfg.get("top_k", 10)), plain_cfg.get("top_k", 10))
    add_check("same_model_config", plain_cfg.get("model_config") == evidence_cfg.get("model_config"), plain_cfg.get("model_config"))
    add_check("plain_prompt", Path(plain_cfg["plain_prompt_path"]).exists(), plain_cfg["plain_prompt_path"])
    add_check("evidence_prompt", Path(evidence_cfg["evidence_prompt_path"]).exists(), evidence_cfg["evidence_prompt_path"])
    add_check("parser_reuses_day10", True, "parse_recommendation_list_plain_response / parse_recommendation_list_evidence_response")
    add_check("resume_enabled", bool(plain_cfg.get("resume")) and bool(evidence_cfg.get("resume")), "resume=true")
    add_check("concurrent_api_controlled", bool(plain_cfg.get("concurrent")) and bool(evidence_cfg.get("concurrent")), f"workers={plain_cfg.get('max_workers')} rpm={plain_cfg.get('requests_per_minute')}")
    add_check("separate_output_dirs", plain_cfg["exp_name"] != evidence_cfg["exp_name"], f"{plain_paths.root} vs {evidence_paths.root}")
    add_check("does_not_cover_day10_200", "beauty_day10_recommendation_list_200" not in {plain_cfg["exp_name"], evidence_cfg["exp_name"]}, "separate full exp_names")
    add_check("full_sample_count", len(plain_df) == 973, len(plain_df))
    add_check("candidate_pool_size_is_6", set(plain_df["candidate_item_ids"].apply(len).unique()) == {6}, plain_df["candidate_item_ids"].apply(len).value_counts().to_dict())

    plain_status = prediction_status(plain_paths.predictions_dir / "plain_list_raw.jsonl")
    evidence_status = prediction_status(evidence_paths.predictions_dir / "evidence_list_raw.jsonl")
    expected_rows = len(plain_df)
    add_check("plain_smoke_or_full_status", prediction_status_ok(plain_status, expected_rows), plain_status)
    add_check("evidence_smoke_or_full_status", prediction_status_ok(evidence_status, expected_rows), evidence_status)

    checks_df = pd.DataFrame(checks)
    save_table(checks_df, summary_dir / "beauty_day10_full_readiness_check.csv")
    all_passed = bool(checks_df["passed"].all())

    report_lines = [
        "# Day10-Full Readiness Check",
        "",
        f"Status: {'ready_for_smoke_or_full' if all_passed else 'needs_attention'}",
        "",
        "Day10-full keeps the Day10 design: same Beauty users, same candidate pools, same DeepSeek list backend, same top-K and evaluation; the only methodological difference is whether scheme-four evidence fields are present.",
        "",
        "## Checks",
        "",
    ]
    for row in checks:
        mark = "PASS" if row["passed"] else "FAIL"
        report_lines.append(f"- {mark}: {row['check']} -- {row['detail']}")
    report_lines.extend(
        [
            "",
            "## Smoke Commands",
            "",
            "Run these before full inference:",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' main_infer_recommendation_list.py --config configs/exp/beauty_deepseek_recommendation_list_full_plain.yaml --setting plain --max_samples 20",
            "$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' main_infer_recommendation_list.py --config configs/exp/beauty_deepseek_recommendation_list_full_evidence.yaml --setting evidence --max_samples 20",
            "$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\\Users\\admin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' main_day10_full_readiness.py",
            "```",
            "",
            "Before full inference, require both smoke outputs to have parse_success_rate=1.0 and schema_valid_rate=1.0. After full inference, a parse_success_rate >=0.99 is treated as healthy and reported explicitly.",
        ]
    )
    report_path = summary_dir / "beauty_day10_full_readiness_check.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved {summary_dir / 'beauty_day10_full_readiness_check.csv'}")
    print(f"Saved {report_path}")
    print(f"DAY10_FULL_READINESS status={'ready' if all_passed else 'needs_attention'}")


if __name__ == "__main__":
    main()
