"""Day29 Movies medium_5neg_2000 preflight and smoke status reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml")
SUMMARY_DIR = Path("output-repaired/summary")
EXP_DIR = Path("output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000")


def _read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            count += chunk.count(b"\n")
    return count


def _split_stats(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    users: dict[str, int] = defaultdict(int)
    labels = Counter()
    missing = Counter()
    for row in rows:
        users[str(row.get("user_id", ""))] += 1
        labels[int(row.get("label", -1))] += 1
        for field in ["user_id", "history", "candidate_item_id", "candidate_text", "label"]:
            if field not in row or row.get(field) in (None, ""):
                missing[field] += 1
        if "candidate_title" not in row or row.get("candidate_title") in (None, ""):
            missing["candidate_title_optional"] += 1
    counts = list(users.values())
    required_missing = {k: v for k, v in missing.items() if k != "candidate_title_optional" and v}
    return {
        "rows": len(rows),
        "users": len(users),
        "candidate_pool_size_min": min(counts) if counts else 0,
        "candidate_pool_size_mean": sum(counts) / len(counts) if counts else 0,
        "candidate_pool_size_max": max(counts) if counts else 0,
        "positive_rows": labels.get(1, 0),
        "negative_rows": labels.get(0, 0),
        "missing_fields": "; ".join(f"{k}:{v}" for k, v in sorted(missing.items()) if v) or "",
        "schema_ok": not required_missing,
    }


def preflight() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    train_path = Path(cfg["train_input_path"])
    valid_path = Path(cfg["split_input_paths"]["valid"])
    test_path = Path(cfg["split_input_paths"]["test"])
    train_rows = _line_count(train_path)
    valid = _split_stats(valid_path)
    test = _split_stats(test_path)
    checks = {
        "config_exists": CONFIG_PATH.exists(),
        "prompt_path_ok": cfg.get("prompt_path") == "prompts/candidate_relevance_evidence.txt",
        "output_dir_ok": cfg.get("output_dir") == str(EXP_DIR).replace("\\", "/"),
        "schema_ok": cfg.get("output_schema") == "relevance_evidence",
        "resume_enabled": bool(cfg.get("resume")),
        "valid_rows_ok": valid["rows"] == 12000,
        "test_rows_ok": test["rows"] == 12000,
        "valid_candidate_pool_ok": valid["candidate_pool_size_min"] == 6 and valid["candidate_pool_size_max"] == 6,
        "test_candidate_pool_ok": test["candidate_pool_size_min"] == 6 and test["candidate_pool_size_max"] == 6,
        "valid_schema_ok": valid["schema_ok"],
        "test_schema_ok": test["schema_ok"],
        "output_dir_isolated": "movies_deepseek_relevance_evidence_medium_5neg_2000" in str(EXP_DIR),
    }
    passed = all(checks.values())
    lines = [
        "# Day29 Movies medium_5neg_2000 Preflight Check",
        "",
        "## 20neg_2000 Pause Note",
        "",
        "Movies medium_20neg_2000 preflight and tiny smoke passed, and partial valid inference is preserved. Full 20neg_2000 inference was intentionally paused due to API/runtime cost.",
        "",
        f"- config: `{CONFIG_PATH}`",
        f"- train_input_path: `{train_path}`",
        f"- valid_input_path: `{valid_path}`",
        f"- test_input_path: `{test_path}`",
        f"- prompt_path: `{cfg.get('prompt_path')}`",
        f"- output_dir: `{cfg.get('output_dir')}`",
        f"- output_schema: `{cfg.get('output_schema')}`",
        f"- resume: `{cfg.get('resume')}`",
        f"- concurrent: `{cfg.get('concurrent')}`",
        "",
        "## Split Stats",
        "",
        "| split | rows | users | pool_min | pool_mean | pool_max | positive | negative | missing_fields |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| train | {train_rows} |  |  |  |  |  |  | not fully scanned |",
        f"| valid | {valid['rows']} | {valid['users']} | {valid['candidate_pool_size_min']} | {valid['candidate_pool_size_mean']:.2f} | {valid['candidate_pool_size_max']} | {valid['positive_rows']} | {valid['negative_rows']} | {valid['missing_fields']} |",
        f"| test | {test['rows']} | {test['users']} | {test['candidate_pool_size_min']} | {test['candidate_pool_size_mean']:.2f} | {test['candidate_pool_size_max']} | {test['positive_rows']} | {test['negative_rows']} | {test['missing_fields']} |",
        "",
        "## Metric Note",
        "",
        "Each user has 6 candidates, so `hr10_trivial_flag=true`. HR@10 must not be used as primary evidence; use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Decision", "", "Preflight passed. Tiny API smoke can start." if passed else "Preflight failed. Do not start API inference."])
    (SUMMARY_DIR / "day29_movies_medium5_2000_preflight_check.md").write_text("\n".join(lines), encoding="utf-8")


def smoke_report() -> None:
    paths = {
        "valid": EXP_DIR / "smoke" / "valid_raw_50.jsonl",
        "test": EXP_DIR / "smoke" / "test_raw_50.jsonl",
    }
    lines = ["# Day29 Movies medium_5neg_2000 Smoke Report", ""]
    passed_all = True
    for split, path in paths.items():
        rows = _read_jsonl(path)
        parse_success = sum(1 for row in rows if row.get("parse_success") is True)
        field_counts = {
            field: sum(1 for row in rows if row.get(field) is not None)
            for field in ["relevance_probability", "positive_evidence", "negative_evidence", "ambiguity", "missing_information", "evidence_risk", "raw_response"]
        }
        rate = parse_success / len(rows) if rows else 0.0
        passed = len(rows) > 0 and rate >= 0.95 and all(v == len(rows) for v in field_counts.values())
        passed_all = passed_all and passed
        lines.extend([f"## {split}", "", f"- path: `{path}`", f"- rows: `{len(rows)}`", f"- parse_success_rate: `{rate:.4f}`", f"- passed: `{passed}`", f"- field_counts: `{field_counts}`", ""])
    lines.extend(["## Decision", "", "Smoke passed. Full valid/test inference can start." if passed_all else "Smoke failed. Do not start full inference."])
    (SUMMARY_DIR / "day29_movies_medium5_2000_smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def runtime_monitor() -> None:
    lines = [
        "# Day29 Movies medium_5neg_2000 Runtime Monitor",
        "",
        "- expected_valid_rows: `12000`",
        "- expected_test_rows: `12000`",
        "- output_dir: `output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000/`",
        "- metric_note: `HR@10 is trivial because candidate_pool_size=6.`",
        "",
        "## Row Count Checks",
        "",
        "```powershell",
        "(Get-Content output-repaired\\movies_deepseek_relevance_evidence_medium_5neg_2000\\predictions\\valid_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines",
        "(Get-Content output-repaired\\movies_deepseek_relevance_evidence_medium_5neg_2000\\predictions\\test_raw.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines",
        "```",
        "",
        "## Resume Commands",
        "",
        "```powershell",
        "py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml --split_name valid --concurrent --resume",
        "py -3.12 main_infer.py --config configs/exp/movies_deepseek_relevance_evidence_medium_5neg_2000.yaml --split_name test --concurrent --resume",
        "```",
    ]
    (SUMMARY_DIR / "day29_movies_medium5_2000_runtime_monitor.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "smoke_report", "runtime_monitor"], required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "smoke_report":
        smoke_report()
    elif args.stage == "runtime_monitor":
        runtime_monitor()


if __name__ == "__main__":
    main()
