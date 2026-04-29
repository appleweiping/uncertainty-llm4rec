"""Prepare Amazon Reviews 2023 JSONL files when local raw data is available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.data import prepare_amazon_from_jsonl, resolve_existing_raw_path  # noqa: E402
from storyflow.utils.config import load_simple_yaml  # noqa: E402


def _config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def _write_missing_report_legacy_mojibake(
    *,
    dataset: str,
    reviews_jsonl: Path,
    metadata_jsonl: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "prepare_readiness_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {dataset} prepare readiness",
                "",
                "当前没有执行 full Amazon preprocessing，因为所需 raw JSONL 文件不存在或未显式提供。",
                "",
                f"- reviews JSONL: `{reviews_jsonl}`",
                f"- metadata JSONL: `{metadata_jsonl}`",
                "- 恢复命令模板:",
                "",
                "```powershell",
                f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix full --allow-full",
                "```",
                "",
                "该报告不是 full processed result。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _write_full_guard_report_legacy_mojibake(
    *,
    dataset: str,
    reviews_jsonl: Path,
    metadata_jsonl: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "prepare_full_guard_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {dataset} full prepare guard",
                "",
                "检测到命令会处理 full Amazon raw JSONL，但未提供 `--allow-full`。",
                "",
                "本项目要求 full Amazon preprocessing 只在明确批准后执行；local sample 请使用 `--sample-mode --max-records N`。",
                "",
                f"- reviews JSONL: `{reviews_jsonl}`",
                f"- metadata JSONL: `{metadata_jsonl}`",
                "",
                "如果用户已经批准 full prepare，可恢复执行:",
                "",
                "```powershell",
                f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix full --allow-full",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _write_missing_report(
    *,
    dataset: str,
    reviews_jsonl: Path,
    metadata_jsonl: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "prepare_readiness_report.md"
    lines = [
        f"# {dataset} prepare readiness",
        "",
        "当前没有执行 full Amazon preprocessing，因为所需 raw JSONL 文件不存在、未显式提供，或本次为 dry-run。",
        "",
        f"- reviews JSONL: `{reviews_jsonl}`",
        f"- metadata JSONL: `{metadata_jsonl}`",
        "- 恢复命令模板:",
        "",
        "```powershell",
        f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix full --allow-full",
        "```",
        "",
        "该报告不是 full processed result。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_full_guard_report(
    *,
    dataset: str,
    reviews_jsonl: Path,
    metadata_jsonl: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "prepare_full_guard_report.md"
    lines = [
        f"# {dataset} full prepare guard",
        "",
        "检测到命令会处理 full Amazon raw JSONL，但没有提供 `--allow-full`。",
        "",
        "本项目要求 full Amazon preprocessing 只在明确批准后执行；local sample 请使用 `--sample-mode --max-records N`。",
        "",
        f"- reviews JSONL: `{reviews_jsonl}`",
        f"- metadata JSONL: `{metadata_jsonl}`",
        "",
        "如果用户已经批准 full prepare，可恢复执行:",
        "",
        "```powershell",
        f"python scripts/prepare_amazon_reviews_2023.py --dataset {dataset} --reviews-jsonl {reviews_jsonl} --metadata-jsonl {metadata_jsonl} --output-suffix full --allow-full",
        "```",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _apply_preprocess_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    override_map = {
        "min_user_interactions": "preprocess_min_user_interactions",
        "user_k_core": "preprocess_user_k_core",
        "item_k_core": "preprocess_item_k_core",
        "min_history": "preprocess_min_history",
        "max_history": "preprocess_max_history",
        "split_policy": "preprocess_split_policy",
    }
    for arg_name, config_name in override_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            updated[config_name] = value
    return updated


def _derive_run_scope(config: dict[str, Any], args: argparse.Namespace) -> tuple[str, int | None]:
    max_records = args.max_records
    if args.sample_mode and max_records is None:
        max_records = int(config.get("local_sample_max_records") or 1000)
    if args.output_suffix:
        output_suffix = args.output_suffix
    elif args.sample_mode:
        output_suffix = f"sample_{max_records}"
    else:
        output_suffix = "full"
    return output_suffix, max_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reviews-jsonl")
    parser.add_argument("--metadata-jsonl")
    parser.add_argument("--output-suffix")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--min-user-interactions", type=int)
    parser.add_argument("--user-k-core", type=int)
    parser.add_argument("--item-k-core", type=int)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--max-history", type=int)
    parser.add_argument(
        "--split-policy",
        choices=["global_chronological", "leave_last_one", "leave_last_two"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = _config_path(args.dataset)
    if not config_path.exists():
        raise SystemExit(f"Unknown dataset config: {config_path}")
    config = _apply_preprocess_overrides(load_simple_yaml(config_path), args)
    output_suffix, max_records = _derive_run_scope(config, args)
    reviews_jsonl = (
        Path(args.reviews_jsonl)
        if args.reviews_jsonl
        else resolve_existing_raw_path(config, "raw_reviews_path")
    )
    metadata_jsonl = (
        Path(args.metadata_jsonl)
        if args.metadata_jsonl
        else resolve_existing_raw_path(config, "raw_metadata_path")
    )
    output_dir = ROOT / "outputs" / "amazon_reviews_2023" / args.dataset / "prepare"
    if args.dry_run or not reviews_jsonl.exists() or not metadata_jsonl.exists():
        report = _write_missing_report(
            dataset=args.dataset,
            reviews_jsonl=reviews_jsonl,
            metadata_jsonl=metadata_jsonl,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "status": "readiness_only",
                    "dataset": args.dataset,
                    "reviews_jsonl_exists": reviews_jsonl.exists(),
                    "metadata_jsonl_exists": metadata_jsonl.exists(),
                    "report": str(report),
                    "full_processed": False,
                    "sample_processed": False,
                    "note": "No full Amazon data was processed.",
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if args.dry_run else 2
    if max_records is None and not args.allow_full:
        report = _write_full_guard_report(
            dataset=args.dataset,
            reviews_jsonl=reviews_jsonl,
            metadata_jsonl=metadata_jsonl,
            output_dir=output_dir,
        )
        print(
            json.dumps(
                {
                    "status": "blocked_full_prepare_requires_allow_full",
                    "dataset": args.dataset,
                    "reviews_jsonl_exists": reviews_jsonl.exists(),
                    "metadata_jsonl_exists": metadata_jsonl.exists(),
                    "report": str(report),
                    "full_processed": False,
                    "sample_processed": False,
                    "note": "Use --sample-mode --max-records N for local readiness, or add --allow-full only after explicit full-run approval.",
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2

    summary = prepare_amazon_from_jsonl(
        config=config,
        reviews_jsonl=reviews_jsonl,
        metadata_jsonl=metadata_jsonl,
        output_suffix=output_suffix,
        max_records=max_records,
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "dataset": summary.dataset,
                "output_dir": str(summary.output_dir),
                "item_count": summary.item_count,
                "interaction_count": summary.interaction_count,
                "user_count": summary.user_count,
                "example_count": summary.example_count,
                "split_counts": summary.split_counts,
                "full_processed": max_records is None,
                "sample_processed": max_records is not None,
                "output_suffix": output_suffix,
                "max_records": max_records,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
