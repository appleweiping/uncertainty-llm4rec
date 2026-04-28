"""Prepare raw datasets into Storyflow observation tables and splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.data import DatasetPreparationError, prepare_movielens_1m
from storyflow.utils.config import load_simple_yaml


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_prepare_report(
    *,
    dataset: str,
    failure: str,
    expected_path: str,
    resume_command: str,
) -> Path:
    report_dir = ROOT / "local_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{_timestamp()}-{dataset}-prepare-note.md"
    path.write_text(
        "\n".join(
            [
                f"# {dataset} prepare 说明",
                "",
                f"- 当前状态：{failure}",
                f"- 期望路径：`{expected_path}`",
                f"- 恢复命令：`{resume_command}`",
                "",
                "该报告由 prepare 脚本自动生成；没有伪造预处理成功。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _config_path(dataset: str) -> Path:
    return ROOT / "configs" / "datasets" / f"{dataset}.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split-policy",
        choices=["leave_last_one", "leave_last_two", "global_chronological"],
    )
    parser.add_argument("--min-user-interactions", type=int)
    parser.add_argument("--user-k-core", type=int)
    parser.add_argument("--item-k-core", type=int)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--max-history", type=int)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--output-suffix")
    args = parser.parse_args(argv)

    path = _config_path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Unknown dataset config: {path}")
    config = load_simple_yaml(path)
    dataset_type = str(config.get("type"))

    try:
        if dataset_type != "movielens_1m":
            report = _write_prepare_report(
                dataset=str(config.get("name", args.dataset)),
                failure=(
                    f"`{dataset_type}` prepare 入口已规划，但本地预处理当前只执行 MovieLens 1M。"
                    " Amazon/Steam full-scale prepare 需要先在服务器或手动 raw 路径放置数据。"
                ),
                expected_path=str(config.get("raw_dir")),
                resume_command=f"python scripts/prepare_dataset.py --dataset {args.dataset}",
            )
            print(json.dumps({"status": "not_prepared", "report": str(report)}, ensure_ascii=False))
            return 2

        result = prepare_movielens_1m(
            config,
            split_policy=args.split_policy,
            min_user_interactions=args.min_user_interactions,
            user_k_core=args.user_k_core,
            item_k_core=args.item_k_core,
            min_history=args.min_history,
            max_history=args.max_history,
            max_users=args.max_users,
            output_suffix=args.output_suffix,
        )
    except DatasetPreparationError as exc:
        report = _write_prepare_report(
            dataset=str(config.get("name", args.dataset)),
            failure=str(exc),
            expected_path=str(Path(str(config["raw_dir"])) / str(config.get("expected_archive_dir", ""))),
            resume_command=f"python scripts/download_datasets.py --dataset {args.dataset}",
        )
        print(json.dumps({"status": "failed", "report": str(report)}, ensure_ascii=False))
        return 1

    print(
        json.dumps(
            {
                "status": "prepared",
                "dataset": result.dataset,
                "output_dir": str(result.output_dir),
                "item_count": result.item_count,
                "interaction_count": result.interaction_count,
                "user_count": result.user_count,
                "example_count": result.example_count,
                "split_counts": result.split_counts,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
