from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


TASK_SPECS: dict[str, dict[str, str]] = {
    "pointwise_yesno": {
        "suffix": "_pointwise",
        "metrics_filename": "diagnostic_metrics.csv",
    },
    "candidate_ranking": {
        "suffix": "_rank",
        "metrics_filename": "ranking_metrics.csv",
    },
    "pairwise_preference": {
        "suffix": "_pairwise",
        "metrics_filename": "pairwise_metrics.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--exp_names", type=str, nargs="*", default=None)
    return parser.parse_args()


def resolve_table_path(exp_dir: Path, filename: str) -> Path:
    candidates = [
        exp_dir / filename,
        exp_dir / "tables" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required result file `{filename}` under {exp_dir}")


def resolve_predictions_path(exp_dir: Path, filename: str) -> Path | None:
    candidates = [
        exp_dir / filename,
        exp_dir / "predictions" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def infer_task_name(exp_name: str) -> str | None:
    normalized = exp_name.strip().lower()
    for task_name, spec in TASK_SPECS.items():
        if normalized.endswith(spec["suffix"]):
            return task_name
    return None


def strip_task_suffix(exp_name: str, task_name: str) -> str:
    suffix = TASK_SPECS[task_name]["suffix"]
    if exp_name.lower().endswith(suffix):
        return exp_name[: -len(suffix)]
    return exp_name


def infer_domain_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    ignore_tokens = {
        "deepseek",
        "qwen",
        "kimi",
        "doubao",
        "glm",
        "gpt",
        "openai",
        "local",
        "small",
        "mini",
        "large",
        "clean",
        "noisy",
        "robustness",
        "calibration",
        "rerank",
        "reranking",
        "pointwise",
        "rank",
        "pairwise",
    }
    domain_tokens = [token for token in tokens if token not in ignore_tokens and not token.startswith("lambda")]
    if not domain_tokens:
        return exp_name.lower()

    domain = domain_tokens[0]
    if domain.startswith("movie"):
        return "movies"
    if domain.startswith("book"):
        return "books"
    if domain.startswith("electronic"):
        return "electronics"
    return domain


def infer_model_name(exp_name: str) -> str:
    tokens = [token.strip().lower() for token in exp_name.split("_") if token.strip()]
    known_models = ["deepseek", "qwen", "kimi", "doubao", "glm", "gpt", "openai", "local"]
    for token in reversed(tokens):
        if token in known_models:
            return token
    return "unknown"


def format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "-"
    if math.isnan(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def extract_head_exposure(path: Path) -> float:
    exposure_df = pd.read_csv(path)
    if exposure_df.empty or "target_popularity_group" not in exposure_df.columns:
        return float("nan")
    head_rows = exposure_df[
        exposure_df["target_popularity_group"].astype(str).str.strip().str.lower() == "head"
    ]
    if head_rows.empty or "high_conf_fraction" not in head_rows.columns:
        return float("nan")
    return float(head_rows.iloc[0]["high_conf_fraction"])


def extract_prediction_stats(path: Path | None) -> tuple[float, float]:
    if path is None or not path.exists():
        return float("nan"), float("nan")

    prediction_df = pd.read_json(path, lines=True)
    parse_success_rate = (
        float(prediction_df["parse_success"].fillna(False).astype(bool).mean())
        if "parse_success" in prediction_df.columns
        else float("nan")
    )
    avg_latency = (
        float(pd.to_numeric(prediction_df["latency"], errors="coerce").mean())
        if "latency" in prediction_df.columns
        else float("nan")
    )
    return parse_success_rate, avg_latency


def load_pointwise_row(exp_dir: Path, exp_name: str) -> dict[str, Any]:
    metrics_path = resolve_table_path(exp_dir, "diagnostic_metrics.csv")
    correctness_path = resolve_table_path(exp_dir, "confidence_correctness_summary.csv")
    exposure_path = resolve_table_path(exp_dir, "high_confidence_exposure.csv")
    predictions_path = resolve_predictions_path(exp_dir, "test_raw.jsonl")

    metrics_row = pd.read_csv(metrics_path).iloc[0].to_dict()
    correctness_row = pd.read_csv(correctness_path).iloc[0].to_dict()
    parse_success_rate, avg_latency = extract_prediction_stats(predictions_path)

    return {
        "exp_name": exp_name,
        "domain": infer_domain_name(exp_name),
        "model": infer_model_name(exp_name),
        "task": "pointwise_yesno",
        "task_role": "diagnostic_layer",
        "samples": metrics_row.get("num_samples"),
        "HR@10": float("nan"),
        "NDCG@10": float("nan"),
        "MRR": float("nan"),
        "pairwise_accuracy": float("nan"),
        "ECE": metrics_row.get("ece"),
        "Brier": metrics_row.get("brier_score"),
        "coverage": float("nan"),
        "head_exposure": extract_head_exposure(exposure_path),
        "longtail_coverage": float("nan"),
        "parse_success_rate": parse_success_rate,
        "avg_latency": avg_latency,
        "avg_confidence": metrics_row.get("avg_confidence"),
        "diagnostic_accuracy": metrics_row.get("accuracy"),
        "high_conf_accuracy": correctness_row.get("high_conf_accuracy"),
        "wrong_high_conf_fraction": correctness_row.get("wrong_high_conf_fraction"),
        "preference_consistency": float("nan"),
        "strict_preference_consistency": float("nan"),
        "out_of_candidate_rate": float("nan"),
        "source_path": str(metrics_path),
    }


def load_ranking_row(exp_dir: Path, exp_name: str) -> dict[str, Any]:
    metrics_path = resolve_table_path(exp_dir, "ranking_metrics.csv")
    metrics_row = pd.read_csv(metrics_path).iloc[0].to_dict()

    return {
        "exp_name": exp_name,
        "domain": infer_domain_name(exp_name),
        "model": infer_model_name(exp_name),
        "task": "candidate_ranking",
        "task_role": "decision_layer",
        "samples": metrics_row.get("sample_count"),
        "HR@10": metrics_row.get("HR@10"),
        "NDCG@10": metrics_row.get("NDCG@10"),
        "MRR": metrics_row.get("MRR"),
        "pairwise_accuracy": float("nan"),
        "ECE": float("nan"),
        "Brier": float("nan"),
        "coverage": metrics_row.get("coverage@10"),
        "head_exposure": metrics_row.get("head_exposure_ratio@10"),
        "longtail_coverage": metrics_row.get("longtail_coverage@10"),
        "parse_success_rate": metrics_row.get("parse_success_rate"),
        "avg_latency": metrics_row.get("avg_latency"),
        "avg_confidence": metrics_row.get("avg_confidence"),
        "diagnostic_accuracy": float("nan"),
        "high_conf_accuracy": float("nan"),
        "wrong_high_conf_fraction": float("nan"),
        "preference_consistency": float("nan"),
        "strict_preference_consistency": float("nan"),
        "out_of_candidate_rate": metrics_row.get("out_of_candidate_rate"),
        "source_path": str(metrics_path),
    }


def load_pairwise_row(exp_dir: Path, exp_name: str) -> dict[str, Any]:
    metrics_path = resolve_table_path(exp_dir, "pairwise_metrics.csv")
    metrics_row = pd.read_csv(metrics_path).iloc[0].to_dict()

    return {
        "exp_name": exp_name,
        "domain": infer_domain_name(exp_name),
        "model": infer_model_name(exp_name),
        "task": "pairwise_preference",
        "task_role": "mechanism_layer",
        "samples": metrics_row.get("sample_count"),
        "HR@10": float("nan"),
        "NDCG@10": float("nan"),
        "MRR": float("nan"),
        "pairwise_accuracy": metrics_row.get("pairwise_accuracy"),
        "ECE": float("nan"),
        "Brier": float("nan"),
        "coverage": float("nan"),
        "head_exposure": float("nan"),
        "longtail_coverage": float("nan"),
        "parse_success_rate": metrics_row.get("parse_success_rate"),
        "avg_latency": metrics_row.get("avg_latency"),
        "avg_confidence": metrics_row.get("avg_confidence"),
        "diagnostic_accuracy": float("nan"),
        "high_conf_accuracy": float("nan"),
        "wrong_high_conf_fraction": float("nan"),
        "preference_consistency": metrics_row.get("preference_consistency"),
        "strict_preference_consistency": metrics_row.get("strict_preference_consistency"),
        "out_of_candidate_rate": float("nan"),
        "source_path": str(metrics_path),
    }


def aggregate_experiment(exp_dir: Path) -> dict[str, Any]:
    exp_name = exp_dir.name
    task_name = infer_task_name(exp_name)
    if task_name is None:
        raise ValueError(f"Cannot infer multitask type from experiment name: {exp_name}")

    if task_name == "pointwise_yesno":
        return load_pointwise_row(exp_dir, exp_name)
    if task_name == "candidate_ranking":
        return load_ranking_row(exp_dir, exp_name)
    if task_name == "pairwise_preference":
        return load_pairwise_row(exp_dir, exp_name)

    raise ValueError(f"Unsupported task type: {task_name}")


def discover_experiment_dirs(output_root: Path) -> list[Path]:
    experiment_dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or child.name in {"summary", "robustness", "baselines"}:
            continue
        task_name = infer_task_name(child.name)
        if task_name is None:
            continue
        required_filename = TASK_SPECS[task_name]["metrics_filename"]
        try:
            resolve_table_path(child, required_filename)
        except FileNotFoundError:
            continue
        experiment_dirs.append(child)
    return experiment_dirs


def filter_summary(summary_df: pd.DataFrame, domain: str | None, model: str | None) -> pd.DataFrame:
    out = summary_df.copy()
    if domain:
        out = out[out["domain"].astype(str).str.lower() == domain.strip().lower()].copy()
    if model:
        out = out[out["model"].astype(str).str.lower() == model.strip().lower()].copy()
    return out


def build_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column)
            if column in {"samples"}:
                values.append(format_metric(value, digits=0))
            elif column in {
                "HR@10",
                "NDCG@10",
                "MRR",
                "pairwise_accuracy",
                "ECE",
                "Brier",
                "coverage",
                "head_exposure",
                "longtail_coverage",
                "parse_success_rate",
                "avg_latency",
            }:
                values.append(format_metric(value, digits=4))
            else:
                values.append(str(value) if value is not None and str(value) != "nan" else "-")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def summarize_complete_bundle(bundle_df: pd.DataFrame) -> list[str]:
    pointwise_row = bundle_df[bundle_df["task"] == "pointwise_yesno"].iloc[0]
    ranking_row = bundle_df[bundle_df["task"] == "candidate_ranking"].iloc[0]
    pairwise_row = bundle_df[bundle_df["task"] == "pairwise_preference"].iloc[0]

    domain = str(pointwise_row["domain"])
    model = str(pointwise_row["model"])
    samples = format_metric(pointwise_row["samples"], digits=0)

    return [
        f"{domain}/{model} 这一组 week5 多任务闭环已经同时具备 pointwise、pairwise、candidate ranking 三层任务结果，当前比较基础来自 {samples} 条样本级第一轮工程验证，因此它更适合作为 week5 的系统收口证据，而不是最终全量实验结论。",
        f"pointwise 仍然适合作为 uncertainty diagnosis 层，因为它已经稳定提供 ECE={format_metric(pointwise_row['ECE'])}、Brier={format_metric(pointwise_row['Brier'])}、high-conf accuracy={format_metric(pointwise_row['high_conf_accuracy'])} 等诊断量；尤其当前 wrong-high-conf fraction={format_metric(pointwise_row['wrong_high_conf_fraction'])}，说明高置信错误并不低，这恰恰证明 pointwise 在项目里最有价值的角色是暴露风险，而不是承担最终决策。",
        f"pairwise 已经进入局部偏好机制层，因为它不再只是 prompt 自测，而是有了 pairwise accuracy={format_metric(pairwise_row['pairwise_accuracy'])}、preference consistency={format_metric(pairwise_row['preference_consistency'])}、strict consistency={format_metric(pairwise_row['strict_preference_consistency'])} 的正式评估口径。当前数字说明它已经能表达局部偏好结构，但稳定性仍然有限，因此更适合做机制分析与中间层表示，而不是直接替代主排序任务。",
        f"candidate ranking 已经开始承担主决策角色，因为它已经进入 HR/NDCG/MRR/coverage/exposure 的真实推荐指标空间。当前 HR@10={format_metric(ranking_row['HR@10'])}、NDCG@10={format_metric(ranking_row['NDCG@10'])}、MRR={format_metric(ranking_row['MRR'])}、coverage={format_metric(ranking_row['coverage'])}、head exposure={format_metric(ranking_row['head_exposure'])}，说明它已经是能被正式评价的列表级决策接口。需要谨慎的是，当前候选集较小，因此 HR@10 偏宽松，后续更应依赖 NDCG、MRR 和曝光行为来判断方法质量。",
    ]


def build_markdown_summary(summary_df: pd.DataFrame) -> str:
    display_columns = [
        "domain",
        "model",
        "task",
        "samples",
        "HR@10",
        "NDCG@10",
        "MRR",
        "pairwise_accuracy",
        "ECE",
        "Brier",
        "coverage",
        "head_exposure",
        "longtail_coverage",
    ]

    complete_groups = []
    for (domain, model), group_df in summary_df.groupby(["domain", "model"], dropna=False):
        task_set = set(group_df["task"].astype(str))
        if task_set == set(TASK_SPECS.keys()):
            complete_groups.append((domain, model, group_df.sort_values("task")))

    paragraphs = [
        "# Week5 Multitask Summary",
        "",
        "这份总结文件对应 week5 的论文级收口阶段。它不是新方法实验，而是把 week5 前四天已经打通的数据、推理、评估三层结果收束成一套可以直接服务论文叙事的多任务比较摘要。当前结果应被理解为第一轮工程级闭环证据：它已经足以回答 Part5 的结构性问题，但还不应替代后续更大规模、更多模型与更多 domain 的正式结论。",
        "",
        "## Structured Table",
        "",
        build_markdown_table(summary_df, display_columns),
        "",
        "## Task-Level Answers",
        "",
    ]

    if complete_groups:
        for domain, model, bundle_df in complete_groups:
            paragraphs.append(f"### {domain}/{model}")
            paragraphs.append("")
            paragraphs.extend(summarize_complete_bundle(bundle_df))
            paragraphs.append("")
    else:
        paragraphs.extend(
            [
                "当前尚未发现同时具备 pointwise、pairwise、candidate ranking 三层结果的完整 domain/model 组合，因此 week5 的自然语言总结只能停留在部分任务层面。这通常意味着还有任务链没有完全跑齐，而不是方法已经失败。",
                "",
            ]
        )

    paragraphs.extend(
        [
            "## Week5 Overall Conclusion",
            "",
            "week5 的总体贡献不是增加了若干孤立脚本，而是完成了研究对象的正式重构。项目已经从单一 pointwise uncertainty pipeline 升级成多决策任务统一框架：pointwise 被保留为诊断层，pairwise preference 被落实为偏好机制层，candidate ranking 被落实为主决策层。三层任务现在已经在统一数据来源、统一配置风格、统一推理接口和统一评估口径下形成了可运行、可比较、可聚合的系统结构。",
            "",
            "这意味着 version pony 到本周结束时，第一次真正进入执行态研究系统：方法部分已经具备任务定义，实验部分已经具备输入输出与评价口径，项目报告已经具备阶段性总表，week6 则可以在这个底座上继续做 uncertainty 跨任务迁移，而不必再回到任务接口和工程边界层面反复返工。",
        ]
    )

    return "\n".join(paragraphs).strip() + "\n"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {output_root}")

    if args.exp_names:
        exp_dirs = [output_root / exp_name for exp_name in args.exp_names]
    else:
        exp_dirs = discover_experiment_dirs(output_root)

    if not exp_dirs:
        raise ValueError(f"No multitask experiment folders with required files were found under {output_root}")

    rows = [aggregate_experiment(exp_dir) for exp_dir in exp_dirs]
    summary_df = pd.DataFrame(rows)
    summary_df = filter_summary(summary_df, args.domain, args.model)
    if summary_df.empty:
        raise ValueError("No multitask rows remain after applying the requested filters.")

    task_order = {
        "pointwise_yesno": 0,
        "pairwise_preference": 1,
        "candidate_ranking": 2,
    }
    summary_df["task_sort_key"] = summary_df["task"].map(task_order).fillna(999)
    summary_df = summary_df.sort_values(["domain", "model", "task_sort_key"]).drop(columns=["task_sort_key"]).reset_index(drop=True)

    summary_dir = output_root / "summary"
    ensure_dir(summary_dir)

    csv_output_path = summary_dir / "week5_multitask_summary.csv"
    md_output_path = summary_dir / "week5_multitask_summary.md"

    summary_df.to_csv(csv_output_path, index=False)
    md_output_path.write_text(build_markdown_summary(summary_df), encoding="utf-8")

    print(f"Aggregated {len(summary_df)} multitask rows.")
    print(f"Saved week5 multitask summary to: {csv_output_path}")
    print(f"Saved week5 multitask markdown summary to: {md_output_path}")


if __name__ == "__main__":
    main()
