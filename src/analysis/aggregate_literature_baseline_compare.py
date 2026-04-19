from __future__ import annotations

from pathlib import Path

import pandas as pd


BASELINE_STATUS_PATH = Path("outputs/summary/week6_final_literature_baseline_status.csv")
DIRECT_RANK_PATH = Path("outputs/beauty_deepseek_rank/tables/ranking_metrics.csv")
STRUCTURED_RISK_PATH = Path("outputs/beauty_deepseek_rank_structured_risk/tables/rerank_results.csv")
OUTPUT_PATH = Path("outputs/summary/week6_final_literature_baseline_compare.csv")
NOTES_PATH = Path("outputs/summary/week6_final_literature_baseline_notes.md")


BASELINE_GROUPS = {
    "candidate_order_rank": "same_task_order_baseline",
    "popularity_prior_rank": "literature_aligned_popularity_prior",
    "longtail_prior_rank": "exposure_oriented_longtail_prior",
}


def _samples(row: pd.Series) -> int | float:
    for col in ["sample_count", "samples"]:
        if col in row and pd.notna(row[col]):
            return int(row[col])
    return float("nan")


def build_compare() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not BASELINE_STATUS_PATH.exists():
        raise FileNotFoundError(f"Missing literature baseline status: {BASELINE_STATUS_PATH}")

    baseline_df = pd.read_csv(BASELINE_STATUS_PATH)
    for record in baseline_df.to_dict(orient="records"):
        name = str(record.get("baseline_name"))
        rows.append(
            {
                "domain": "beauty",
                "task": "candidate_ranking",
                "baseline_group": BASELINE_GROUPS.get(name, "task_aligned_baseline"),
                "method_name": name,
                "samples": record.get("sample_count"),
                "HR@10": record.get("HR@10"),
                "NDCG@10": record.get("NDCG@10"),
                "MRR": record.get("MRR"),
                "notes": record.get("notes"),
            }
        )

    if DIRECT_RANK_PATH.exists():
        direct_row = pd.read_csv(DIRECT_RANK_PATH).iloc[0]
        rows.append(
            {
                "domain": "beauty",
                "task": "candidate_ranking",
                "baseline_group": "direct_llm_candidate_ranking",
                "method_name": "direct_candidate_ranking",
                "samples": _samples(direct_row),
                "HR@10": direct_row.get("HR@10"),
                "NDCG@10": direct_row.get("NDCG@10"),
                "MRR": direct_row.get("MRR"),
                "notes": "Direct DeepSeek candidate ranking over the same compact candidate set; used as the main same-task reference.",
            }
        )

    if STRUCTURED_RISK_PATH.exists():
        sr_df = pd.read_csv(STRUCTURED_RISK_PATH)
        sr_rows = sr_df[sr_df["method"].astype(str).str.contains("structured_risk", na=False)]
        if not sr_rows.empty:
            sr_row = sr_rows.iloc[0]
            rows.append(
                {
                    "domain": "beauty",
                    "task": "candidate_ranking",
                    "baseline_group": "uncertainty_aware_current_best_family",
                    "method_name": "structured_risk_current_best",
                    "samples": _samples(sr_row),
                    "HR@10": sr_row.get("HR@10"),
                    "NDCG@10": sr_row.get("NDCG@10"),
                    "MRR": sr_row.get("MRR"),
                    "notes": "Current best uncertainty-aware ranking family retained as the Part5 default method line.",
                }
            )

    return pd.DataFrame(rows)


def write_notes(compare_df: pd.DataFrame) -> None:
    compact = compare_df[["baseline_group", "method_name", "NDCG@10", "MRR"]].to_string(index=False)
    lines = [
        "# Week6 Final Literature-Aligned Baseline Notes",
        "",
        "本轮 baseline 扩充没有追求数量堆砌，而是把 candidate ranking 层最需要的防守性参照补到统一 schema 中。`candidate_order_rank` 提供不使用模型重排的同任务顺序参照，`popularity_prior_rank` 对齐推荐文献中常见的 popularity prior，`longtail_prior_rank` 用于观察 exposure-oriented ranking heuristic 在同一候选集上的表现；三者都运行在同一 split、同一 candidate set、同一 HR@10/NDCG@10/MRR 口径下。",
        "",
        "这个 compare 表同时放入 direct DeepSeek candidate ranking 和 structured risk current best family。这样论文叙事里可以清楚区分三类对象：任务对齐的 non-uncertainty baseline、直接 LLM ranking reference、以及当前 uncertainty-aware 主线。local swap 与 fully fused 仍保留在 Part5 family compare 中，但不作为本轮默认 baseline 竞争对象。",
        "",
        "紧凑结果如下：",
        "",
        "```text",
        compact,
        "```",
        "",
        "阶段性结论是：Part5 的 literature-aligned baseline 已从“入口存在”推进到“有初步防守力的 baseline 组”。当前仍不是最终强 baseline 全量矩阵，但已经足以支撑 compact evidence 版本中对同任务、同候选集、同指标的基本防守。",
    ]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    compare_df = build_compare()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(OUTPUT_PATH, index=False)
    write_notes(compare_df)
    print(f"Saved literature baseline compare to: {OUTPUT_PATH}")
    print(f"Saved literature baseline notes to: {NOTES_PATH}")


if __name__ == "__main__":
    main()
