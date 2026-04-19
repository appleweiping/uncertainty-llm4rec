from __future__ import annotations

from pathlib import Path

import pandas as pd


PART5_FINAL_PATH = Path("outputs/summary/part5_multitask_final_results.csv")
FOUR_DOMAIN_PATH = Path("outputs/summary/week6_magic7_4domain_deepseek_compare.csv")
STRUCTURED_RISK_PATH = Path("outputs/summary/week6_final_4domain_structured_risk_compare.csv")
PAIRWISE_UPGRADE_PATH = Path("outputs/summary/week6_final_pairwise_coverage_upgrade.csv")
BASELINE_COMPARE_PATH = Path("outputs/summary/week6_final_literature_baseline_compare.csv")

SINGLE_DOMAIN_OUTPUT = Path("outputs/summary/part5_single_domain_main_table.csv")
FOUR_DOMAIN_OUTPUT = Path("outputs/summary/part5_4domain_main_table.csv")
PAIRWISE_OUTPUT = Path("outputs/summary/part5_pairwise_boundary_table.csv")
MD_OUTPUT = Path("outputs/summary/part5_consolidated_tables.md")


def _safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_single_domain_table() -> pd.DataFrame:
    final_df = _safe_read(PART5_FINAL_PATH)
    baseline_df = _safe_read(BASELINE_COMPARE_PATH)
    rows: list[dict[str, object]] = []

    if not final_df.empty:
        for record in final_df.to_dict(orient="records"):
            task = str(record.get("task", ""))
            method_variant = str(record.get("method_variant", record.get("method", "")))
            role = "current_line"
            if "baseline" in str(record.get("method_family", "")).lower() or "direct" in method_variant:
                role = "same_task_baseline"
            if "local" in method_variant or "fused" in method_variant:
                role = "retained_exploratory_line"
            if "pairwise" in task:
                role = "mechanism_layer_evidence"
            rows.append(
                {
                    "domain": record.get("domain", "beauty"),
                    "model": record.get("model", "qwen_or_deepseek"),
                    "task": task,
                    "method_family": record.get("method_family", ""),
                    "method_variant": method_variant,
                    "current_role": role,
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "ECE": record.get("ECE"),
                    "Brier": record.get("Brier"),
                    "coverage": record.get("coverage", record.get("uncertainty_coverage")),
                    "notes": record.get("notes", ""),
                }
            )

    if not baseline_df.empty:
        for record in baseline_df.to_dict(orient="records"):
            rows.append(
                {
                    "domain": record.get("domain", "beauty"),
                    "model": "deepseek",
                    "task": record.get("task", "candidate_ranking"),
                    "method_family": record.get("baseline_group"),
                    "method_variant": record.get("method_name"),
                    "current_role": "same_task_or_literature_aligned_baseline",
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "ECE": "",
                    "Brier": "",
                    "coverage": "",
                    "notes": record.get("notes", ""),
                }
            )

    return pd.DataFrame(rows)


def build_four_domain_table() -> pd.DataFrame:
    cross_df = _safe_read(FOUR_DOMAIN_PATH)
    sr_df = _safe_read(STRUCTURED_RISK_PATH)
    baseline_df = _safe_read(BASELINE_COMPARE_PATH)
    rows: list[dict[str, object]] = []

    if not cross_df.empty:
        for record in cross_df.to_dict(orient="records"):
            rows.append(
                {
                    "domain": record.get("domain"),
                    "model": record.get("model", "deepseek"),
                    "task": record.get("task"),
                    "method_family": "pointwise_diagnosis" if record.get("task") == "pointwise_yesno" else "direct_candidate_ranking",
                    "method_variant": record.get("exp_name", "deepseek_compact"),
                    "samples": record.get("samples"),
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "ECE": record.get("ECE"),
                    "Brier": record.get("Brier"),
                    "parse_success_rate": record.get("parse_success_rate"),
                    "current_role": "diagnosis_layer" if record.get("task") == "pointwise_yesno" else "direct_reference",
                }
            )

    if not sr_df.empty:
        for record in sr_df.to_dict(orient="records"):
            rows.append(
                {
                    "domain": record.get("domain"),
                    "model": record.get("model", "deepseek"),
                    "task": record.get("task"),
                    "method_family": record.get("method_family"),
                    "method_variant": record.get("method_variant"),
                    "samples": record.get("samples"),
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "ECE": "",
                    "Brier": "",
                    "parse_success_rate": "",
                    "current_role": "current_best_family" if record.get("method_family") == "structured_risk_family" else "direct_reference",
                }
            )

    if not baseline_df.empty:
        for record in baseline_df.to_dict(orient="records"):
            rows.append(
                {
                    "domain": record.get("domain"),
                    "model": "deepseek",
                    "task": record.get("task"),
                    "method_family": record.get("baseline_group"),
                    "method_variant": record.get("method_name"),
                    "samples": record.get("samples"),
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "ECE": "",
                    "Brier": "",
                    "parse_success_rate": "",
                    "current_role": "baseline_reference",
                }
            )

    return pd.DataFrame(rows)


def build_pairwise_boundary_table() -> pd.DataFrame:
    upgrade_df = _safe_read(PAIRWISE_UPGRADE_PATH)
    if upgrade_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for record in upgrade_df.to_dict(orient="records"):
        rows.append(
            {
                "domain": record.get("domain", "beauty"),
                "model": record.get("model", "deepseek"),
                "task": record.get("task", "pairwise_to_rank"),
                "supported_event_fraction": record.get("supported_event_fraction"),
                "pairwise_pair_coverage_rate": record.get("pairwise_pair_coverage_rate"),
                "overlap_NDCG@10": record.get("overlap_NDCG@10"),
                "overlap_MRR": record.get("overlap_MRR"),
                "expanded_NDCG@10": record.get("expanded_NDCG@10"),
                "expanded_MRR": record.get("expanded_MRR"),
                "plain_vs_weighted_gap": record.get("plain_vs_weighted_gap"),
                "current_role": "mechanism_layer_evidence_not_main_replacement",
                "notes": record.get("notes"),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(single_df: pd.DataFrame, four_df: pd.DataFrame, pairwise_df: pd.DataFrame) -> None:
    lines = [
        "# Part5 Consolidated Tables",
        "",
        "本文件把 week5/week6 以及 magic 补充阶段的分散结果压成论文写作可以直接引用的三张主表骨架。它不新增实验，也不新增 ranking family，而是把 single-domain Part5 主结果、四域 DeepSeek compact matrix、pairwise coverage boundary 汇总到同一层级。",
        "",
        "第一张表 `outputs/summary/part5_single_domain_main_table.csv` 面向 Part5 方法主叙事，集中记录 pointwise、candidate ranking、pairwise 三层任务的 current line、same-task baseline、retained exploratory line 和机制层证据。",
        "",
        "第二张表 `outputs/summary/part5_4domain_main_table.csv` 面向跨域紧凑证据，集中记录 Movies、Beauty、Books、Electronics 四域的 pointwise diagnosis、direct candidate ranking、structured risk current best family 和主要 baseline。",
        "",
        "第三张表 `outputs/summary/part5_pairwise_boundary_table.csv` 面向 pairwise 边界叙事，集中记录 supported event fraction、pair coverage、overlap/expanded 结果和 plain-vs-weighted gap。它的作用是把 pairwise 写成机制层证据，而不是过早写成主决策替代。",
        "",
        f"当前 single-domain 主表行数：{len(single_df)}；四域主表行数：{len(four_df)}；pairwise boundary 主表行数：{len(pairwise_df)}。",
    ]
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    single_df = build_single_domain_table()
    four_df = build_four_domain_table()
    pairwise_df = build_pairwise_boundary_table()

    SINGLE_DOMAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    single_df.to_csv(SINGLE_DOMAIN_OUTPUT, index=False)
    four_df.to_csv(FOUR_DOMAIN_OUTPUT, index=False)
    pairwise_df.to_csv(PAIRWISE_OUTPUT, index=False)
    write_markdown(single_df, four_df, pairwise_df)

    print(f"Saved single-domain main table to: {SINGLE_DOMAIN_OUTPUT}")
    print(f"Saved four-domain main table to: {FOUR_DOMAIN_OUTPUT}")
    print(f"Saved pairwise boundary table to: {PAIRWISE_OUTPUT}")
    print(f"Saved consolidated table notes to: {MD_OUTPUT}")


if __name__ == "__main__":
    main()
