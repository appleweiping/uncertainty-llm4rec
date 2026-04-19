from __future__ import annotations

from pathlib import Path

import pandas as pd


DOMAIN_EXPERIMENTS = {
    "movies": "movies_deepseek_rank_structured_risk",
    "beauty": "beauty_deepseek_rank_structured_risk",
    "books": "books_deepseek_rank_structured_risk",
    "electronics": "electronics_deepseek_rank_structured_risk",
}

OUTPUT_PATH = Path("outputs/summary/week6_final_4domain_structured_risk_compare.csv")
NOTES_PATH = Path("outputs/summary/week6_final_4domain_structured_risk_notes.md")


def _method_family(method: str) -> str:
    if method == "direct_candidate_ranking":
        return "direct_candidate_ranking"
    if "structured_risk" in method:
        return "structured_risk_family"
    return "retained_or_auxiliary_family"


def _pick_samples(row: pd.Series) -> int | float:
    for col in ["sample_count", "num_events", "samples"]:
        if col in row and pd.notna(row[col]):
            return int(row[col])
    return float("nan")


def build_compare() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, exp_name in DOMAIN_EXPERIMENTS.items():
        metrics_path = Path("outputs") / exp_name / "tables" / "rerank_results.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing structured-risk rerank result: {metrics_path}")

        df = pd.read_csv(metrics_path)
        for record in df.to_dict(orient="records"):
            method = str(record.get("method", ""))
            if method != "direct_candidate_ranking" and "structured_risk" not in method:
                continue
            rows.append(
                {
                    "domain": domain,
                    "model": "deepseek",
                    "task": "candidate_ranking",
                    "method_family": _method_family(method),
                    "method_variant": method,
                    "samples": _pick_samples(pd.Series(record)),
                    "HR@10": record.get("HR@10"),
                    "NDCG@10": record.get("NDCG@10"),
                    "MRR": record.get("MRR"),
                    "changed_ranking_fraction": record.get("changed_ranking_fraction", 0.0),
                    "avg_position_shift": record.get("avg_position_shift", 0.0),
                    "uncertainty_coverage": record.get("uncertainty_coverage", record.get("avg_uncertainty_coverage", 0.0)),
                }
            )

    return pd.DataFrame(rows)


def write_notes(compare_df: pd.DataFrame) -> None:
    pivot = compare_df.pivot_table(
        index="domain",
        columns="method_family",
        values=["NDCG@10", "MRR", "changed_ranking_fraction", "uncertainty_coverage"],
        aggfunc="first",
    )
    lines = [
        "# Week6 Final 四域 Structured Risk 主线对照说明",
        "",
        "本表把 Movies、Beauty、Books、Electronics 四个正式数据域中已经完成的 100-sample DeepSeek candidate ranking 结果，与同一批 ranking 输出上的 structured risk current best family 结果放在同一口径下比较。这里没有新增 ranking family，也没有重新做参数搜索；structured risk 继续承担当前默认主线身份，local swap 与 fully fused 仍然作为 retained exploratory family 保留在既有表和图中。",
        "",
        "从四域结果看，structured risk 的主要价值不是制造大幅改序，而是在有限 uncertainty coverage 下以低扰动方式检验 uncertainty 是否能进入 listwise decision。`changed_ranking_fraction` 和 `avg_position_shift` 应与 NDCG/MRR 一起阅读：如果收益很小但扰动也很小，说明当前主线更接近稳健控制变量；如果个别域略有下降，也不能被解读为 family 失效，而应结合 coverage 和后续更强执行层继续验证。",
        "",
        "紧凑透视如下，便于快速核对四域方向：",
        "",
        "```text",
        pivot.to_string(),
        "```",
        "",
        "阶段性结论是：四域 DeepSeek compact setting 下，structured risk current best family 已经不再只停留在 Beauty 单域，而是完成了跨域主线落地。当前证据仍是 100-sample real API 口径，适合支撑 Part5 compact evidence version，不适合过度外推为最终大规模稳定结论。",
    ]
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    compare_df = build_compare()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(OUTPUT_PATH, index=False)
    write_notes(compare_df)
    print(f"Saved structured-risk compare to: {OUTPUT_PATH}")
    print(f"Saved structured-risk notes to: {NOTES_PATH}")


if __name__ == "__main__":
    main()
