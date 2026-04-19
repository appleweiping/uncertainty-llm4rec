from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OLD_COVERAGE_PATH = Path("outputs/summary/week6_day3_pairwise_coverage_compare.csv")
PLAIN_COMPARE_PATH = Path("outputs/summary/beauty_deepseek_pairwise_coverage_plain_to_rank_compare.csv")
WEIGHTED_COMPARE_PATH = Path("outputs/summary/beauty_deepseek_pairwise_coverage_to_rank_compare.csv")
OUTPUT_TABLE_PATH = Path("outputs/summary/week6_final_pairwise_coverage_upgrade.csv")
PLOT_SOURCE_PATH = Path("outputs/summary/tables/part5/part5_pairwise_coverage_upgraded_plot_source.csv")
FIGURE_PATH = Path("outputs/summary/figures/part5/part5_pairwise_coverage_upgraded.png")
NOTES_PATH = Path("outputs/summary/week6_final_pairwise_coverage_upgrade_notes.md")


def _read_method_row(path: Path, method_contains: str | None = None, method_equals: str | None = None) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing pairwise compare file: {path}")
    df = pd.read_csv(path)
    if method_equals is not None:
        match = df[df["method"].astype(str) == method_equals]
    elif method_contains is not None:
        match = df[df["method"].astype(str).str.contains(method_contains, na=False)]
    else:
        match = df
    if match.empty:
        raise ValueError(f"No matching method row found in {path}")
    return match.iloc[0]


def build_upgrade_table() -> pd.DataFrame:
    direct = _read_method_row(WEIGHTED_COMPARE_PATH, method_equals="direct_candidate_ranking")
    weighted = _read_method_row(WEIGHTED_COMPARE_PATH, method_contains="weighted_win_count")
    plain = _read_method_row(PLAIN_COMPARE_PATH, method_contains="plain_win_count")

    total_events = int(weighted.get("total_ranking_event_count", direct.get("sample_count", 100)))
    supported_fraction = float(weighted.get("pairwise_supported_event_fraction", 0.0))
    supported_events = int(round(total_events * supported_fraction))

    row = {
        "domain": "beauty",
        "model": "deepseek",
        "task": "pairwise_to_rank",
        "total_ranking_events": total_events,
        "supported_ranking_events": supported_events,
        "supported_event_fraction": supported_fraction,
        "pairwise_pair_coverage_rate": float(weighted.get("pairwise_pair_coverage_rate", 0.0)),
        "overlap_NDCG@10": float(weighted.get("NDCG@10")),
        "overlap_MRR": float(weighted.get("MRR")),
        "expanded_NDCG@10": float(weighted.get("NDCG@10")),
        "expanded_MRR": float(weighted.get("MRR")),
        "plain_NDCG@10": float(plain.get("NDCG@10")),
        "plain_MRR": float(plain.get("MRR")),
        "weighted_NDCG@10": float(weighted.get("NDCG@10")),
        "weighted_MRR": float(weighted.get("MRR")),
        "direct_reference_NDCG@10": float(direct.get("NDCG@10")),
        "direct_reference_MRR": float(direct.get("MRR")),
        "plain_vs_weighted_gap": float(weighted.get("NDCG@10")) - float(plain.get("NDCG@10")),
        "notes": "Coverage-balanced pair generation raises supported ranking events under the same compact 100-sample setting; aggregation formula remains the same weighted win-count mechanism, with plain win-count used only as the non-uncertainty baseline.",
    }
    return pd.DataFrame([row])


def build_plot_source(upgrade_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if OLD_COVERAGE_PATH.exists():
        old_df = pd.read_csv(OLD_COVERAGE_PATH)
        if not old_df.empty:
            old_row = old_df.iloc[0]
            rows.append(
                {
                    "panel": "coverage",
                    "setting": "before_coverage_upgrade",
                    "metric": "supported_event_fraction",
                    "value": float(old_row.get("supported_event_fraction", old_row.get("pairwise_supported_event_fraction", 0.0))),
                }
            )

    row = upgrade_df.iloc[0]
    rows.extend(
        [
            {
                "panel": "coverage",
                "setting": "after_coverage_upgrade",
                "metric": "supported_event_fraction",
                "value": float(row["supported_event_fraction"]),
            },
            {
                "panel": "ranking",
                "setting": "direct_reference",
                "metric": "NDCG@10",
                "value": float(row["direct_reference_NDCG@10"]),
            },
            {
                "panel": "ranking",
                "setting": "plain_win_count",
                "metric": "NDCG@10",
                "value": float(row["plain_NDCG@10"]),
            },
            {
                "panel": "ranking",
                "setting": "weighted_win_count",
                "metric": "NDCG@10",
                "value": float(row["weighted_NDCG@10"]),
            },
            {
                "panel": "ranking",
                "setting": "direct_reference",
                "metric": "MRR",
                "value": float(row["direct_reference_MRR"]),
            },
            {
                "panel": "ranking",
                "setting": "plain_win_count",
                "metric": "MRR",
                "value": float(row["plain_MRR"]),
            },
            {
                "panel": "ranking",
                "setting": "weighted_win_count",
                "metric": "MRR",
                "value": float(row["weighted_MRR"]),
            },
        ]
    )
    return pd.DataFrame(rows)


def plot_upgrade(plot_df: pd.DataFrame) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    coverage_df = plot_df[plot_df["panel"] == "coverage"]
    axes[0].bar(coverage_df["setting"], coverage_df["value"], color=["#9aa6b2", "#2f6f73"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Pairwise supported event fraction")
    axes[0].set_ylabel("fraction")
    axes[0].tick_params(axis="x", rotation=20)

    ranking_df = plot_df[plot_df["panel"] == "ranking"]
    ranking_pivot = ranking_df.pivot(index="setting", columns="metric", values="value")
    ranking_pivot.loc[["direct_reference", "plain_win_count", "weighted_win_count"]].plot(
        kind="bar",
        ax=axes[1],
        color=["#c96f53", "#2f6f73"],
    )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Overlap ranking quality after coverage upgrade")
    axes[1].set_ylabel("score")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def write_notes(upgrade_df: pd.DataFrame) -> None:
    row = upgrade_df.iloc[0]
    lines = [
        "# Week6 Final Pairwise Coverage Upgrade Notes",
        "",
        "本轮 pairwise 修复的对象是事件支撑率，而不是 aggregation family。原先 pairwise-to-rank 的积极信号只能在较小 overlap 子集上讨论，主要风险是 supported ranking events 太少；本轮通过 coverage-balanced pair generation，在不扩大真实 API 总样本到 1000、不改变 Part5 三层任务定义的前提下，让 compact 100-sample Beauty setting 中更多 ranking event 获得 pairwise 支持。",
        "",
        f"升级后，Beauty + DeepSeek pairwise-to-rank 的 supported ranking events 为 {int(row['supported_ranking_events'])}/{int(row['total_ranking_events'])}，supported_event_fraction 为 {float(row['supported_event_fraction']):.4f}，pairwise_pair_coverage_rate 为 {float(row['pairwise_pair_coverage_rate']):.4f}。这说明机制层覆盖已经从原先的低事件支撑，提升到可以在完整 compact event 范围中讨论；但每个 event 内部仍然只覆盖候选对空间的一小部分，因此不能写成 pairwise 完全替代 listwise ranking。",
        "",
        f"在统一 compact scope 下，direct reference 的 NDCG@10/MRR 为 {float(row['direct_reference_NDCG@10']):.4f}/{float(row['direct_reference_MRR']):.4f}，plain win-count 为 {float(row['plain_NDCG@10']):.4f}/{float(row['plain_MRR']):.4f}，weighted win-count 为 {float(row['weighted_NDCG@10']):.4f}/{float(row['weighted_MRR']):.4f}。plain_vs_weighted_gap 按 NDCG@10 记为 {float(row['plain_vs_weighted_gap']):.4f}，用于说明 uncertainty reliability weight 在当前覆盖修复后的边际作用。",
        "",
        "阶段性边界是：pairwise 当前已经从低覆盖观察线升级为更扎实的机制层证据线，但仍不升级为 candidate ranking 主决策替代。后续 week7 可以继续扩大执行层和覆盖，但 Part5 compact evidence 中应该诚实表述为机制层增强与边界清晰化。",
    ]
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    upgrade_df = build_upgrade_table()
    OUTPUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    upgrade_df.to_csv(OUTPUT_TABLE_PATH, index=False)
    plot_df = build_plot_source(upgrade_df)
    plot_df.to_csv(PLOT_SOURCE_PATH, index=False)
    plot_upgrade(plot_df)
    write_notes(upgrade_df)
    print(f"Saved pairwise coverage upgrade table to: {OUTPUT_TABLE_PATH}")
    print(f"Saved pairwise coverage upgraded plot source to: {PLOT_SOURCE_PATH}")
    print(f"Saved pairwise coverage upgraded figure to: {FIGURE_PATH}")
    print(f"Saved pairwise coverage upgrade notes to: {NOTES_PATH}")


if __name__ == "__main__":
    main()
