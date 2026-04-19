from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def build_pairwise_coverage_plot_source(*, output_root: str | Path = "outputs") -> pd.DataFrame:
    output_root = Path(output_root)
    coverage_df = _read_csv(output_root / "summary" / "week6_day3_pairwise_coverage_compare.csv")
    final_df = _read_csv(output_root / "summary" / "part5_multitask_final_results.csv")

    pairwise_final = final_df[final_df["task"].astype(str) == "pairwise_to_rank"].copy()
    if pairwise_final.empty:
        raise ValueError("No pairwise_to_rank rows found in part5_multitask_final_results.csv")

    rows: list[dict[str, object]] = []
    for row in coverage_df.to_dict("records"):
        estimator = str(row.get("estimator", "unknown"))
        rows.append(
            {
                "plot_name": "coverage",
                "domain": row.get("domain"),
                "model": row.get("model"),
                "scope": "event_support",
                "estimator": estimator,
                "method_label": "supported_events",
                "metric": "supported_event_count",
                "value": row.get("supported_event_count"),
                "denominator": row.get("total_ranking_event_count"),
                "supported_event_fraction": row.get("pairwise_supported_event_fraction"),
                "pairwise_pair_coverage_rate": row.get("pairwise_pair_coverage_rate"),
            }
        )
        rows.append(
            {
                "plot_name": "coverage",
                "domain": row.get("domain"),
                "model": row.get("model"),
                "scope": "pair_support",
                "estimator": estimator,
                "method_label": "covered_pairs",
                "metric": "pairwise_pair_coverage_rate",
                "value": row.get("pairwise_pair_coverage_rate"),
                "denominator": 1.0,
                "supported_event_fraction": row.get("pairwise_supported_event_fraction"),
                "pairwise_pair_coverage_rate": row.get("pairwise_pair_coverage_rate"),
            }
        )

    label_map = {
        "direct_overlap_reference": "Direct reference",
        "plain_win_count_overlap": "Plain aggregation",
        "weighted_win_count": "Weighted aggregation",
        "direct_expanded_reference": "Direct reference",
        "plain_win_count_expanded": "Plain aggregation",
        "weighted_win_count_expanded": "Weighted aggregation",
    }
    pairwise_final = _to_numeric(
        pairwise_final,
        ["NDCG@10", "MRR", "pairwise_supported_event_fraction", "pairwise_pair_coverage_rate"],
    )
    for row in pairwise_final.to_dict("records"):
        scope = str(row.get("evaluation_scope", "unknown"))
        for metric in ["NDCG@10", "MRR"]:
            rows.append(
                {
                    "plot_name": "scope_compare",
                    "domain": row.get("domain"),
                    "model": row.get("model"),
                    "scope": scope,
                    "estimator": row.get("estimator"),
                    "method_label": label_map.get(str(row.get("method_variant")), str(row.get("method_variant"))),
                    "metric": metric,
                    "value": row.get(metric),
                    "denominator": None,
                    "supported_event_fraction": row.get("pairwise_supported_event_fraction"),
                    "pairwise_pair_coverage_rate": row.get("pairwise_pair_coverage_rate"),
                }
            )

    return pd.DataFrame(rows)


def plot_pairwise_coverage(source_df: pd.DataFrame, output_path: Path) -> None:
    source_df = _to_numeric(source_df, ["value", "denominator", "supported_event_fraction", "pairwise_pair_coverage_rate"])
    coverage_df = source_df[source_df["plot_name"] == "coverage"].copy()
    best_row = coverage_df[coverage_df["method_label"] == "supported_events"].head(1)
    if best_row.empty:
        raise ValueError("No supported event coverage row available.")
    supported = float(best_row["value"].iloc[0])
    total = float(best_row["denominator"].iloc[0])
    unsupported = max(total - supported, 0)
    pair_rate = float(best_row["pairwise_pair_coverage_rate"].iloc[0])

    plt.figure(figsize=(7.2, 4.8))
    plt.bar(["Supported events", "Unsupported events"], [supported, unsupported], color=["#2a9d8f", "#c9c9c9"])
    plt.ylabel("Ranking event count")
    plt.title(f"Pairwise Coverage Boundary ({int(supported)}/{int(total)} events, pair coverage={pair_rate:.2f})")
    for x, value in zip(["Supported events", "Unsupported events"], [supported, unsupported]):
        plt.text(x, value + 0.5, f"{int(value)}", ha="center", fontsize=9)
    plt.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_pairwise_scope_compare(source_df: pd.DataFrame, output_path: Path) -> None:
    compare_df = source_df[source_df["plot_name"] == "scope_compare"].copy()
    compare_df = _to_numeric(compare_df, ["value"]).dropna(subset=["value"])
    if compare_df.empty:
        raise ValueError("No pairwise scope compare rows available.")

    scope_label = {
        "pairwise_event_overlap_subset": "Overlap",
        "expanded_with_direct_fallback": "Expanded",
    }
    compare_df["scope_label"] = compare_df["scope"].map(scope_label).fillna(compare_df["scope"])
    methods = ["Direct reference", "Plain aggregation", "Weighted aggregation"]
    scopes = ["Overlap", "Expanded"]
    metrics = ["NDCG@10", "MRR"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    colors = {"Direct reference": "#8d99ae", "Plain aggregation": "#457b9d", "Weighted aggregation": "#e76f51"}
    width = 0.23
    x_positions = list(range(len(scopes)))

    for ax, metric in zip(axes, metrics):
        metric_df = compare_df[compare_df["metric"] == metric]
        for idx, method in enumerate(methods):
            values = []
            for scope in scopes:
                subset = metric_df[(metric_df["scope_label"] == scope) & (metric_df["method_label"] == method)]
                values.append(float(subset["value"].iloc[0]) if not subset.empty else float("nan"))
            offsets = [x + (idx - 1) * width for x in x_positions]
            ax.bar(offsets, values, width=width, label=method, color=colors[method], alpha=0.9)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(scopes)
        ax.set_title(metric)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Pairwise-To-Rank Scope Compare")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def build_pairwise_coverage_summary(*, output_root: str | Path = "outputs") -> dict[str, Path]:
    output_root = Path(output_root)
    figure_dir = output_root / "summary" / "figures" / "part5"
    table_dir = output_root / "summary" / "tables" / "part5"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    source_df = build_pairwise_coverage_plot_source(output_root=output_root)
    source_path = table_dir / "part5_pairwise_coverage_plot_source.csv"
    source_df.to_csv(source_path, index=False)

    coverage_path = figure_dir / "part5_pairwise_coverage.png"
    scope_path = figure_dir / "part5_pairwise_scope_compare.png"
    plot_pairwise_coverage(source_df, coverage_path)
    plot_pairwise_scope_compare(source_df, scope_path)

    return {
        "source": source_path,
        "coverage": coverage_path,
        "scope_compare": scope_path,
    }

