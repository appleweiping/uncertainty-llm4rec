from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import ensure_dir


POINTWISE_EXP_DEFAULT = "beauty_qwen_pointwise"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _prediction_frame(prediction_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(_read_jsonl(prediction_path))
    if df.empty:
        return df

    recommend = df.get("recommend", pd.Series([""] * len(df))).astype(str).str.lower()
    labels = pd.to_numeric(df.get("label", 0), errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df.get("confidence", 0.0), errors="coerce").clip(0, 1)
    df["pred_label"] = recommend.eq("yes").astype(int)
    df["is_correct"] = (df["pred_label"] == labels).astype(int)
    if "target_popularity_group" not in df.columns:
        df["target_popularity_group"] = "unknown"
    return df


def build_pointwise_plot_source(
    *,
    output_root: str | Path = "outputs",
    pointwise_exp_name: str = POINTWISE_EXP_DEFAULT,
) -> pd.DataFrame:
    output_root = Path(output_root)
    tables_dir = output_root / pointwise_exp_name / "tables"
    predictions_path = output_root / pointwise_exp_name / "predictions" / "test_raw.jsonl"

    reliability_df = _read_csv(tables_dir / "reliability_bins.csv")
    popularity_df = _read_csv(tables_dir / "popularity_group_stats.csv")
    prediction_df = _prediction_frame(predictions_path)

    rows: list[dict[str, Any]] = []
    for row in reliability_df.to_dict("records"):
        rows.append(
            {
                "plot_name": "reliability_diagram",
                "source_type": "reliability_bin",
                "group": row.get("bin_center"),
                "x_value": row.get("avg_confidence"),
                "y_value": row.get("accuracy"),
                "count": row.get("count"),
                "metric_name": "accuracy_vs_confidence",
            }
        )

    for row in popularity_df.to_dict("records"):
        rows.append(
            {
                "plot_name": "popularity_confidence_distribution",
                "source_type": "popularity_group_summary",
                "group": row.get("target_popularity_group"),
                "x_value": row.get("target_popularity_group"),
                "y_value": row.get("avg_confidence"),
                "count": row.get("num_samples"),
                "metric_name": "avg_confidence",
            }
        )

    for row in prediction_df.to_dict("records"):
        rows.append(
            {
                "plot_name": "confidence_histogram",
                "source_type": "prediction",
                "group": "correct" if int(row.get("is_correct", 0)) == 1 else "wrong",
                "x_value": row.get("confidence"),
                "y_value": row.get("is_correct"),
                "count": 1,
                "metric_name": "confidence",
            }
        )

    return pd.DataFrame(rows)


def build_family_compare_plot_source(*, output_root: str | Path = "outputs") -> pd.DataFrame:
    output_root = Path(output_root)
    final_df = _read_csv(output_root / "summary" / "part5_multitask_final_results.csv")
    rank_df = final_df[final_df["task"].astype(str) == "candidate_ranking"].copy()
    if rank_df.empty:
        raise ValueError("No candidate_ranking rows found in part5_multitask_final_results.csv")

    label_map = {
        "direct_candidate_ranking": "Direct ranking",
        "structured_risk_family": "Structured risk",
        "local_margin_swap_family": "Local swap",
        "structured_risk_plus_local_swap_family": "Structured + local swap",
    }
    rank_df["family_label"] = rank_df["method_family"].map(label_map).fillna(rank_df["method_family"])
    rank_df["family_role"] = rank_df["part5_final_role"]
    rank_df = _to_numeric(rank_df, ["NDCG@10", "MRR", "changed_ranking_fraction", "avg_position_shift"])
    return rank_df[
        [
            "domain",
            "model",
            "task",
            "family_role",
            "method_family",
            "method_variant",
            "family_label",
            "estimator",
            "NDCG@10",
            "MRR",
            "changed_ranking_fraction",
            "avg_position_shift",
            "notes",
        ]
    ].reset_index(drop=True)


def plot_reliability(source_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = source_df[source_df["plot_name"] == "reliability_diagram"].copy()
    plot_df = _to_numeric(plot_df, ["x_value", "y_value", "count"]).dropna(subset=["x_value", "y_value"])

    plt.figure(figsize=(5.8, 5.4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2, label="Perfect calibration")
    if not plot_df.empty:
        sizes = plot_df["count"].fillna(1).clip(lower=1) * 8
        plt.scatter(plot_df["x_value"], plot_df["y_value"], s=sizes, color="#2a6f97", alpha=0.82)
        plt.plot(plot_df["x_value"], plot_df["y_value"], color="#2a6f97", linewidth=1.8, label="Observed")
    plt.xlabel("Average confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Part5 Pointwise Reliability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_confidence_histogram(source_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = source_df[source_df["plot_name"] == "confidence_histogram"].copy()
    plot_df = _to_numeric(plot_df, ["x_value"])

    correct = plot_df[plot_df["group"] == "correct"]["x_value"].dropna()
    wrong = plot_df[plot_df["group"] == "wrong"]["x_value"].dropna()
    bins = [i / 10 for i in range(11)]

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(correct, bins=bins, alpha=0.72, color="#287271", label="Correct")
    plt.hist(wrong, bins=bins, alpha=0.68, color="#e76f51", label="Wrong")
    plt.xlabel("Verbalized confidence")
    plt.ylabel("Sample count")
    plt.title("Part5 Confidence Distribution")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_popularity_confidence(source_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = source_df[source_df["plot_name"] == "popularity_confidence_distribution"].copy()
    plot_df = _to_numeric(plot_df, ["y_value", "count"])
    order = ["head", "mid", "tail", "unknown"]
    plot_df["group"] = pd.Categorical(plot_df["group"].astype(str), categories=order, ordered=True)
    plot_df = plot_df.sort_values("group")

    plt.figure(figsize=(6.8, 4.6))
    plt.bar(plot_df["group"].astype(str), plot_df["y_value"], color="#8ab17d")
    for _, row in plot_df.iterrows():
        if pd.notna(row.get("y_value")):
            plt.text(str(row["group"]), float(row["y_value"]) + 0.01, f"n={int(row.get('count', 0))}", ha="center", fontsize=8)
    plt.xlabel("Popularity group")
    plt.ylabel("Average confidence")
    plt.title("Confidence by Popularity Group")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_family_compare(source_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = _to_numeric(source_df, ["NDCG@10", "MRR"]).copy()
    metric_df = plot_df.melt(
        id_vars=["family_label", "family_role", "method_family"],
        value_vars=["NDCG@10", "MRR"],
        var_name="metric",
        value_name="value",
    )
    metric_df = metric_df.dropna(subset=["value"])
    if metric_df.empty:
        raise ValueError("No NDCG@10/MRR values available for family compare plot.")

    families = list(dict.fromkeys(plot_df["family_label"].tolist()))
    x_positions = list(range(len(families)))
    width = 0.36
    colors = {"NDCG@10": "#264653", "MRR": "#f4a261"}

    plt.figure(figsize=(8.4, 5.0))
    for offset_idx, metric in enumerate(["NDCG@10", "MRR"]):
        values = []
        for family in families:
            subset = metric_df[(metric_df["family_label"] == family) & (metric_df["metric"] == metric)]
            values.append(float(subset["value"].iloc[0]) if not subset.empty else float("nan"))
        offsets = [x + (offset_idx - 0.5) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=metric, color=colors[metric], alpha=0.88)

    plt.xticks(x_positions, families, rotation=18, ha="right")
    plt.ylabel("Metric value")
    plt.title("Part5 Candidate Ranking Family Compare")
    plt.ylim(0, 0.75)
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def write_figure_pack_markdown(*, output_root: str | Path = "outputs") -> Path:
    output_root = Path(output_root)
    summary_path = output_root / "summary" / "part5_figure_pack.md"
    lines = [
        "# Part5 Figure Pack",
        "",
        "This figure pack turns the closed Part5 method loop into paper-facing visual evidence. It does not introduce new ranking families; it visualizes the existing pointwise diagnosis layer, candidate-ranking family compare, and pairwise coverage boundary.",
        "",
        "## Pointwise Diagnosis",
        "",
        "`reliability_diagram_part5.png` and `confidence_histogram_part5.png` answer whether verbalized confidence can be treated as a trustworthy uncertainty source. The current interpretation is conservative: pointwise remains the diagnosis and calibration layer, not the final recommendation task.",
        "",
        "`popularity_confidence_distribution_part5.png` records whether confidence behavior differs across popularity groups. It supports the paper narrative that uncertainty must be inspected together with exposure behavior rather than only through ranking metrics.",
        "",
        "## Candidate Ranking Family Compare",
        "",
        "`part5_family_compare.png` compares direct candidate ranking, the structured risk family, local margin swap, and the structured-risk-plus-local-swap retained family under NDCG@10 and MRR. The figure supports the current Part5 decision: structured risk remains the default current-best family, while local swap and fully fused variants stay as retained exploratory families.",
        "",
        "## Pairwise Coverage Boundary",
        "",
        "`part5_pairwise_coverage.png` and `part5_pairwise_scope_compare.png` should be read as mechanism evidence rather than a mainline replacement claim. Pairwise-to-rank shows positive signal, especially with calibrated reliability weighting, but current supported-event coverage is limited and must be disclosed.",
        "",
        "## Current Boundary",
        "",
        "The figure pack supports a restrained paper claim: pointwise diagnoses uncertainty, candidate ranking carries the main decision layer, and pairwise-to-rank offers coverage-limited mechanism evidence. No new formula family is introduced in this artifact step.",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def build_part5_figure_pack(
    *,
    output_root: str | Path = "outputs",
    pointwise_exp_name: str = POINTWISE_EXP_DEFAULT,
) -> dict[str, Path]:
    output_root = Path(output_root)
    figure_dir = output_root / "summary" / "figures" / "part5"
    table_dir = output_root / "summary" / "tables" / "part5"
    ensure_dir(figure_dir)
    ensure_dir(table_dir)

    pointwise_source = build_pointwise_plot_source(output_root=output_root, pointwise_exp_name=pointwise_exp_name)
    family_source = build_family_compare_plot_source(output_root=output_root)

    pointwise_source_path = table_dir / "part5_pointwise_plot_source.csv"
    family_source_path = table_dir / "part5_family_compare_plot_source.csv"
    pointwise_source.to_csv(pointwise_source_path, index=False)
    family_source.to_csv(family_source_path, index=False)

    reliability_path = figure_dir / "reliability_diagram_part5.png"
    histogram_path = figure_dir / "confidence_histogram_part5.png"
    popularity_path = figure_dir / "popularity_confidence_distribution_part5.png"
    family_path = figure_dir / "part5_family_compare.png"

    plot_reliability(pointwise_source, reliability_path)
    plot_confidence_histogram(pointwise_source, histogram_path)
    plot_popularity_confidence(pointwise_source, popularity_path)
    plot_family_compare(family_source, family_path)
    markdown_path = write_figure_pack_markdown(output_root=output_root)

    return {
        "pointwise_source": pointwise_source_path,
        "family_source": family_source_path,
        "reliability": reliability_path,
        "histogram": histogram_path,
        "popularity": popularity_path,
        "family_compare": family_path,
        "markdown": markdown_path,
    }

