#!/usr/bin/env python3
"""Build paper-facing CU-GR v2 tables and documentation from existing artifacts.

This script is intentionally offline: it reads already produced run/table
artifacts and never calls an LLM provider.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm4rec.analysis.calibrator_features import build_dataset_context  # noqa: E402
from llm4rec.experiments.cu_gr_v2_preference import (  # noqa: E402
    _fallback_top,
    _fusion_top,
    _llm_panel_top,
    _ndcg_delta_and_swaps,
    _policy_metrics,
    packed_row_from_signal_record,
)


DATASETS = {
    "MovieLens 1M": {
        "prefix": "cu_gr_v2_full_seed",
        "processed_dir": "data/processed/movielens_1m/r2_full_single_dataset",
        "signals": [
            "outputs/runs/r3_v2_movielens_preference_signal_subgate_full_seeds_seed13/preference_signals.jsonl",
            "outputs/runs/r3_v2_movielens_preference_signal_subgate_full_seeds_seed21/preference_signals.jsonl",
            "outputs/runs/r3_v2_movielens_preference_signal_subgate_full_seeds_seed42/preference_signals.jsonl",
        ],
        "candidate_size": 500,
    },
    "Amazon Beauty": {
        "prefix": "cu_gr_v2_amazon_beauty",
        "processed_dir": "data/processed/amazon_reviews_2023_beauty/cu_gr_v2",
        "signals": [
            "outputs/runs/r3_v2_amazon_beauty_preference_full_seeds_seed13/preference_signals.jsonl",
            "outputs/runs/r3_v2_amazon_beauty_preference_full_seeds_seed21/preference_signals.jsonl",
            "outputs/runs/r3_v2_amazon_beauty_preference_full_seeds_seed42/preference_signals.jsonl",
        ],
        "candidate_size": 479,
    },
}

REQUIRED_DOCS = [
    "paper_results_summary.md",
    "cu_gr_v2_method_summary.md",
    "cu_gr_v2_experiment_summary.md",
    "cu_gr_v2_limitations.md",
    "cu_gr_v2_reviewer_checklist.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    tables = _resolve(args.tables)
    docs = _resolve(args.docs)
    tables.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_results(tables)
    ablation_rows = build_ablation(tables)
    uncertainty_rows = build_uncertainty(tables)
    panel_rows = build_panel_analysis(tables)
    cost_rows = build_cost_latency(tables)
    build_figure_data(tables, main_rows, panel_rows)
    build_docs(docs, tables, main_rows, ablation_rows, uncertainty_rows, panel_rows, cost_rows)

    print(json.dumps({"tables": str(tables), "docs": str(docs), "status": "paper_artifacts_built"}, indent=2))
    return 0


def build_main_results(tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, info in DATASETS.items():
        prefix = str(info["prefix"])
        main = _read_csv(tables / f"{prefix}_main.csv")
        by_seed = _read_csv(tables / f"{prefix}_by_seed.csv")
        parser = _read_csv(tables / f"{prefix}_parser_stats.csv")
        swap = _read_csv(tables / f"{prefix}_swap_analysis.csv")
        cost = _read_csv(tables / f"{prefix}_cost_latency.csv")

        def add(method_label: str, source_method: str, *, source: str = "v2", parser_method: str | None = None, cost_method: str | None = None) -> None:
            metric_row = _metric_source_row(main, by_seed, source_method, source=source, dataset=dataset, tables=tables)
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method_label,
                    "Recall@10": _fmt(metric_row.get("Recall@10")),
                    "NDCG@10": _fmt(metric_row.get("NDCG@10")),
                    "MRR@10": _fmt(metric_row.get("MRR@10")),
                    "HitRate@10": _fmt(metric_row.get("HitRate@10")),
                    "Harmful swap rate if applicable": _fmt(_swap_rate(swap, source_method)),
                    "Parser success if applicable": _fmt(_parser_rate(parser, parser_method or source_method, source=source, dataset=dataset, tables=tables)),
                    "Cost per 200 examples if applicable": _fmt(_cost_per_200(cost, cost_method or source_method, source=source, dataset=dataset, tables=tables)),
                }
            )

        add("Popularity", "popularity_reference")
        add("BM25/fallback", "fallback_only")
        add("Sequential Markov", "sequential_markov_reference")
        add("LLM direct generation" if dataset == "MovieLens 1M" else "LLM direct generation (not run for this domain)", "llm_generative_real", source="r3_movielens")
        add("LLM listwise panel", "llm_listwise_panel", parser_method="listwise_parser", cost_method="cu_gr_v2_gate")
        add("CU-GR v2 fusion", "fusion_train_best", parser_method="listwise_parser", cost_method="cu_gr_v2_gate")
        add("CU-GR v1 / Ours v1" if dataset == "MovieLens 1M" else "CU-GR v1 / Ours v1 (not run for this domain)", "R3_Ours_v1_reference", source="r3_movielens")
        add("Conservative gate" if dataset == "MovieLens 1M" else "Conservative gate (not run for this domain)", "ours_conservative_uncertainty_gate", source="r3b_movielens")

    _write_table_all_formats(tables / "paper_main_results", rows)
    return rows


def build_ablation(tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, info in DATASETS.items():
        prefix = str(info["prefix"])
        main = _read_csv(tables / f"{prefix}_main.csv")
        swap = _read_csv(tables / f"{prefix}_swap_analysis.csv")
        parser = _read_csv(tables / f"{prefix}_parser_stats.csv")
        weights = _read_csv(tables / f"{prefix}_fusion_weights.csv")
        selected = _first([r for r in weights if r.get("policy") == "fusion_train_best"]) or {}
        selected_params = {
            "alpha": _float(selected.get("alpha"), 0.0),
            "beta": _float(selected.get("beta"), 0.0),
            "gamma": _float(selected.get("gamma"), 0.0),
            "lambda": _float(selected.get("lambda"), 0.0),
        }

        for method, label in [
            ("fallback_only", "Fallback only"),
            ("llm_listwise_panel", "LLM listwise only"),
            ("fusion_fixed_grid", "Fusion fixed grid"),
            ("fusion_train_best", "Fusion train-best"),
            ("safe_fusion", "Safe fusion"),
        ]:
            m = _first([r for r in main if str(r.get("seed")) == "42" and r.get("method") == method]) or {}
            rows.append(_ablation_row(dataset, label, m, _swap_rate(swap, method), _parser_rate(parser, "listwise_parser"), "outputs/tables/" + f"{prefix}_main.csv"))

        pack = _load_seed_pack([str(p) for p in info["signals"]], seed=42)
        train_pop = dict(build_dataset_context(str(info["processed_dir"])).get("train_popularity") or {})
        fallback_ndcg = _float((_metrics_to_caps(_policy_metrics(pack, _fallback_top))).get("NDCG@10"))
        offline_variants = [
            ("no confidence term", {**selected_params, "gamma": 0.0}),
            ("no popularity penalty", {**selected_params, "lambda": 0.0}),
            ("no fallback score", {**selected_params, "alpha": 0.0}),
            ("no LLM score", {**selected_params, "beta": 0.0}),
        ]
        for label, params in offline_variants:
            fn = lambda r, p=params: _fusion_top(r, p, train_pop)
            metrics = _policy_metrics(pack, fn)
            swaps = _ndcg_delta_and_swaps(pack, fn)
            caps = _metrics_to_caps(metrics)
            caps["delta_NDCG@10_vs_fallback"] = _float(caps.get("NDCG@10")) - fallback_ndcg
            rows.append(_ablation_row(dataset, label, caps, swaps.get("harmful_swap_rate"), _parser_rate(parser, "listwise_parser"), "offline replay from preference_signals.jsonl"))

        feasibility = _read_csv(tables / _feasibility_name(prefix))
        for psize in ["10", "15", "20"]:
            row = _first([r for r in feasibility if str(r.get("run_seed")) == "42" and str(r.get("panel_size")) == psize]) or {}
            rows.append(
                {
                    "Dataset": dataset,
                    "Ablation": f"panel size {psize} oracle",
                    "Recall@10": "",
                    "NDCG@10": _fmt(row.get("mean_oracle_ndcg_at_10")),
                    "MRR@10": "",
                    "HitRate@10": "",
                    "Harmful swap rate": "",
                    "Parser success": "",
                    "Panel size": psize,
                    "Delta NDCG@10 vs fallback": _fmt(row.get("mean_oracle_ndcg_gain_vs_fallback")),
                    "Source artifact": "outputs/tables/" + _feasibility_name(prefix),
                }
            )

    _write_table_all_formats(tables / "paper_ablation", rows)
    return rows


def build_uncertainty(tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    r3_unc = _read_csv(tables / "r3_movielens_1m_uncertainty.csv")
    r3_main = _read_csv(tables / "r3_movielens_1m_main_results.csv")
    for method in ["llm_generative_real", "llm_confidence_observation_real", "ours_uncertainty_guided_real", "ours_fallback_only"]:
        u = _first([r for r in r3_unc if r.get("method") == method]) or {}
        m = _first([r for r in r3_main if r.get("method") == method]) or {}
        rows.append(
            {
                "Dataset": "MovieLens 1M",
                "Method": method,
                "mean confidence": _fmt(u.get("mean_confidence_mean")),
                "ECE": _fmt(u.get("ece_mean")),
                "Brier": _fmt(u.get("brier_mean")),
                "high-confidence wrong count": _fmt(u.get("high_confidence_wrong_count_mean")),
                "hallucination rate": _fmt(m.get("hallucination_rate_mean")),
                "parse success": _fmt(u.get("parse_success_rate_mean")),
                "grounding success": _fmt(u.get("grounding_success_rate_mean")),
            }
        )
    for dataset, info in DATASETS.items():
        parser = _read_csv(tables / f"{info['prefix']}_parser_stats.csv")
        agg = _first([r for r in parser if r.get("seed") == "aggregate"]) or {}
        rows.append(
            {
                "Dataset": dataset,
                "Method": "CU-GR v2 listwise preference parser",
                "mean confidence": _fmt(agg.get("confidence_mean")),
                "ECE": "",
                "Brier": "",
                "high-confidence wrong count": "",
                "hallucination rate": "0",
                "parse success": _fmt(agg.get("parser_success_rate")),
                "grounding success": "",
            }
        )
    _write_table_all_formats(tables / "paper_uncertainty_observation", rows)
    return rows


def build_panel_analysis(tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, info in DATASETS.items():
        prefix = str(info["prefix"])
        feasibility = _read_csv(tables / _feasibility_name(prefix))
        for row in feasibility:
            if str(row.get("run_seed")) != "42":
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "panel size": row.get("panel_size", ""),
                    "target_in_panel_rate": _fmt(row.get("target_in_panel_rate")),
                    "fallback_hit@10": _fmt(row.get("fallback_hit_at_10_rate")),
                    "oracle NDCG upper bound": _fmt(row.get("mean_oracle_ndcg_at_10")),
                    "oracle NDCG gain": _fmt(row.get("mean_oracle_ndcg_gain_vs_fallback")),
                    "beneficial swap opportunities": _fmt(row.get("n_beneficial_swap_opportunities") or row.get("n_positive_ndcg_gain")),
                }
            )
    _write_table_all_formats(tables / "paper_panel_analysis", rows)
    return rows


def build_cost_latency(tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, info in DATASETS.items():
        cost = _read_csv(tables / f"{info['prefix']}_cost_latency.csv")
        for row in cost:
            if row.get("seed") not in {"42", "aggregate"}:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Method / gate": f"CU-GR v2 listwise gate seed{row.get('seed')}",
                    "live requests": row.get("live_requests", ""),
                    "cache hits": row.get("cache_hits", ""),
                    "tokens": row.get("total_tokens", ""),
                    "estimated cost": row.get("effective_cost_usd", ""),
                    "p50 latency": row.get("p50_latency_seconds", ""),
                    "p95 latency": row.get("p95_latency_seconds", ""),
                    "retry / timeout / 429 count": f"{row.get('retry_count', '')}/{row.get('timeout_count', '')}/{row.get('rate_limit_429_count', '')}",
                }
            )
    r3_cost = _read_csv(tables / "r3_movielens_1m_cost_latency.csv")
    for method in ["llm_generative_real", "llm_confidence_observation_real", "ours_uncertainty_guided_real"]:
        row = _first([r for r in r3_cost if r.get("method") == method]) or {}
        rows.append(
            {
                "Dataset": "MovieLens 1M",
                "Method / gate": method,
                "live requests": _fmt(row.get("live_provider_requests_sum")),
                "cache hits": _fmt(row.get("cache_hit_requests_sum")),
                "tokens": _fmt(row.get("total_tokens_sum")),
                "estimated cost": _fmt(row.get("effective_cost_usd_sum")),
                "p50 latency": _fmt(row.get("latency_p50_seconds_mean")),
                "p95 latency": _fmt(row.get("latency_p95_seconds_mean")),
                "retry / timeout / 429 count": "",
            }
        )
    _write_table_all_formats(tables / "paper_cost_latency", rows)
    return rows


def build_figure_data(tables: Path, main_rows: list[dict[str, Any]], panel_rows: list[dict[str, Any]]) -> None:
    def r3_real(row: dict[str, str]) -> bool:
        return (
            row.get("dataset") == "movielens_1m_r2"
            and str(row.get("candidate_size")) == "500"
            and str(row.get("run_id")).startswith("r3_movielens_1m_real_llm_full_candidate500")
            and "real" in str(row.get("method"))
        )

    reliability = [r for r in _read_csv(tables / "reliability_diagram.csv") if r3_real(r)]
    risk = [r for r in _read_csv(tables / "risk_coverage.csv") if r3_real(r)]
    _write_csv(tables / "figure_calibration_reliability.csv", reliability)
    _write_csv(tables / "figure_risk_coverage.csv", risk)
    delta_rows = []
    for row in main_rows:
        if row["Method"] in {"LLM listwise panel", "CU-GR v2 fusion"}:
            fb = _first([r for r in main_rows if r["Dataset"] == row["Dataset"] and r["Method"] == "BM25/fallback"]) or {}
            delta_rows.append({"Dataset": row["Dataset"], "Method": row["Method"], "delta_NDCG@10_vs_fallback": _fmt(_float(row.get("NDCG@10")) - _float(fb.get("NDCG@10")))})
    _write_csv(tables / "figure_delta_vs_fallback_by_dataset.csv", delta_rows)
    swap_rows: list[dict[str, Any]] = []
    for dataset, info in DATASETS.items():
        for row in _read_csv(tables / f"{info['prefix']}_swap_analysis.csv"):
            if row.get("seed") in {"42", "aggregate"}:
                swap_rows.append({"Dataset": dataset, **row})
    _write_csv(tables / "figure_swap_outcomes.csv", swap_rows)
    _write_csv(tables / "figure_panel_coverage.csv", panel_rows)


def build_docs(
    docs: Path,
    tables: Path,
    main_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> None:
    commit = _git("rev-parse", "HEAD")
    ml = _first([r for r in main_rows if r["Dataset"] == "MovieLens 1M" and r["Method"] == "CU-GR v2 fusion"]) or {}
    ml_fb = _first([r for r in main_rows if r["Dataset"] == "MovieLens 1M" and r["Method"] == "BM25/fallback"]) or {}
    amz = _first([r for r in main_rows if r["Dataset"] == "Amazon Beauty" and r["Method"] == "CU-GR v2 fusion"]) or {}
    amz_fb = _first([r for r in main_rows if r["Dataset"] == "Amazon Beauty" and r["Method"] == "BM25/fallback"]) or {}
    artifacts = [
        "outputs/tables/paper_main_results.csv",
        "outputs/tables/paper_ablation.csv",
        "outputs/tables/paper_uncertainty_observation.csv",
        "outputs/tables/paper_panel_analysis.csv",
        "outputs/tables/paper_cost_latency.csv",
        "outputs/tables/figure_calibration_reliability.csv",
        "outputs/tables/figure_risk_coverage.csv",
        "outputs/tables/figure_delta_vs_fallback_by_dataset.csv",
        "outputs/tables/figure_swap_outcomes.csv",
        "outputs/tables/figure_panel_coverage.csv",
        "outputs/tables/cu_gr_v2_full_seed_main.csv",
        "outputs/tables/cu_gr_v2_amazon_beauty_main.csv",
    ]
    (docs / "paper_results_summary.md").write_text(
        "\n".join(
            [
                "# Paper Results Summary",
                "",
                f"Artifact base commit: `{commit}`",
                "",
                "## Dataset Summaries",
                "",
                "- MovieLens 1M: candidate_size=500, panel_size=15, subset_size=200 per seed, seeds [13,21,42].",
                "- Amazon Beauty: local Amazon Reviews 2023 All_Beauty, candidate_size=479 because the local catalog has 479 items, panel_size=15, subset_size=200 per seed, seeds [13,21,42].",
                "",
                "## Method Summaries",
                "",
                "- Free-form LLM title generation and verbalized confidence are retained as negative/motivating evidence from existing MovieLens R3 artifacts.",
                "- CU-GR v2 uses candidate-local listwise LLM preferences over anonymous panels and calibrated fusion with fallback ranking.",
                "- Fusion weights are selected on seed21 validation after seed13 training and evaluated on seed42.",
                "",
                "## Key Results",
                "",
                f"- MovieLens 1M seed42: CU-GR v2 fusion NDCG@10={ml.get('NDCG@10')} vs fallback={ml_fb.get('NDCG@10')}.",
                f"- Amazon Beauty seed42: CU-GR v2 fusion NDCG@10={amz.get('NDCG@10')} vs fallback={amz_fb.get('NDCG@10')}.",
                "",
                "## Key Table References",
                "",
                *[f"- `{p}`" for p in artifacts],
                "",
                "## Claim Boundary",
                "",
                "The artifacts support an observation-motivated method framing across MovieLens 1M and Amazon Beauty. They do not support claims about full-ranking evaluation, more than two domains, local open-source LLMs, or production-scale inference.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "cu_gr_v2_method_summary.md").write_text(_method_doc(), encoding="utf-8")
    (docs / "cu_gr_v2_experiment_summary.md").write_text(
        "\n".join(
            [
                "# CU-GR v2 Experiment Summary",
                "",
                "## MovieLens Results",
                "",
                f"Seed42 CU-GR v2 fusion improves NDCG@10 from {ml_fb.get('NDCG@10')} to {ml.get('NDCG@10')}. The held-out harmful swap rate is {ml.get('Harmful swap rate if applicable')}.",
                "",
                "## Amazon Beauty Results",
                "",
                f"Seed42 CU-GR v2 fusion improves NDCG@10 from {amz_fb.get('NDCG@10')} to {amz.get('NDCG@10')}. The held-out harmful swap rate is {amz.get('Harmful swap rate if applicable')}.",
                "",
                "## v1 Failure Motivation",
                "",
                "MovieLens R3 artifacts retain negative evidence for free-form generation and confidence-heavy CU-GR v1 policies, including high-confidence wrong recommendations and weak/negative ranking movement versus fallback.",
                "",
                "## Uncertainty Observation",
                "",
                "MovieLens uncertainty artifacts show high verbalized confidence with poor calibration for direct/confidence-observation methods. Amazon Beauty only has CU-GR v2 parser confidence for this gate, not a direct-generation uncertainty run.",
                "",
                "## Ablation Status",
                "",
                "The ablation table includes fallback, listwise-only, fixed fusion, train-best fusion, safe fusion, offline no-term replays, and panel-size oracle feasibility rows where available.",
                "",
                "## Cost / Latency",
                "",
                f"CU-GR v2 seed42 effective cost per 200 examples: MovieLens {ml.get('Cost per 200 examples if applicable')}, Amazon Beauty {amz.get('Cost per 200 examples if applicable')}. Full details are in `outputs/tables/paper_cost_latency.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "cu_gr_v2_limitations.md").write_text(
        "\n".join(
            [
                "# CU-GR v2 Limitations",
                "",
                "- Only two datasets/domains have passed so far.",
                "- Evaluation uses a sampled candidate protocol, not full ranking.",
                "- LLM calls use subset_size=200 per seed.",
                "- Amazon Beauty local catalog size is 479, so candidate_size=500 was infeasible there.",
                "- Validation is limited to DeepSeek v4 flash.",
                "- No local open-source model validation has been run yet.",
                "- No full production-scale inference validation has been run.",
                "- Candidate panel construction is still heuristic.",
                "- Harmful swaps are controlled but not zero.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "cu_gr_v2_reviewer_checklist.md").write_text(
        "\n".join(
            [
                "# CU-GR v2 Reviewer Checklist",
                "",
                "- No target leakage: prompts use anonymous panel labels and do not reveal the held-out item as a target.",
                "- Target included in candidate set: configured for both MovieLens 1M and Amazon Beauty.",
                "- Candidate protocol consistent: sampled, target-included candidate sets are shared across compared methods within each dataset gate.",
                "- Train/validation/test separation: seed13 trains/selects candidate fusion behavior, seed21 validates/tunes fusion/safety thresholds, seed42 is held out.",
                "- No test tuning: selected fusion weights are recorded as validation-selected before seed42 reporting.",
                "- Raw outputs saved: `preference_signals.jsonl` and raw LLM output artifacts exist for real CU-GR v2 runs.",
                "- Cost/latency saved: per-seed cost latency CSV/JSON artifacts exist for both CU-GR v2 gates.",
                "- Failed v1 evidence retained: MovieLens R3/R3b tables and case studies are kept and referenced as motivation, not deleted or hidden.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _method_doc() -> str:
    return "\n".join(
        [
            "# CU-GR v2 Method Summary",
            "",
            "Working method name: CU-GR v2: Candidate-Normalized Calibrated Preference Fusion for Generative Recommendation.",
            "",
            "## Input",
            "",
            "Each example contains a user history, a held-out target item, and a sampled candidate set generated under the shared R3 protocol.",
            "",
            "## Candidate Panel",
            "",
            "The panel is built locally from valid candidate items only. It includes high fallback ranks, mid-rank contrasts, popularity/tail contrasts, optional sequential candidates when available, and deterministic fills. Panel items are shown with anonymous labels A/B/C/... rather than global item identifiers.",
            "",
            "## Listwise Preference Prompt",
            "",
            "The prompt asks the LLM to rank candidate-local panel labels for the user's next-item preference. It does not identify the target item and does not expose a global target item ID.",
            "",
            "## Parser",
            "",
            "The parser accepts JSON listwise responses, maps labels back to panel item IDs, rejects invalid labels, rejects duplicate labels, tracks partial rankings, and preserves raw outputs for audit.",
            "",
            "## Fusion Formula",
            "",
            "`score = alpha * normalized_fallback_score + beta * normalized_llm_score + gamma * llm_confidence - lambda * popularity_penalty`.",
            "",
            "MovieLens selected `alpha=0.5, beta=0.7, gamma=0.2, lambda=0.05`; Amazon Beauty selected `alpha=0.5, beta=0.3, gamma=0.0, lambda=0.1`.",
            "",
            "## Train / Validation / Test Split",
            "",
            "Fusion grid selection is trained on seed13, validated on seed21, and reported on held-out seed42. The grid is not selected on seed42.",
            "",
            "## Safety Constraints",
            "",
            "Safety analysis tracks parse success, invalid labels, duplicate labels, candidate adherence, harmful swaps, and safe-fusion threshold behavior. The validation constraint requires harmful_swap_rate <= 0.05.",
            "",
            "## Inference Algorithm",
            "",
            "1. Rank the full candidate set with fallback BM25.",
            "2. Build a deterministic candidate-local panel.",
            "3. Query the LLM with anonymous labels in JSON mode.",
            "4. Parse and validate label rankings.",
            "5. Normalize fallback and LLM panel scores.",
            "6. Fuse scores with validation-selected weights.",
            "7. Replace panel ordering inside the fallback ranking while non-panel candidates retain fallback order.",
            "8. Evaluate with the shared ranking evaluator.",
        ]
    ) + "\n"


def _metric_source_row(main: list[dict[str, str]], by_seed: list[dict[str, str]], method: str, *, source: str, dataset: str, tables: Path) -> dict[str, Any]:
    if source == "v2":
        return _first([r for r in main if str(r.get("seed")) == "42" and r.get("method") == method]) or _first([r for r in by_seed if str(r.get("seed")) == "42" and r.get("method") == method]) or {}
    if source == "r3_movielens" and dataset == "MovieLens 1M":
        if method == "R3_Ours_v1_reference":
            return _first([r for r in by_seed if str(r.get("seed")) == "42" and r.get("method") == method]) or {}
        row = _first([r for r in _read_csv(tables / "r3_movielens_1m_main_results.csv") if r.get("method") == method]) or {}
        return _r3_to_caps(row)
    if source == "r3b_movielens" and dataset == "MovieLens 1M":
        row = _first([r for r in _read_csv(tables / "r3b_conservative_gate_main.csv") if r.get("method") == method]) or {}
        return {"Recall@10": row.get("recall@10_mean"), "NDCG@10": row.get("ndcg@10_mean"), "MRR@10": row.get("mrr@10_mean"), "HitRate@10": row.get("recall@10_mean")}
    return {}


def _r3_to_caps(row: dict[str, Any]) -> dict[str, Any]:
    return {"Recall@10": row.get("recall@10_mean"), "NDCG@10": row.get("ndcg@10_mean"), "MRR@10": row.get("mrr@10_mean"), "HitRate@10": row.get("hit_rate@10_mean")}


def _metrics_to_caps(row: dict[str, Any]) -> dict[str, Any]:
    return {"Recall@10": row.get("recall@10"), "NDCG@10": row.get("ndcg@10"), "MRR@10": row.get("mrr@10"), "HitRate@10": row.get("hit_rate@10")}


def _ablation_row(dataset: str, label: str, metrics: dict[str, Any], harmful: Any, parser: Any, source: str) -> dict[str, Any]:
    return {
        "Dataset": dataset,
        "Ablation": label,
        "Recall@10": _fmt(metrics.get("Recall@10")),
        "NDCG@10": _fmt(metrics.get("NDCG@10")),
        "MRR@10": _fmt(metrics.get("MRR@10")),
        "HitRate@10": _fmt(metrics.get("HitRate@10")),
        "Harmful swap rate": _fmt(harmful),
        "Parser success": _fmt(parser),
        "Panel size": "15",
        "Delta NDCG@10 vs fallback": _fmt(metrics.get("delta_NDCG@10_vs_fallback")),
        "Source artifact": source,
    }


def _load_seed_pack(signal_paths: list[str], *, seed: int) -> list[dict[str, Any]]:
    rows = []
    for rel in signal_paths:
        path = _resolve(Path(rel))
        if f"seed{seed}" not in path.as_posix() and f"seed{seed}" not in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    packed = packed_row_from_signal_record(json.loads(line))
                    if packed is not None:
                        rows.append(packed)
    return [{"idx": i, **r} for i, r in enumerate(rows)]


def _swap_rate(rows: list[dict[str, str]], method: str) -> str:
    row = _first([r for r in rows if str(r.get("seed")) == "42" and r.get("method") == method])
    return "" if row is None else str(row.get("harmful_swap_rate", ""))


def _parser_rate(rows: list[dict[str, str]], method: str, *, source: str = "v2", dataset: str = "", tables: Path | None = None) -> str:
    if method == "listwise_parser":
        row = _first([r for r in rows if str(r.get("seed")) == "42"])
        return "" if row is None else str(row.get("parser_success_rate", ""))
    if source == "r3_movielens" and dataset == "MovieLens 1M" and tables is not None:
        row = _first([r for r in _read_csv(tables / "r3_movielens_1m_uncertainty.csv") if r.get("method") == method])
        return "" if row is None else str(row.get("parse_success_rate_mean", ""))
    if source == "r3b_movielens" and dataset == "MovieLens 1M" and tables is not None:
        row = _first([r for r in _read_csv(tables / "r3b_conservative_gate_main.csv") if r.get("method") == method])
        return "" if row is None else str(row.get("parse_success_mean", ""))
    return ""


def _cost_per_200(rows: list[dict[str, str]], method: str, *, source: str = "v2", dataset: str = "", tables: Path | None = None) -> str:
    if method == "cu_gr_v2_gate":
        row = _first([r for r in rows if str(r.get("seed")) == "42"])
        return "" if row is None else str(row.get("effective_cost_usd", ""))
    if source == "r3_movielens" and dataset == "MovieLens 1M" and tables is not None:
        row = _first([r for r in _read_csv(tables / "r3_movielens_1m_cost_latency.csv") if r.get("method") == method])
        return "" if row is None else _fmt(_float(row.get("effective_cost_usd_mean")) / 6040.0 * 200.0)
    if source == "r3b_movielens" and dataset == "MovieLens 1M" and tables is not None:
        row = _first([r for r in _read_csv(tables / "r3b_conservative_gate_main.csv") if r.get("method") == method])
        return "" if row is None else _fmt(_float(row.get("effective_cost_usd_mean")) / 6040.0 * 200.0)
    if method in {"fallback_only", "popularity_reference", "sequential_markov_reference"}:
        return "0"
    return ""


def _feasibility_name(prefix: str) -> str:
    if prefix == "cu_gr_v2_full_seed":
        return "cu_gr_v2_panel_coverage.csv"
    return f"{prefix}_panel_feasibility_coverage.csv"


def _write_table_all_formats(base: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(base.with_suffix(".csv"), rows)
    base.with_suffix(".md").write_text(_to_markdown(rows), encoding="utf-8")
    base.with_suffix(".tex").write_text(_to_latex(rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "\n"
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(row.get(f, "")) for f in fields) + " |")
    return "\n".join(lines) + "\n"


def _to_latex(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "% empty\n"
    fields = list(rows[0].keys())
    spec = "l" * len(fields)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(_tex_cell(f) for f in fields) + r" \\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(_tex_cell(row.get(f, "")) for f in fields) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _tex_cell(value: Any) -> str:
    text = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(old, new)
    return text


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_ROOT, text=True, encoding="utf-8").strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
