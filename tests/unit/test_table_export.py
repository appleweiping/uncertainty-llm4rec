from __future__ import annotations

import json

from llm4rec.evaluation.table_export import export_phase5_tables


def test_export_phase5_tables_writes_aggregate_and_confidence_tables(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "smoke_llm_seed13"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "metadata": {"method": "llm_generative_mock", "methods": ["llm_generative_mock"], "dataset": "tiny", "seed": 13},
            "aggregate": {
                "recall@1": 0.0,
                "confidence": {"selective_risk_coverage_data": [{"coverage": 1.0, "risk": 1.0, "threshold": 0.8}]},
                "calibration": {"reliability_diagram_data": [{"bin": 8, "count": 1, "accuracy": 0.0, "mean_confidence": 0.8}]},
                "long_tail": {"confidence_by_popularity_bucket": {"head": {"count": 1, "mean_confidence": 0.8}}},
            },
            "by_domain": {"tiny": {"recall@1": 0.0}},
        }),
        encoding="utf-8",
    )
    manifest = export_phase5_tables(tmp_path / "runs", output_dir=tmp_path / "tables")
    assert (tmp_path / "tables" / "aggregate_metrics.csv").exists()
    assert (tmp_path / "tables" / "aggregate_metrics.md").exists()
    assert (tmp_path / "tables" / "aggregate_metrics.tex").exists()
    assert (tmp_path / "tables" / "reliability_diagram.csv").exists()
    assert (tmp_path / "tables" / "risk_coverage.csv").exists()
    assert (tmp_path / "tables" / "confidence_by_popularity_bucket.csv").exists()
    assert "experiment_summary_json" in manifest


def test_export_phase5_tables_writes_r3_alias_tables(tmp_path) -> None:
    for seed, recall in [(13, 0.1), (21, 0.2)]:
        run_dir = tmp_path / "runs" / f"r3_movielens_1m_real_llm_full_candidate500_ours_uncertainty_guided_real_seed{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "method": "ours_uncertainty_guided_real",
                        "methods": ["ours_uncertainty_guided_real"],
                        "dataset": "movielens_1m_r2",
                        "seed": seed,
                        "candidate_size": 500,
                    },
                    "aggregate": {
                        "recall@10": recall,
                        "ndcg@10": recall / 2,
                        "mrr@10": recall / 3,
                        "hit_rate@10": recall,
                        "validity_rate": 1.0,
                        "hallucination_rate": 0.0,
                        "parse_success_rate": 1.0,
                        "grounding_success_rate": 1.0,
                        "confidence": {"mean_confidence": 0.8},
                        "calibration": {"ece": 0.7, "brier_score": 0.5},
                        "coverage_metrics": {"catalog_coverage@10": {"coverage_rate": 0.25}},
                        "novelty": {"novelty@10": 12.0},
                        "diversity": {"intra_list_diversity@10": 0.9},
                        "long_tail": {
                            "recall_by_popularity_bucket@10": {"head": 0.1, "mid": 0.2, "tail": 0.3},
                            "hit_rate_by_popularity_bucket@10": {"head": 0.1, "mid": 0.2, "tail": 0.3},
                        },
                        "efficiency": {
                            "total_requests": 2,
                            "live_provider_requests": 1,
                            "cache_hit_requests": 1,
                            "cache_hit_rate": 0.5,
                            "total_tokens": 100,
                            "live_cost_usd": 0.01,
                            "effective_cost_usd": 0.02,
                            "latency_p50_seconds": 1.0,
                            "latency_p95_seconds": 2.0,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "target_item": "i1",
                    "predicted_items": ["i2"],
                    "metadata": {"confidence": 0.9},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    manifest = export_phase5_tables(tmp_path / "runs", output_dir=tmp_path / "tables")

    assert (tmp_path / "tables" / "r3_movielens_1m_main_results.csv").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_main_results.md").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_main_results.tex").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_ablation.csv").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_uncertainty.csv").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_popularity_longtail.csv").exists()
    assert (tmp_path / "tables" / "r3_movielens_1m_cost_latency.csv").exists()
    assert "r3_movielens_1m_main_results_csv" in manifest
