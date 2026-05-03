from __future__ import annotations

import json
from pathlib import Path

from llm4rec.analysis.r3b_table_export import export_r3b_tables
from llm4rec.io.artifacts import write_jsonl


def _minimal_metrics() -> dict:
    return {
        "aggregate": {
            "recall@10": 0.05,
            "ndcg@10": 0.02,
            "mrr@10": 0.01,
            "validity_rate": 1.0,
            "hallucination_rate": 0.0,
            "parse_success_rate": 1.0,
            "grounding_success_rate": 1.0,
            "confidence": {"mean_confidence": 0.5, "confidence_count": 2},
            "calibration": {"ece": 0.1, "brier_score": 0.2},
        }
    }


def _pred(example_id: str, *, conf: float, correct: bool, decision: str | None = None) -> dict:
    meta = {
        "example_id": example_id,
        "parse_success": True,
        "confidence": conf,
        "is_grounded_hit": correct,
    }
    if decision:
        meta["uncertainty_decision"] = decision
    tgt = "t1" if example_id == "e1" else "t2"
    pred = tgt if correct else "x"
    return {
        "user_id": "u1",
        "target_item": tgt,
        "candidate_items": [tgt, "x"],
        "predicted_items": [pred, "x"],
        "scores": [1.0, 0.1],
        "method": "ours_fallback_only",
        "domain": "movies",
        "raw_output": None,
        "metadata": meta,
    }


def test_export_r3b_tables_skips_incomplete_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    out = tmp_path / "tables"
    for method, seed in [("bm25", 13), ("ours_conservative_uncertainty_gate", 13)]:
        rd = runs / f"r3b_movielens_1m_conservative_gate_cache_replay_{method}_seed{seed}"
        rd.mkdir(parents=True)
        if method == "bm25":
            (rd / "metrics.json").write_text(json.dumps(_minimal_metrics()), encoding="utf-8")
            preds = [_pred("e1", conf=0.8, correct=True)]
            for p in preds:
                p["method"] = method
            write_jsonl(rd / "predictions.jsonl", preds)
            (rd / "cost_latency.json").write_text("{}", encoding="utf-8")
        else:
            (rd / "logs.txt").write_text("crashed before metrics", encoding="utf-8")
    manifest = export_r3b_tables(runs_dir=runs, output_dir=out)
    assert manifest["run_count_complete"] == 1
    assert len(manifest["skipped_incomplete_runs"]) == 1
    assert (out / "r3b_conservative_gate_main.csv").exists()


def test_export_r3b_tables_smoke(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    out = tmp_path / "tables"
    for method, seed in [("bm25", 13), ("ours_fallback_only", 13)]:
        rd = runs / f"r3b_movielens_1m_conservative_gate_cache_replay_{method}_seed{seed}"
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(json.dumps(_minimal_metrics()), encoding="utf-8")
        preds = [
            _pred("e1", conf=0.8, correct=True),
            _pred("e2", conf=0.2, correct=False),
        ]
        for p in preds:
            p["method"] = method
        write_jsonl(rd / "predictions.jsonl", preds)
        (rd / "cost_latency.json").write_text(
            json.dumps({"cache_hit_requests": 2.0, "live_provider_requests": 0.0, "replay_latency_seconds_sum": 0.01}),
            encoding="utf-8",
        )
    manifest = export_r3b_tables(runs_dir=runs, output_dir=out)
    assert (out / "r3b_conservative_gate_main.csv").exists()
    assert manifest["run_count_complete"] == 2
    assert manifest["run_count_discovered"] == 2
