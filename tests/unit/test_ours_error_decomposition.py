from __future__ import annotations

import json
from pathlib import Path

from llm4rec.analysis.ours_error_decomposition import decision_attribution


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(example_id: str, target: str, predicted: list[str], decision: str, confidence: float = 0.9) -> dict:
    return {
        "user_id": example_id.split(":")[0],
        "target_item": target,
        "candidate_items": ["a", "b", "c", target],
        "predicted_items": predicted,
        "scores": [1.0 for _ in predicted],
        "method": "ours_uncertainty_guided_real",
        "domain": "movies",
        "raw_output": "{}",
        "metadata": {
            "example_id": example_id,
            "uncertainty_decision": decision,
            "confidence": confidence,
            "parse_success": True,
            "grounding_success": True,
            "grounding_score": 1.0,
            "candidate_adherent": True,
            "grounded_item_id": predicted[0] if predicted else None,
            "is_grounded_hit": bool(predicted and predicted[0] == target),
        },
    }


def test_decision_attribution_counts_help_and_hurt(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "items.csv").write_text(
        "item_id,title,description,category,brand,domain,raw_text\n"
        "a,A,,,,movies,A\nb,B,,,,movies,B\nc,C,,,,movies,C\nt,T,,,,movies,T\n",
        encoding="utf-8",
    )
    _write_jsonl(
        processed / "examples.jsonl",
        [{"example_id": "u:1", "split": "train", "history": ["a"], "target": "b"}],
    )
    ours = [
        _row("u:1", "t", ["a", "b", "c"], "accept"),
        _row("u:2", "t", ["t", "a", "b"], "accept"),
    ]
    fallback = [
        _row("u:1", "t", ["t", "a", "b"], "fallback", confidence=0.0),
        _row("u:2", "t", ["a", "b", "c"], "fallback", confidence=0.0),
    ]
    _write_jsonl(
        runs / "r3_movielens_1m_real_llm_full_candidate500_ours_uncertainty_guided_real_seed13" / "predictions.jsonl",
        ours,
    )
    _write_jsonl(
        runs / "r3_movielens_1m_real_llm_full_candidate500_ours_fallback_only_seed13" / "predictions.jsonl",
        fallback,
    )
    result = decision_attribution(runs, processed_dir=processed, seeds=(13,))
    accept = next(row for row in result["decision_attribution"] if row["decision"] == "accept")
    assert accept["count"] == 2
    assert accept["help_count"] == 1
    assert accept["hurt_count"] == 1
    assert accept["delta_hit@10_sum"] == 0

