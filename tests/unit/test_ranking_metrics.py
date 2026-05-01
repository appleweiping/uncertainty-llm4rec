from __future__ import annotations

from llm4rec.metrics.ranking import ranking_metrics


def _row(target: str, predicted: list[str], candidates: list[str] | None = None) -> dict[str, object]:
    return {
        "target_item": target,
        "predicted_items": predicted,
        "candidate_items": candidates or ["i1", "i2", "i3"],
    }


def test_ranking_metrics_perfect() -> None:
    metrics = ranking_metrics([_row("i1", ["i1", "i2"])], top_k=[1, 2])
    assert metrics["recall@1"] == 1.0
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0


def test_ranking_metrics_miss_and_empty() -> None:
    metrics = ranking_metrics([
        _row("i1", ["i2", "i3"]),
        _row("i1", []),
    ], top_k=[1, 5])
    assert metrics["recall@1"] == 0.0
    assert metrics["mrr@5"] == 0.0
    assert metrics["ndcg@5"] == 0.0


def test_ranking_metrics_deduplicates_predictions() -> None:
    metrics = ranking_metrics([_row("i2", ["i1", "i1", "i2"])], top_k=[2, 3])
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.5


def test_ranking_metrics_target_outside_candidates_does_not_crash() -> None:
    metrics = ranking_metrics([_row("missing", ["i1"], ["i1", "i2"])], top_k=[1])
    assert metrics["recall@1"] == 0.0
    assert metrics["coverage@1"] == 0.5
