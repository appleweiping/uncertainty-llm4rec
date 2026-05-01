from __future__ import annotations

import pytest

from llm4rec.evaluation.prediction_schema import validate_prediction


def _record() -> dict[str, object]:
    return {
        "user_id": "u1",
        "target_item": "i3",
        "candidate_items": ["i1", "i2", "i3"],
        "predicted_items": ["i3", "i2"],
        "scores": [1.0, 0.5],
        "method": "skeleton",
        "domain": "tiny",
        "raw_output": None,
        "metadata": {},
    }


def test_prediction_schema_accepts_valid_record() -> None:
    assert validate_prediction(_record())["scores"] == [1.0, 0.5]


def test_prediction_schema_requires_predicted_items_list() -> None:
    record = _record()
    record["predicted_items"] = "i1"
    with pytest.raises(ValueError, match="predicted_items must be list"):
        validate_prediction(record)


def test_prediction_schema_rejects_score_length_mismatch() -> None:
    record = _record()
    record["scores"] = [1.0]
    with pytest.raises(ValueError, match="scores length"):
        validate_prediction(record)


def test_prediction_schema_allows_invalid_item_for_validity_metric() -> None:
    record = _record()
    record["predicted_items"] = ["ghost"]
    record["scores"] = [1.0]
    assert validate_prediction(record)["predicted_items"] == ["ghost"]
