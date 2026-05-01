from __future__ import annotations

from llm4rec.metrics.cold_start import metadata_availability_bucket, user_history_bucket


def test_user_history_buckets() -> None:
    assert user_history_bucket(2) == "cold_user"
    assert user_history_bucket(3) == "warm_user"
    assert user_history_bucket(5) == "warm_user"
    assert user_history_bucket(6) == "heavy_user"


def test_metadata_availability_bucket() -> None:
    assert metadata_availability_bucket({"title": "A"}) == "metadata_available"
    assert metadata_availability_bucket({}) == "metadata_missing"
    assert metadata_availability_bucket(None) == "metadata_missing"
