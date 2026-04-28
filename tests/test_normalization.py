from __future__ import annotations

import pytest

from storyflow.grounding import normalize_title


def test_normalize_title_collapses_case_punctuation_and_articles() -> None:
    assert normalize_title("  The Matrix!!! ") == "matrix"
    assert normalize_title("A  SPACE   ODYSSEY") == "space odyssey"
    assert normalize_title("Wall-E & Friends") == "wall e and friends"


def test_normalize_title_rejects_none() -> None:
    with pytest.raises(ValueError):
        normalize_title(None)  # type: ignore[arg-type]
