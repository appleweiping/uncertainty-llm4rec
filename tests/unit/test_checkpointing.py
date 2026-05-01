from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.trainers.checkpointing import load_checkpoint_manifest, load_model_state, save_checkpoint_artifacts


def test_checkpoint_artifacts_roundtrip(tmp_path: Path) -> None:
    saved = save_checkpoint_artifacts(
        tmp_path,
        method="sequential_markov",
        model_state={"method": "sequential_markov", "state": {"transitions": {"i1": {"i2": 2}}}},
        config={"method": {"name": "sequential_markov"}},
        metadata={"train_example_count": 3},
    )
    assert Path(saved["manifest_path"]).exists()
    manifest = load_checkpoint_manifest(tmp_path, expected_method="sequential_markov")
    assert manifest["method"] == "sequential_markov"
    assert load_model_state(tmp_path)["state"]["transitions"]["i1"]["i2"] == 2
    assert (tmp_path / "trainer_config.yaml").exists()


def test_checkpoint_missing_and_incompatible_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint manifest not found"):
        load_checkpoint_manifest(tmp_path)
    save_checkpoint_artifacts(
        tmp_path,
        method="sequential_last_item",
        model_state={"method": "sequential_last_item"},
        config={},
    )
    with pytest.raises(ValueError, match="incompatible checkpoint method"):
        load_checkpoint_manifest(tmp_path, expected_method="sequential_markov")
