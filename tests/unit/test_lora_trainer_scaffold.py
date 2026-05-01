from __future__ import annotations

from pathlib import Path

import pytest

from llm4rec.trainers.lora import LoraTrainer


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "base_model_name_or_path": "local-placeholder-model",
        "output_dir": str(tmp_path / "lora"),
        "dataset": "tiny",
        "peft_method": "lora",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "max_steps": 1,
    }


def test_lora_trainer_dry_run_manifest(tmp_path: Path) -> None:
    trainer = LoraTrainer(_config(tmp_path), dry_run=True)
    result = trainer.train()
    assert result.method == "lora_dry_run"
    assert result.metadata["actual_training_executed"] is False
    assert result.metadata["no_model_download"] is True
    assert (tmp_path / "lora" / "lora_training_plan.json").exists()
    assert (tmp_path / "lora" / "checkpoints" / "checkpoint_manifest.json").exists()


def test_lora_trainer_validates_required_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["target_modules"] = []
    with pytest.raises(ValueError, match="missing LoRA config fields"):
        LoraTrainer(config).validate_config()


def test_lora_trainer_blocks_actual_training(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="dry-run"):
        LoraTrainer(_config(tmp_path), dry_run=False).train()
