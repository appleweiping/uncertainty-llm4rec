"""LoRA/QLoRA trainer scaffold for safe Phase 4 dry runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm4rec.io.artifacts import write_json
from llm4rec.trainers.base import TrainResult
from llm4rec.trainers.checkpointing import save_checkpoint_artifacts


REQUIRED_LORA_FIELDS = {
    "base_model_name_or_path",
    "output_dir",
    "dataset",
    "peft_method",
    "rank",
    "alpha",
    "dropout",
    "target_modules",
}


class LoraTrainer:
    def __init__(self, config: dict[str, Any], *, dry_run: bool = True) -> None:
        self.config = dict(config)
        self.dry_run = bool(dry_run)

    def validate_config(self) -> dict[str, Any]:
        missing = sorted(field for field in REQUIRED_LORA_FIELDS if self.config.get(field) in (None, "", []))
        if missing:
            raise ValueError(f"missing LoRA config fields: {missing}")
        if self.config["peft_method"] not in {"lora", "qlora"}:
            raise ValueError("peft_method must be lora or qlora")
        if int(self.config["rank"]) < 1:
            raise ValueError("rank must be >= 1")
        if float(self.config["alpha"]) <= 0:
            raise ValueError("alpha must be > 0")
        dropout = float(self.config["dropout"])
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        if not isinstance(self.config["target_modules"], list) or not self.config["target_modules"]:
            raise ValueError("target_modules must be a non-empty list")
        if self.config.get("max_steps") in (None, "") and self.config.get("epochs") in (None, ""):
            raise ValueError("max_steps or epochs is required")
        return self.config

    def planned_manifest(self) -> dict[str, Any]:
        config = self.validate_config()
        return {
            "trainer": "LoraTrainer",
            "phase": "phase4_lora_dry_run_scaffold",
            "dry_run": self.dry_run,
            "base_model_name_or_path": str(config["base_model_name_or_path"]),
            "dataset": config["dataset"],
            "peft_method": str(config["peft_method"]),
            "rank": int(config["rank"]),
            "alpha": float(config["alpha"]),
            "dropout": float(config["dropout"]),
            "target_modules": [str(value) for value in config["target_modules"]],
            "max_steps": config.get("max_steps"),
            "epochs": config.get("epochs"),
            "no_model_download": True,
            "no_gpu_required": True,
            "actual_training_executed": False,
        }

    def train(self) -> TrainResult:
        manifest = self.planned_manifest()
        output_dir = Path(str(self.config["output_dir"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "lora_training_plan.json", manifest)
        checkpoint_dir = output_dir / "checkpoints"
        self.save_checkpoint(checkpoint_dir)
        if not self.dry_run:
            raise NotImplementedError(
                "Phase 4 only supports LoRA/QLoRA dry-run manifests; actual local training is intentionally disabled."
            )
        return TrainResult(
            method="lora_dry_run",
            artifact_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            metadata=manifest,
        )

    def evaluate(self) -> dict[str, Any]:
        return {"method": "lora_dry_run", "trainer_local_eval": False}

    def predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []

    def fit_predict(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.train()
        return []

    def save_checkpoint(self, path: str | Path) -> None:
        manifest = self.planned_manifest()
        save_checkpoint_artifacts(
            path,
            method="lora_dry_run",
            model_state={
                "method": "lora_dry_run",
                "actual_training_executed": False,
                "planned_training": manifest,
            },
            config=self.config,
            metadata=manifest,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        manifest_path = Path(path) / "checkpoint_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"LoRA dry-run manifest not found: {manifest_path}")
