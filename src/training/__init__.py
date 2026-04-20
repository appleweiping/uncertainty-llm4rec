from __future__ import annotations

from src.training.framework_artifacts import update_framework_manifest, write_compare_markdown

__all__ = ["run_lora_rank_training", "update_framework_manifest", "write_compare_markdown"]


def __getattr__(name: str):
    if name == "run_lora_rank_training":
        from src.training.lora_rank_trainer import run_lora_rank_training

        return run_lora_rank_training
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")
