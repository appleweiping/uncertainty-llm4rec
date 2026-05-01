from __future__ import annotations

import json
from pathlib import Path

from llm4rec.evaluation.evaluator import evaluate_predictions
from llm4rec.experiments.runner import run_all


def test_training_eval_pipeline_checkpoint_then_evaluate() -> None:
    result = run_all("configs/experiments/smoke_training_eval.yaml")
    run_dir = Path(result["run_dir"])
    assert (run_dir / "checkpoints" / "checkpoint_manifest.json").exists()
    metrics = evaluate_predictions(
        predictions_jsonl=run_dir / "predictions.jsonl",
        output_dir=run_dir,
        top_k=[1, 5, 10],
    )
    assert metrics["count"] > 0
    manifest = json.loads((run_dir / "checkpoints" / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "sequential_markov"
