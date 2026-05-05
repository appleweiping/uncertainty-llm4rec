#!/usr/bin/env python3
"""Import external per-candidate scores into TRUCE prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.evaluator import evaluate_predictions  # noqa: E402
from llm4rec.experiments.config import load_config  # noqa: E402
from llm4rec.external_baselines.prediction_import import import_scored_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--source-project", default="external")
    parser.add_argument("--model-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", help="optional split filter, e.g. test")
    args = parser.parse_args()
    if args.config is not None:
        result = _import_from_config(args.config)
        print(json.dumps(result, indent=2))
        return 0
    if args.scores is None or args.examples is None or args.output is None or args.method is None or args.model_name is None:
        raise SystemExit("--scores, --examples, --output, --method, and --model-name are required unless --config is used")
    result = import_scored_candidates(
        scores_path=args.scores,
        examples_path=args.examples,
        output_path=args.output,
        method=args.method,
        source_project=args.source_project,
        model_name=args.model_name,
        seed=args.seed,
        split=args.split,
    )
    print(json.dumps(result, indent=2))
    return 0


def _import_from_config(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    baseline = config.get("external_baseline") if isinstance(config.get("external_baseline"), dict) else {}
    candidate = config.get("candidate") if isinstance(config.get("candidate"), dict) else {}
    name = str(baseline.get("name") or "")
    dataset_name = str(dataset.get("name") or baseline.get("dataset_name") or config.get("run_name"))
    seed = int(config.get("seed") or 0)
    run_dir = ROOT / "outputs" / "runs" / f"{dataset_name}_{name}_seed{seed}"
    manifest_path = run_dir / "external_baseline_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"external baseline manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = run_dir / "artifacts" / "candidate_scores.csv"
    examples = ROOT / str(manifest["export_manifest"]["examples"])
    output = run_dir / "predictions.jsonl"
    result = import_scored_candidates(
        scores_path=scores,
        examples_path=examples,
        output_path=output,
        method=name,
        source_project="RecBole",
        model_name=str(manifest.get("model_name") or baseline.get("name") or ""),
        training_config=baseline.get("training") if isinstance(baseline.get("training"), dict) else {},
        checkpoint_path=str(manifest.get("checkpoint_path") or ""),
        seed=seed,
        candidate_protocol=candidate,
        split=str(baseline.get("score_split") or "test"),
    )
    metrics = evaluate_predictions(predictions_jsonl=output, output_dir=run_dir, top_k=[1, 5, 10])
    _write_cost_latency(run_dir, manifest=manifest, result=result, dataset_name=dataset_name, method=name, seed=seed)
    manifest["predictions"] = result["predictions"]
    manifest["metrics"] = str(run_dir / "metrics.json")
    manifest["truce_evaluator_used"] = True
    manifest["status"] = "completed"
    manifest["reason"] = ""
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {"run_dir": str(run_dir), "import": result, "metrics": metrics}


def _write_cost_latency(
    run_dir: Path,
    *,
    manifest: dict[str, object],
    result: dict[str, object],
    dataset_name: str,
    method: str,
    seed: int,
) -> None:
    prediction_count = int(result.get("count") or 0)
    training_seconds = float(manifest.get("training_seconds") or 0.0)
    scoring_seconds = float(manifest.get("scoring_seconds") or 0.0)
    payload = {
        "dataset": dataset_name,
        "method": method,
        "model_name": str(manifest.get("model_name") or ""),
        "library": "RecBole",
        "external_baseline": True,
        "seed": int(seed),
        "prediction_count": prediction_count,
        "training_seconds": training_seconds,
        "scoring_seconds": scoring_seconds,
        "total_external_seconds": training_seconds + scoring_seconds,
        "latency_mean_seconds_per_example": (scoring_seconds / prediction_count) if prediction_count else 0.0,
        "token_count": 0,
        "api_cost": 0.0,
        "cost_per_200_examples": 0.0,
        "truce_evaluator_used": True,
    }
    (run_dir / "cost_latency.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
