#!/usr/bin/env python3
"""Import and evaluate completed Ours Qwen adapter candidate scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.evaluator import evaluate_predictions  # noqa: E402
from llm4rec.external_baselines.prediction_import import import_scored_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = import_and_evaluate(manifest=manifest, scores_path=args.scores, run_dir=args.run_dir, split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def import_and_evaluate(
    *,
    manifest: dict[str, Any],
    scores_path: Path | None = None,
    run_dir: Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    name = str(manifest.get("controlled_baseline_name") or manifest.get("method") or "ours_truce_qwen_adapter")
    output_dir = Path(str(manifest["output_dir"]))
    scores = scores_path or (output_dir / "candidate_scores.csv")
    if not scores.exists():
        raise SystemExit(f"candidate scores not found: {scores}")
    examples = Path(str(manifest["files"]["test_examples"]))
    if not examples.exists():
        raise SystemExit(f"test examples not found: {examples}")
    run = run_dir or (ROOT / "outputs" / "runs" / f"{name}_seed{int(manifest.get('seed') or 13)}")
    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    copied_scores = artifacts / "candidate_scores.csv"
    copied_scores.write_bytes(scores.read_bytes())
    predictions = run / "predictions.jsonl"
    import_result = import_scored_candidates(
        scores_path=copied_scores,
        examples_path=examples,
        output_path=predictions,
        method=name,
        source_project="TRUCE-Ours",
        model_name=f"{name}:Qwen3-8B",
        training_config=manifest.get("training") if isinstance(manifest.get("training"), dict) else {},
        checkpoint_path=str(output_dir / "adapter"),
        seed=int(manifest.get("seed") or 13),
        candidate_protocol={
            "source": "TRUCE same-candidate",
            "score_schema": "candidate_scores_csv_v1",
            "adapter_training_policy": manifest.get("adapter_training_policy", ""),
        },
        split=split,
    )
    metrics = evaluate_predictions(predictions_jsonl=predictions, output_dir=run, top_k=[1, 5, 10])
    result = {
        "method": name,
        "run_dir": str(run),
        "scores": str(copied_scores),
        "predictions": str(predictions),
        "metrics": str(run / "metrics.json"),
        "import": import_result,
        "aggregate": metrics.get("aggregate", {}),
        "count": metrics.get("count"),
        "truce_evaluator_used": True,
        "implementation_fidelity": "ours_original_framework",
        "is_paper_result": False,
    }
    (run / "ours_adapter_import_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
