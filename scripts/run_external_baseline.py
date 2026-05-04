#!/usr/bin/env python3
"""Prepare and run an external baseline adapter.

The current implementation exports data and writes a RecBole config, then fails
clearly if RecBole is unavailable. It never reports external metrics as TRUCE
paper metrics.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.evaluation.evaluator import evaluate_predictions  # noqa: E402
from llm4rec.experiments.config import load_config, save_resolved_config  # noqa: E402
from llm4rec.external_baselines.data_export import export_recbole_atomic  # noqa: E402
from llm4rec.external_baselines.prediction_import import import_scored_candidates  # noqa: E402
from llm4rec.external_baselines.recbole_adapter import build_recbole_config, run_recbole_training, score_recbole_candidates, write_recbole_config  # noqa: E402
from llm4rec.external_baselines.bert4rec_adapter import bert4rec_config  # noqa: E402
from llm4rec.external_baselines.sasrec_adapter import sasrec_config  # noqa: E402
from llm4rec.external_baselines.lightgcn_adapter import lightgcn_config  # noqa: E402
from llm4rec.io.artifacts import write_environment, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="export data and write RecBole config without training")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    baseline = config.get("external_baseline") if isinstance(config.get("external_baseline"), dict) else {}
    candidate = config.get("candidate") if isinstance(config.get("candidate"), dict) else {}
    name = str(baseline.get("name") or "")
    out_dir = ROOT / str(config.get("output_dir") or "outputs/external_baselines")
    dataset_name = str(baseline.get("dataset_name") or dataset.get("name") or config.get("run_name"))
    run_dataset = str(dataset.get("name") or dataset_name)
    seed = int(config.get("seed") or 0)
    run_id = f"{run_dataset}_{name}_seed{seed}"
    run_dir = ROOT / "outputs" / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = str(dataset.get("processed_dir") or "")
    if not processed_dir:
        raise SystemExit("config.dataset.processed_dir is required")
    if name == "sasrec_recbole":
        ext_config = sasrec_config(dataset_name=dataset_name, processed_dir=processed_dir, output_dir=out_dir, seed=seed, training_config=baseline.get("training") or {}, candidate_protocol=candidate)
    elif name == "bert4rec_recbole":
        ext_config = bert4rec_config(dataset_name=dataset_name, processed_dir=processed_dir, output_dir=out_dir, seed=seed, training_config=baseline.get("training") or {}, candidate_protocol=candidate)
    elif name == "lightgcn_recbole":
        ext_config = lightgcn_config(dataset_name=dataset_name, processed_dir=processed_dir, output_dir=out_dir, seed=seed, training_config=baseline.get("training") or {}, candidate_protocol=candidate)
    else:
        raise SystemExit(f"unsupported external_baseline.name: {name}")
    save_resolved_config(config, run_dir / "resolved_config.yaml")
    write_environment(run_dir)
    write_json(run_dir / "git_info.json", _git_info())
    manifest = export_recbole_atomic(
        processed_dir=processed_dir,
        output_dir=out_dir / "recbole_data",
        dataset_name=dataset_name,
        seed=ext_config.seed,
        sasrec_max_item_list_length=int(ext_config.training_config.get("MAX_ITEM_LIST_LENGTH") or 50),
    )
    recbole_cfg = build_recbole_config(ext_config, exported_dataset_dir=manifest["exported_dir"])
    cfg_path = write_recbole_config(run_dir / "recbole_config.yaml", recbole_cfg)
    if args.prepare_only:
        manifest_path = run_dir / "external_baseline_manifest.json"
        if not _has_completed_manifest(manifest_path):
            _write_manifest(
                run_dir,
                status="adapter_prepared",
                reason="prepare-only",
                config=config,
                ext_config=ext_config,
                export_manifest=manifest,
                recbole_config_path=cfg_path,
                training={},
                scoring={},
                imported={},
                metrics={},
            )
        print(json.dumps({"status": "adapter_prepared", "run_dir": str(run_dir), "export_manifest": manifest, "recbole_config": str(cfg_path)}, indent=2))
        return 0
    try:
        training = run_recbole_training(ext_config, exported_dataset_dir=manifest["exported_dir"])
        scores_path = artifacts_dir / "candidate_scores.csv"
        scoring = score_recbole_candidates(
            config=ext_config,
            checkpoint_path=training["checkpoint_path"],
            examples_path=Path(manifest["exported_dir"]) / "truce_examples.jsonl",
            output_path=scores_path,
            split=str(baseline.get("score_split") or "test"),
            batch_size=int(baseline.get("score_batch_size") or 8192),
            rb_config=training.get("_rb_config"),
            model=training.get("_model"),
            dataset=training.get("_dataset"),
        )
        predictions_path = run_dir / "predictions.jsonl"
        imported = import_scored_candidates(
            scores_path=scores_path,
            examples_path=Path(manifest["exported_dir"]) / "truce_examples.jsonl",
            output_path=predictions_path,
            method=name,
            source_project="RecBole",
            model_name=ext_config.model_name,
            training_config=ext_config.training_config,
            checkpoint_path=training["checkpoint_path"],
            seed=seed,
            candidate_protocol=candidate,
            split=str(baseline.get("score_split") or "test"),
        )
        metrics = evaluate_predictions(predictions_jsonl=predictions_path, output_dir=run_dir, top_k=[1, 5, 10])
        _write_cost_latency(run_dir, training=training, scoring=scoring, imported=imported, config=config, ext_config=ext_config)
        status = "completed"
    except Exception as exc:
        status = "adapter_prepared_training_not_completed"
        training = {}
        scoring = {}
        imported = {}
        metrics = {}
        _append_log(run_dir / "logs.txt", f"{status}: {exc}\n")
        _write_manifest(
            run_dir,
            status=status,
            reason=str(exc),
            config=config,
            ext_config=ext_config,
            export_manifest=manifest,
            recbole_config_path=cfg_path,
            training=training,
            scoring=scoring,
            imported=imported,
            metrics=metrics,
        )
        print(json.dumps({"status": status, "reason": str(exc), "run_dir": str(run_dir), "export_manifest": manifest, "recbole_config": str(cfg_path)}, indent=2))
        return 2
    _append_log(run_dir / "logs.txt", f"completed {name} on {run_dataset} seed {seed}\n")
    _write_manifest(
        run_dir,
        status=status,
        reason="",
        config=config,
        ext_config=ext_config,
        export_manifest=manifest,
        recbole_config_path=cfg_path,
        training=training,
        scoring=scoring,
        imported=imported,
        metrics=metrics,
    )
    print(json.dumps({"status": status, "run_dir": str(run_dir), "export_manifest": manifest, "recbole_config": str(cfg_path), "scores": scoring.get("scores"), "predictions": imported.get("predictions"), "metrics": str(run_dir / "metrics.json")}, indent=2))
    return 0


def _write_manifest(
    run_dir: Path,
    *,
    status: str,
    reason: str,
    config: dict[str, object],
    ext_config,
    export_manifest: dict[str, object],
    recbole_config_path: Path,
    training: dict[str, object],
    scoring: dict[str, object],
    imported: dict[str, object],
    metrics: dict[str, object],
) -> None:
    try:
        import recbole

        recbole_version = str(recbole.__version__)
    except Exception:
        recbole_version = ""
    payload = {
        "status": status,
        "reason": reason,
        "external_baseline": True,
        "library": "RecBole",
        "recbole_version": recbole_version,
        "model_name": ext_config.model_name,
        "training_config": ext_config.training_config,
        "seed": ext_config.seed,
        "dataset": ((config.get("dataset") or {}) if isinstance(config.get("dataset"), dict) else {}).get("name"),
        "candidate_protocol": ext_config.candidate_protocol,
        "score_import_method": "per_candidate_score_csv",
        "truce_evaluator_used": bool(metrics),
        "export_manifest": export_manifest,
        "recbole_config": str(recbole_config_path),
        "checkpoint_path": str(training.get("checkpoint_path") or ""),
        "training_seconds": training.get("training_seconds"),
        "scoring_seconds": scoring.get("scoring_seconds"),
        "predictions": imported.get("predictions"),
        "metrics": str(run_dir / "metrics.json") if metrics else "",
        "environment_summary": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    write_json(run_dir / "external_baseline_manifest.json", payload)


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _write_cost_latency(
    run_dir: Path,
    *,
    training: dict[str, object],
    scoring: dict[str, object],
    imported: dict[str, object],
    config: dict[str, object],
    ext_config,
) -> None:
    prediction_count = int(imported.get("count") or 0)
    training_seconds = float(training.get("training_seconds") or 0.0)
    scoring_seconds = float(scoring.get("scoring_seconds") or 0.0)
    payload = {
        "dataset": ((config.get("dataset") or {}) if isinstance(config.get("dataset"), dict) else {}).get("name"),
        "method": ext_config.name,
        "model_name": ext_config.model_name,
        "library": "RecBole",
        "external_baseline": True,
        "seed": ext_config.seed,
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
    write_json(run_dir / "cost_latency.json", payload)


def _git_info() -> dict[str, object]:
    def run(args: list[str]) -> str:
        if shutil.which("git") is None:
            return ""
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()

    return {
        "commit": run(["rev-parse", "HEAD"]),
        "branch": run(["branch", "--show-current"]),
        "status_short": run(["status", "--short"]),
    }


def _has_completed_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(data.get("status") or "").startswith("completed")


if __name__ == "__main__":
    raise SystemExit(main())
