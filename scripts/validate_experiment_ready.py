"""Validate that an experiment config is ready for planning or explicit launch."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm4rec.experiments.config import load_config  # noqa: E402


def validate_experiment_ready(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return {
            "ready": False,
            "config": str(path),
            "errors": [f"config does not exist: {path}"],
            "warnings": [],
            "checks": {},
        }

    config = load_config(path)
    checks = {
        "config_exists": True,
        "method_configured": _method_configured(config),
        "metrics_configured": _metrics_configured(config),
        "seeds_configured": _seeds_configured(config),
        "output_dir_writable": _output_dir_writable(config, errors),
        "split_strategy_configured": _split_strategy_configured(config),
        "candidate_protocol_configured": _candidate_protocol_configured(config),
        "data_protocol_configured": _data_protocol_configured(config, errors, warnings),
        "leakage_safeguards_present": _leakage_safeguards_present(config),
        "dataset_paths_ok_or_tbd": _dataset_paths_ok_or_tbd(config, errors, warnings),
        "safety_defaults_safe": _safety_defaults_safe(config, errors),
        "expensive_flags_safe": _expensive_flags_safe(config, errors, warnings),
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(f"check failed: {name}")
    return {
        "ready": not errors,
        "config": str(path),
        "template": bool(config.get("template", False)),
        "dry_run": bool(config.get("dry_run", False)),
        "requires_confirm": bool(config.get("requires_confirm", False)),
        "errors": _unique(errors),
        "warnings": _unique(warnings),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = validate_experiment_ready(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 1


def _method_configured(config: dict[str, Any]) -> bool:
    return bool(config.get("method") or config.get("baselines"))


def _metrics_configured(config: dict[str, Any]) -> bool:
    return bool(config.get("metrics") or config.get("top_k") or (config.get("evaluation") or {}).get("top_k"))


def _seeds_configured(config: dict[str, Any]) -> bool:
    seeds = config.get("seeds")
    return bool(config.get("seed") is not None or (isinstance(seeds, list) and seeds))


def _output_dir_writable(config: dict[str, Any], errors: list[str]) -> bool:
    output = config.get("output") if isinstance(config.get("output"), dict) else {}
    output_dir = Path(str(output.get("output_dir") or config.get("output_dir") or "outputs/runs"))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".write_test_", dir=output_dir, delete=True):
            pass
        return True
    except OSError as exc:
        errors.append(f"output dir is not writable: {output_dir}: {exc}")
        return False


def _leakage_safeguards_present(config: dict[str, Any]) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    candidate = config.get("candidate") if isinstance(config.get("candidate"), dict) else {}
    split_strategy = config.get("split_strategy") if isinstance(config.get("split_strategy"), dict) else {}
    return bool(
        safety.get("leakage_safeguards")
        and config.get("split")
        and split_strategy.get("future_interactions") == "forbidden"
        and split_strategy.get("history_policy") == "past_interactions_only"
        and candidate.get("protocol")
        and candidate.get("same_set_for_comparable_baselines") is True
        and candidate.get("target_policy")
        and candidate.get("target_excluding_protocol") == "diagnostic_only_not_accuracy"
    )


def _split_strategy_configured(config: dict[str, Any]) -> bool:
    split_strategy = config.get("split_strategy") if isinstance(config.get("split_strategy"), dict) else {}
    required = [
        "name",
        "timestamp_field",
        "future_interactions",
        "history_policy",
        "repeat_target_policy",
    ]
    return bool(config.get("split") and all(split_strategy.get(key) for key in required))


def _candidate_protocol_configured(config: dict[str, Any]) -> bool:
    candidate = config.get("candidate") if isinstance(config.get("candidate"), dict) else {}
    required = [
        "protocol",
        "seed",
        "set_path",
        "target_policy",
        "target_excluding_protocol",
    ]
    return bool(all(candidate.get(key) is not None and candidate.get(key) != "" for key in required))


def _data_protocol_configured(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> bool:
    data_protocol = config.get("data_protocol") if isinstance(config.get("data_protocol"), dict) else {}
    required = [
        "interaction_schema",
        "item_schema",
        "user_item_id_mapping",
        "timestamp_handling",
        "train_popularity_source",
        "domain_field",
        "candidate_set_saved_path",
    ]
    missing = [key for key in required if not data_protocol.get(key)]
    if missing:
        errors.append(f"data_protocol missing fields: {', '.join(missing)}")
        return False
    if data_protocol.get("train_popularity_source") != "train_split_only":
        errors.append("data_protocol.train_popularity_source must be train_split_only")
        return False
    if _is_tbd(str(data_protocol.get("candidate_set_saved_path", ""))):
        warnings.append("data_protocol.candidate_set_saved_path is marked TBD")
    return True


def _safety_defaults_safe(config: dict[str, Any], errors: list[str]) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    template = bool(config.get("template", False))
    if not safety:
        errors.append("safety section is missing")
        return False
    ok = True
    for key in ["allow_api_calls", "allow_download", "allow_training"]:
        if safety.get(key) is not False:
            errors.append(f"safety.{key} must be false by default")
            ok = False
    if template and not (config.get("dry_run") is True or config.get("requires_confirm") is True):
        errors.append("template must be dry_run=true or requires_confirm=true")
        ok = False
    return ok


def _dataset_paths_ok_or_tbd(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> bool:
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    if not dataset:
        errors.append("dataset section is missing")
        return False
    ok = True
    for key, value in sorted(dataset.items()):
        if not key.endswith("_path") and key not in {"config_path", "processed_dir"}:
            continue
        text = str(value or "").strip()
        if _is_tbd(text):
            warnings.append(f"dataset.{key} is marked TBD")
            continue
        if not text:
            errors.append(f"dataset.{key} is empty")
            ok = False
            continue
        if not Path(text).exists():
            errors.append(f"dataset.{key} does not exist: {text}")
            ok = False
    return ok


def _expensive_flags_safe(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    dry_or_confirm = bool(config.get("dry_run") or config.get("requires_confirm"))
    acknowledged = bool(safety.get("acknowledged_expensive_run", False))

    provider = str(llm.get("provider") or llm.get("type") or "")
    if provider == "openai_compatible" and not bool(safety.get("allow_api_calls", False)):
        warnings.append("API provider is configured but API calls are disabled by safety.allow_api_calls=false")
    if provider == "openai_compatible" and bool(safety.get("allow_api_calls", False)) and not acknowledged:
        errors.append("API calls enabled without acknowledged_expensive_run=true")

    if provider == "hf_local" and bool(llm.get("allow_download", False)) and not acknowledged:
        errors.append("HF download enabled without acknowledged_expensive_run=true")
    if provider == "hf_local" and not bool(llm.get("allow_download", False)):
        warnings.append("HF auto-download is disabled")

    training_type = str(training.get("type") or training.get("name") or "")
    real_training_requested = bool(training) and not bool(training.get("dry_run", config.get("dry_run", True)))
    if ("lora" in training_type.lower() or real_training_requested) and not bool(safety.get("allow_training", False)):
        warnings.append("training config present but safety.allow_training=false")
    if bool(safety.get("allow_training", False)) and not acknowledged:
        errors.append("training enabled without acknowledged_expensive_run=true")

    if not dry_or_confirm and not acknowledged:
        errors.append("config is neither dry_run nor requires_confirm")
    return not any(
        message
        for message in errors
        if "acknowledged_expensive_run" in message or "neither dry_run" in message
    )


def _is_tbd(value: str) -> bool:
    return value == "" or value.upper() == "TBD" or value.endswith("_TBD") or "TBD" in value


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
