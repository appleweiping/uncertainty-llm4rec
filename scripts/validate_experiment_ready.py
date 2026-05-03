"""Validate that an experiment config is ready for planning or explicit launch."""

from __future__ import annotations

import argparse
import json
import os
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
        "api_environment_ready": _api_environment_ready(config, errors, warnings),
        "request_budget_configured": _request_budget_configured(config, errors, warnings),
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
    if str(config.get("experiment_kind") or "") == "cu_gr_v2_preference_subgate":
        return True
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
            if key == "allow_api_calls" and _api_execution_explicitly_approved(config):
                continue
            errors.append(f"safety.{key} must be false by default")
            ok = False
    if template and not (config.get("dry_run") is True or config.get("requires_confirm") is True):
        errors.append("template must be dry_run=true or requires_confirm=true")
        ok = False
    return ok


def _api_execution_explicitly_approved(config: dict[str, Any]) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    provider = str(llm.get("provider") or llm.get("type") or "")
    return bool(
        provider == "openai_compatible"
        and config.get("dry_run") is False
        and config.get("requires_confirm") is False
        and safety.get("dry_run") is False
        and safety.get("requires_confirm") is False
        and safety.get("allow_api_calls") is True
        and safety.get("acknowledged_expensive_run") is True
        and not _is_tbd(str(llm.get("model") or ""))
        and not _is_tbd(str(llm.get("base_url") or ""))
        and not _is_tbd(str(llm.get("api_key_env") or ""))
    )


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


def _api_environment_ready(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    provider = str(llm.get("provider") or llm.get("type") or "")
    if provider != "openai_compatible":
        return True
    if safety.get("allow_api_calls") is not True:
        return True
    ok = True
    for key in ["model", "base_url", "api_key_env"]:
        value = str(llm.get(key) or "").strip()
        if _is_tbd(value):
            errors.append(f"llm.{key} must be set before approved API execution")
            ok = False
    base_url = str(llm.get("base_url") or "")
    if base_url and not _is_tbd(base_url) and not base_url.startswith(("http://", "https://")):
        errors.append("llm.base_url must start with http:// or https://")
        ok = False
    api_key_env = str(llm.get("api_key_env") or "").strip()
    if api_key_env and not _is_tbd(api_key_env) and not os.environ.get(api_key_env):
        errors.append(f"API key environment variable is not set: {api_key_env}")
        ok = False
    if ok:
        warnings.append("API key environment variable is present; key value was not inspected")
    return ok


def _request_budget_configured(
    config: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> bool:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    provider = str(llm.get("provider") or llm.get("type") or "")
    if provider != "openai_compatible" or safety.get("allow_api_calls") is not True:
        return True
    max_examples = _configured_max_examples(config)
    max_requests = _configured_max_requests(config)
    ok = True
    if max_examples is None:
        errors.append("approved API execution requires safety.max_examples or subset_size")
        ok = False
    if max_requests is None:
        errors.append("approved API execution requires safety.max_requests")
        ok = False
    if max_examples is None or max_requests is None:
        return False
    estimate = _estimated_real_llm_requests(config, max_examples=max_examples)
    warnings.append(f"estimated real LLM requests for configured method set: {estimate}")
    if estimate > max_requests:
        errors.append(f"estimated real LLM requests ({estimate}) exceed safety.max_requests ({max_requests})")
        ok = False
    return ok


def _configured_max_examples(config: dict[str, Any]) -> int | None:
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    values = [
        _optional_int(dataset.get("subset_size")),
        _optional_int(safety.get("subset_size")),
        _optional_int(safety.get("max_examples")),
    ]
    active = [value for value in values if value is not None]
    return min(active) if active else None


def _configured_max_requests(config: dict[str, Any]) -> int | None:
    safety = config.get("safety") if isinstance(config.get("safety"), dict) else {}
    llm = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    request_limits = llm.get("request_limits") if isinstance(llm.get("request_limits"), dict) else {}
    return _optional_int(safety.get("max_requests") or request_limits.get("max_requests"))


def _estimated_real_llm_requests(config: dict[str, Any], *, max_examples: int) -> int:
    if str(config.get("experiment_kind") or "") == "cu_gr_v2_preference_subgate":
        seeds = config.get("seeds") if isinstance(config.get("seeds"), list) and config.get("seeds") else [config.get("seed", 13)]
        n_seed = len(seeds) if isinstance(seeds, list) else 1
        return int(max_examples) * int(n_seed)
    baselines = config.get("baselines")
    methods = baselines if isinstance(baselines, list) and baselines else [config.get("method")]
    candidate_sizes = config.get("candidate_sizes")
    candidate_multiplier = len(candidate_sizes) if isinstance(candidate_sizes, list) and candidate_sizes else 1
    seeds = config.get("seeds")
    seed_multiplier = len(seeds) if isinstance(seeds, list) and seeds else 1
    return candidate_multiplier * seed_multiplier * sum(_requests_per_example(method) * max_examples for method in methods)


def _requests_per_example(method: Any) -> int:
    method_config = _method_config(method)
    name = str(method_config.get("name") or method or "")
    if name in {"llm_generative", "llm_generative_mock"}:
        return 1
    if name == "llm_rerank":
        return 1
    if name == "llm_confidence_observation":
        return 3
    if name.startswith("ours_") or str(method_config.get("type") or "") == "ours_method":
        components = method_config.get("params", {}).get("components") if isinstance(method_config.get("params"), dict) else None
        if not isinstance(components, dict):
            components = method_config.get("components") if isinstance(method_config.get("components"), dict) else {}
        if components.get("fallback_only") is True:
            return 0
        return 1 + int(components.get("candidate_normalized_confidence", True) is not False)
    return 0


def _method_config(method: Any) -> dict[str, Any]:
    if isinstance(method, dict):
        return method
    name = str(method or "")
    path = ROOT / "configs" / "methods" / f"{name}.yaml"
    if path.exists():
        return load_config(path)
    return {"name": name}


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


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
