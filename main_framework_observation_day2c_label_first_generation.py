from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import (
    _compact_json,
    _extract_json_text,
    _read_jsonl,
)
from main_framework_observation_day2_generative_recommendation_analysis import (
    _load_catalog,
    auroc,
    brier,
    ece,
    explanatory_text_after_json,
    ground_prediction,
    is_placeholder_title,
    normalize_title,
)
from main_framework_observation_day2_generative_recommendation_infer import (
    _append_jsonl,
    _candidate_payload,
    _existing_ids,
    _history_payload,
    _load_transformers,
    _load_vllm,
    _lora_request,
    _sample_id,
    _select_user_pools,
    _target_row,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_base_generative_label_first.yaml"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _label_map(candidate_pool: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {labels[idx]: cand for idx, cand in enumerate(candidate_pool) if idx < len(labels)}


def format_label_prompt(pool: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    template = Path(str(cfg["prompt_template"])).read_text(encoding="utf-8").strip()
    target = _target_row(pool)
    payload = {
        "user_history": _history_payload(target, cfg),
        "candidate_pool": _candidate_payload(pool, cfg),
    }
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


def parse_label_response(raw_text: str) -> dict[str, Any]:
    json_text = _extract_json_text(raw_text)
    if json_text is None:
        return {
            "parse_success": False,
            "schema_valid": False,
            "selected_label": "",
            "recommended_title": "",
            "confidence": None,
            "json_truncation": looks_like_json_truncation(raw_text),
            "parse_error": "no_json_object",
        }
    try:
        obj = json.loads(json_text)
    except Exception as exc:
        return {
            "parse_success": False,
            "schema_valid": False,
            "selected_label": "",
            "recommended_title": "",
            "confidence": None,
            "json_truncation": looks_like_json_truncation(raw_text),
            "parse_error": f"json_error:{type(exc).__name__}",
        }
    selected_label = str(obj.get("selected_label", "")).strip().upper()
    title = str(obj.get("recommended_title", "")).strip()
    try:
        confidence = float(obj.get("confidence"))
    except Exception:
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    schema_valid = bool(re.fullmatch(r"[A-F]", selected_label)) and bool(title) and confidence is not None
    return {
        "parse_success": True,
        "schema_valid": schema_valid,
        "selected_label": selected_label,
        "recommended_title": title,
        "confidence": confidence,
        "json_truncation": False,
        "parse_error": "" if schema_valid else "missing_or_invalid_label_title_confidence",
    }


def looks_like_json_truncation(raw_text: Any) -> bool:
    text = str(raw_text or "").strip()
    if not text:
        return False
    if not text.startswith("{"):
        return False
    if text.count("{") > text.count("}"):
        return True
    if text.endswith('"confidence":') or text.endswith('"recommended_title":'):
        return True
    if text.count('"') % 2 == 1:
        return True
    return False


def _label_diagnostics(parsed: dict[str, Any], pool: list[dict[str, Any]], raw_text: str) -> dict[str, Any]:
    labels = _label_map(pool)
    selected = parsed.get("selected_label", "")
    title = str(parsed.get("recommended_title", "")).strip()
    selected_row = labels.get(str(selected))
    label_valid = selected_row is not None
    expected_title = str(selected_row.get("candidate_title", "")) if selected_row else ""
    title_matches_selected_label = label_valid and title == expected_title
    candidate_title_exact_match = any(title == str(row.get("candidate_title", "")) for row in pool)
    generation_valid = (
        bool(parsed.get("schema_valid"))
        and label_valid
        and title_matches_selected_label
        and candidate_title_exact_match
        and not is_placeholder_title(title)
    )
    return {
        "generation_valid": generation_valid,
        "label_valid": label_valid,
        "title_matches_selected_label": title_matches_selected_label,
        "candidate_title_exact_match": candidate_title_exact_match,
        "placeholder_title": is_placeholder_title(title),
        "explanatory_text_after_json": explanatory_text_after_json(raw_text),
        "selected_label_expected_title": expected_title,
    }


def _prediction_row(
    split: str,
    pool: list[dict[str, Any]],
    prompt: str,
    raw_text: str,
    parsed: dict[str, Any],
    model_variant: str,
    backend: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    target = _target_row(pool)
    label_map = _label_map(pool)
    candidate_pool = []
    for idx, row in enumerate(pool):
        label = chr(ord("A") + idx)
        candidate_pool.append(
            {
                "candidate_label": label,
                "candidate_item_id": str(row.get("candidate_item_id", "")),
                "candidate_title": str(row.get("candidate_title", "")),
                "label": int(row.get("label", 0)),
                "input_position": idx + 1,
            }
        )
    diagnostics = _label_diagnostics(parsed, pool, raw_text)
    selected_row = label_map.get(str(parsed.get("selected_label", "")))
    return {
        "sample_id": _sample_id(split, pool),
        "domain": target.get("domain", cfg.get("domain", "beauty")),
        "split": split,
        "setting": "label_first_candidate_grounded",
        "model_variant": model_variant,
        "backend": backend,
        "user_id": str(target.get("user_id", "")),
        "history": target.get("history", []),
        "candidate_labels": [row["candidate_label"] for row in candidate_pool],
        "candidate_pool": candidate_pool,
        "target_item_id": str(target.get("candidate_item_id", "")),
        "target_title": str(target.get("candidate_title", "")),
        "raw_response": raw_text,
        "selected_label": parsed["selected_label"],
        "selected_label_item_id": str(selected_row.get("candidate_item_id", "")) if selected_row else "",
        "recommended_title": parsed["recommended_title"],
        "confidence": parsed["confidence"],
        "parse_success": parsed["parse_success"],
        "schema_valid": parsed["schema_valid"],
        "generation_valid": diagnostics["generation_valid"],
        "label_valid": diagnostics["label_valid"],
        "title_matches_selected_label": diagnostics["title_matches_selected_label"],
        "candidate_title_exact_match": diagnostics["candidate_title_exact_match"],
        "placeholder_title": diagnostics["placeholder_title"],
        "explanatory_text_after_json": diagnostics["explanatory_text_after_json"],
        "json_truncation": bool(parsed.get("json_truncation", False)),
        "parse_error": parsed["parse_error"],
        "prompt_token_proxy_chars": len(prompt),
        "candidate_order_seeded_shuffle": bool(cfg.get("shuffle_candidates", True)),
    }


def run_vllm(cfg: dict[str, Any], split: str, model_variant: str, max_users: int | None, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore

    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    prompt_rows = []
    for pool in _select_user_pools(cfg, split, max_users):
        sid = _sample_id(split, pool)
        if sid not in finished:
            prompt_rows.append((pool, format_label_prompt(pool, cfg)))
    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    sampling_params = SamplingParams(
        max_tokens=int(cfg.get("max_new_tokens", 64)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        seed=int(cfg.get("seed", 42)),
        stop=cfg.get("vllm_stop") or None,
    )
    batch_size = int(cfg.get("vllm_batch_size", 12))
    for start in range(0, len(prompt_rows), batch_size):
        batch = prompt_rows[start : start + batch_size]
        outputs = llm.generate([prompt for _, prompt in batch], sampling_params, lora_request=lora_request)
        rows = []
        for (pool, prompt), output in zip(batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed = parse_label_response(raw_text)
            rows.append(_prediction_row(split, pool, prompt, raw_text, parsed, model_variant, "vllm", cfg))
        _append_jsonl(output_path, rows)
    return output_path


def run_transformers(cfg: dict[str, Any], split: str, model_variant: str, max_users: int | None, resume: bool) -> Path:
    import torch  # type: ignore

    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    prompt_rows = []
    for pool in _select_user_pools(cfg, split, max_users):
        sid = _sample_id(split, pool)
        if sid not in finished:
            prompt_rows.append((pool, format_label_prompt(pool, cfg)))
    model, tokenizer = _load_transformers(cfg, model_variant)
    batch_size = int(cfg.get("transformers_batch_size", 1))
    for start in range(0, len(prompt_rows), batch_size):
        batch = prompt_rows[start : start + batch_size]
        prompts = [prompt for _, prompt in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(cfg.get("max_new_tokens", 64)),
                do_sample=bool(cfg.get("do_sample", False)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        rows = []
        for i, (pool, prompt) in enumerate(batch):
            prompt_len = int(inputs["attention_mask"][i].sum().item())
            raw_text = tokenizer.decode(generated[i][prompt_len:], skip_special_tokens=True)
            parsed = parse_label_response(raw_text)
            rows.append(_prediction_row(split, pool, prompt, raw_text, parsed, model_variant, "transformers", cfg))
        _append_jsonl(output_path, rows)
    return output_path


def run_inference(cfg: dict[str, Any], split: str, model_variant: str, backend: str, max_users: int | None, resume: bool) -> Path:
    if backend == "vllm":
        return run_vllm(cfg, split, model_variant, max_users, resume)
    if backend == "transformers":
        return run_transformers(cfg, split, model_variant, max_users, resume)
    raise ValueError(f"Unsupported backend: {backend}")


def _enrich(row: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    grounded = ground_prediction(row, catalog)
    label_map = {str(c.get("candidate_label", "")): c for c in row.get("candidate_pool", [])}
    selected = str(row.get("selected_label", "")).strip().upper()
    selected_row = label_map.get(selected)
    title = str(row.get("recommended_title", "")).strip()
    label_valid = selected_row is not None
    title_matches_selected_label = label_valid and title == str(selected_row.get("candidate_title", ""))
    generation_valid = (
        bool(row.get("schema_valid"))
        and label_valid
        and title_matches_selected_label
        and bool(grounded.get("is_valid_catalog_item"))
        and not is_placeholder_title(title)
    )
    selected_label_hit = label_valid and str(selected_row.get("candidate_item_id", "")) == str(row.get("target_item_id", ""))
    return {
        **row,
        **grounded,
        "label_valid": label_valid,
        "title_matches_selected_label": title_matches_selected_label,
        "generation_valid": generation_valid,
        "selected_label_hit": selected_label_hit,
        "matched_title_hit": bool(grounded.get("hit_target")),
    }


def _summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    labels = [1 if row.get("matched_title_hit") else 0 for row in rows]
    scores = [float(row.get("confidence") or 0.0) for row in rows]
    selected_labels = [str(row.get("selected_label", "")) or "INVALID" for row in rows]
    counts = Counter(selected_labels)
    return {
        "split": split,
        "num_users": len(rows),
        "parse_success_rate": _safe_mean([1.0 if row.get("parse_success") else 0.0 for row in rows]),
        "schema_valid_rate": _safe_mean([1.0 if row.get("schema_valid") else 0.0 for row in rows]),
        "generation_valid_rate": _safe_mean([1.0 if row.get("generation_valid") else 0.0 for row in rows]),
        "label_valid_rate": _safe_mean([1.0 if row.get("label_valid") else 0.0 for row in rows]),
        "title_matches_selected_label_rate": _safe_mean([1.0 if row.get("title_matches_selected_label") else 0.0 for row in rows]),
        "candidate_title_exact_match_rate": _safe_mean([1.0 if row.get("candidate_title_exact_match") else 0.0 for row in rows]),
        "catalog_match_rate": _safe_mean([1.0 if row.get("is_valid_catalog_item") else 0.0 for row in rows]),
        "matched_title_hit_rate": _safe_mean([1.0 if row.get("matched_title_hit") else 0.0 for row in rows]),
        "selected_label_hit_rate": _safe_mean([1.0 if row.get("selected_label_hit") else 0.0 for row in rows]),
        "hallucination_rate": _safe_mean([1.0 if row.get("hallucination") else 0.0 for row in rows]),
        "placeholder_title_rate": _safe_mean([1.0 if row.get("placeholder_title") else 0.0 for row in rows]),
        "json_truncation_rate": _safe_mean([1.0 if row.get("json_truncation") else 0.0 for row in rows]),
        "explanatory_text_after_json_rate": _safe_mean([1.0 if row.get("explanatory_text_after_json") else 0.0 for row in rows]),
        "selected_label_distribution": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "confidence_mean": _safe_mean(scores),
        "confidence_std": _safe_std(scores),
        "confidence_unique_count": len(set(round(score, 6) for score in scores)),
        "confidence_AUROC_for_hit": auroc(scores, labels),
        "confidence_ECE_for_hit": ece(scores, labels),
        "confidence_Brier_for_hit": brier(scores, labels),
        "high_conf_wrong_rate": _safe_mean(
            [1.0 if float(row.get("confidence") or 0.0) >= 0.9 and not row.get("matched_title_hit") else 0.0 for row in rows]
        ),
    }


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _comparison(day2c_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    day2_path = _first_existing([
        Path("data_done/framework_observation_day2_generative_candidate_grounded_diagnostics.csv"),
        Path(".tmp_day2_analysis/data_done/framework_observation_day2_generative_candidate_grounded_diagnostics.csv"),
    ])
    day2b_path = _first_existing([
        Path("data_done/framework_observation_day2b_generative_candidate_grounded_repair_diagnostics.csv"),
        Path(".tmp_day2b_analysis/data_done/framework_observation_day2b_generative_candidate_grounded_repair_diagnostics.csv"),
    ])
    prior_specs = [
        ("day2_placeholder_schema", "recommended_title+confidence", "NA", "recommended_title,confidence", day2_path, "Day2 placeholder/schema failure; parse/schema rates are superficial."),
        ("day2b_no_placeholder_exact_title", "recommended_title+confidence", "96", "recommended_title,confidence", day2b_path, "Day2b fixed validity but retained explanatory text and unusable confidence."),
    ]
    for prompt_version, schema, max_tokens, field_order, path, note in prior_specs:
        if not path:
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        row = next((r for r in rows if r.get("split") == "test"), rows[-1])
        out.append(
            {
                "method": "base_qwen_candidate_grounded",
                "prompt_version": prompt_version,
                "output_schema": schema,
                "max_new_tokens": max_tokens,
                "field_order": field_order,
                "num_users": row.get("num_users", "NA"),
                "parse_success_rate": row.get("parse_success_rate", "NA"),
                "schema_valid_rate": row.get("schema_valid_rate", "NA"),
                "generation_valid_rate": row.get("generation_valid_rate", row.get("catalog_match_rate", "NA")),
                "placeholder_title_rate": row.get("placeholder_title_rate", "NA"),
                "json_truncation_rate": row.get("json_truncation_rate", "NA"),
                "label_valid_rate": "NA",
                "title_matches_selected_label_rate": "NA",
                "candidate_title_exact_match_rate": row.get("candidate_title_exact_match_rate", row.get("valid_candidate_title_rate", "NA")),
                "catalog_match_rate": row.get("catalog_match_rate", "NA"),
                "matched_title_hit_rate": row.get("matched_title_hit_rate", row.get("HR@1", "NA")),
                "hallucination_rate": row.get("hallucination_rate", "NA"),
                "explanatory_text_after_json_rate": row.get("explanatory_text_after_json_rate", "NA"),
                "confidence_mean": row.get("confidence_mean", "NA"),
                "confidence_std": row.get("confidence_std", "NA"),
                "confidence_AUROC": row.get("confidence_AUROC_for_hit", row.get("AUROC_for_generation_correctness", "NA")),
                "confidence_ECE": row.get("confidence_ECE_for_hit", row.get("ECE_for_generation_correctness", "NA")),
                "interpretation": note,
                "recommendation": "do_not_full_run",
            }
        )
    if day2c_rows:
        row = next((r for r in day2c_rows if r.get("split") == "test"), day2c_rows[-1])
        conf_bad = float(row.get("confidence_AUROC_for_hit", 0.5)) < 0.55 or float(row.get("confidence_ECE_for_hit", 1.0)) > 0.2
        rec = "move_to_non_verbal_uncertainty" if conf_bad else "audit_calibrated_confidence"
        out.append(
            {
                "method": "base_qwen_candidate_grounded",
                "prompt_version": str(cfg.get("prompt_version", "day2c_label_first")),
                "output_schema": "selected_label+recommended_title+confidence",
                "max_new_tokens": cfg.get("max_new_tokens", "NA"),
                "field_order": str(cfg.get("field_order", "selected_label,recommended_title,confidence")),
                "num_users": row.get("num_users", "NA"),
                "parse_success_rate": row.get("parse_success_rate", "NA"),
                "schema_valid_rate": row.get("schema_valid_rate", "NA"),
                "generation_valid_rate": row.get("generation_valid_rate", "NA"),
                "placeholder_title_rate": row.get("placeholder_title_rate", "NA"),
                "json_truncation_rate": row.get("json_truncation_rate", "NA"),
                "label_valid_rate": row.get("label_valid_rate", "NA"),
                "title_matches_selected_label_rate": row.get("title_matches_selected_label_rate", "NA"),
                "candidate_title_exact_match_rate": row.get("candidate_title_exact_match_rate", "NA"),
                "catalog_match_rate": row.get("catalog_match_rate", "NA"),
                "matched_title_hit_rate": row.get("matched_title_hit_rate", "NA"),
                "hallucination_rate": row.get("hallucination_rate", "NA"),
                "explanatory_text_after_json_rate": row.get("explanatory_text_after_json_rate", "NA"),
                "confidence_mean": row.get("confidence_mean", "NA"),
                "confidence_std": row.get("confidence_std", "NA"),
                "confidence_AUROC": row.get("confidence_AUROC_for_hit", "NA"),
                "confidence_ECE": row.get("confidence_ECE_for_hit", "NA"),
                "interpretation": "Label-first succeeds if validity and title-label agreement approach 1.0 and explanatory text drops.",
                "recommendation": rec,
            }
        )
    return out


def _write_report(path: Path, rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    is_repair = "repair" in str(path).lower()
    title = (
        "# Framework-Observation-Day2c-Repair Label-First Generation Report"
        if is_repair
        else "# Framework-Observation-Day2c Label-First Generation Report"
    )
    lines = [
        title,
        "",
        "Status: observation only. This is not training, evidence, CEP, external API use, open-title generation, or a full run.",
        "",
        "## Framing",
        "",
        "Day2 exposed placeholder generation failure. Day2b fixed candidate-title validity but still had explanatory text and unusable confidence. Day2c fixed explanatory tails but failed due to long-title truncation. Day2c-repair tests whether increasing token budget and compact field order can make label-first title generation output-stable.",
        "",
        "## Diagnostics",
        "",
    ]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend(["", "## Control Comparison", ""])
    if comparison_rows:
        headers = list(comparison_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in comparison_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If generation validity and title-label agreement are near 1.0 and explanatory text falls, output control is clean.",
            "- If matched-title hit rate remains around Day2b, base Qwen candidate-grounded recommendation ability is modest; do not overclaim.",
            "- If confidence remains low-variance, AUROC near 0.5, or ECE high, raw verbalized confidence remains unusable.",
            "- If confidence is unusable, move to selected-label logprob, title logprob, retrieval margin, self-consistency title agreement, or label-selection entropy.",
            "- Day2d non-verbal uncertainty remains paused until parse/schema/generation/title-label validity are >= 0.95, JSON truncation <= 0.05, and explanatory text after JSON <= 0.05.",
            "- If output control passes but matched-title hit rate remains 0.15-0.20, candidate-grounded generation is controllable but base Qwen recommendation choice is modest.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(cfg: dict[str, Any]) -> dict[str, str]:
    pred_dir = Path(str(cfg["output_dir"]))
    catalog = _load_catalog(Path(str(cfg.get("catalog_file", ""))))
    diag_rows = []
    for split in ["valid", "test"]:
        path = pred_dir / f"{split}_raw.jsonl"
        if not path.exists():
            continue
        rows = [_enrich(row, catalog) for row in _read_jsonl(path)]
        diag_rows.append(_summarize(rows, split))
    diag_path = Path(str(cfg.get("diagnostics_csv", "data_done/framework_observation_day2c_label_first_generation_diagnostics.csv")))
    report_path = Path(str(cfg.get("report_md", "data_done/framework_observation_day2c_label_first_generation_report.md")))
    comparison_path = Path(str(cfg.get("comparison_csv", "data_done/framework_observation_day2c_generative_control_comparison.csv")))
    comparison_rows = _comparison(diag_rows, cfg)
    _write_csv(diag_path, diag_rows)
    _write_csv(comparison_path, comparison_rows)
    _write_report(report_path, diag_rows, comparison_rows)
    return {
        "diagnostics": str(diag_path),
        "report": str(report_path),
        "comparison": str(comparison_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run_inference", choices=["valid", "test"], default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default=None)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    model_variant = args.model_variant or str(cfg.get("model_variant", "base"))
    backend = args.backend or str(cfg.get("backend", "vllm"))
    result: dict[str, Any] = {}
    if args.run_inference:
        path = run_inference(cfg, args.run_inference, model_variant, backend, args.max_users, args.resume)
        result["output_path"] = str(path)
        result["split"] = args.run_inference
        result["model_variant"] = model_variant
        result["backend"] = backend
    if args.analyze_only or not args.run_inference:
        result.update(analyze(cfg))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
