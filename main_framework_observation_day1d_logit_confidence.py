from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import (
    _compact_json,
    _read_jsonl,
    _sample_id,
    _truncate_text,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_logit_confidence.yaml"
DIAG_CSV = Path("data_done/framework_observation_day1d_logit_confidence_diagnostics.csv")
CAL_CSV = Path("data_done/framework_observation_day1d_logit_confidence_calibration.csv")
REPORT_MD = Path("data_done/framework_observation_day1d_logit_confidence_report.md")
PROMPT_COMPARISON_CSV = Path("data_done/framework_observation_day1_prompt_comparison.csv")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in _read_jsonl(output_path)}


def format_logit_prompt(
    sample: dict[str, Any],
    template_path: str | Path,
    max_history_items: int,
    max_title_chars: int,
    max_history_text_chars: int,
    max_candidate_text_chars: int,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8").strip()
    history = sample.get("history", [])
    if max_history_items > 0:
        history = history[-max_history_items:]
    payload = {
        "user_history": [
            {
                "item_id": str(item.get("item_id", "")),
                "title": _truncate_text(item.get("title", ""), max_title_chars),
                "text": _truncate_text(item.get("text", ""), max_history_text_chars),
                "text_missing": bool(item.get("text_missing", False)),
            }
            for item in history
        ],
        "candidate_item": {
            "candidate_item_id": str(sample.get("candidate_item_id", "")),
            "title": _truncate_text(sample.get("candidate_title", ""), max_title_chars),
            "text": _truncate_text(sample.get("candidate_text", ""), max_candidate_text_chars),
            "candidate_text_missing": bool(sample.get("candidate_text_missing", False)),
        },
    }
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


def _extract_recommend(raw_text: str) -> bool | None:
    if not raw_text:
        return None
    match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    value = obj.get("recommend")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _load_model(cfg: dict[str, Any], model_variant: str):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(cfg.get("bf16", "true")).lower() == "true" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    if model_variant == "lora":
        from peft import PeftModel  # type: ignore

        adapter_path = str(cfg.get("adapter_path") or "")
        if not adapter_path or not Path(adapter_path).exists():
            raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.cuda()
    model.eval()
    return model, tokenizer


def _candidate_logprob(
    model: Any,
    tokenizer: Any,
    prefix: str,
    continuation: str,
    max_seq_len: int,
) -> float:
    import torch  # type: ignore

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    max_prefix_len = max_seq_len - len(continuation_ids)
    if max_prefix_len <= 0:
        raise ValueError(f"Continuation is longer than max_seq_len: {continuation!r}")
    if len(prefix_ids) > max_prefix_len:
        prefix_ids = prefix_ids[-max_prefix_len:]
    input_ids = torch.tensor([prefix_ids + continuation_ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    start = len(prefix_ids)
    total = 0.0
    for offset, token_id in enumerate(continuation_ids):
        pos = start + offset
        total += float(log_probs[pos - 1, token_id].item())
    return total


def _true_false_scores(model: Any, tokenizer: Any, prompt: str, max_seq_len: int) -> dict[str, float]:
    prefix = f'{prompt}{{"recommend": '
    logprob_true = _candidate_logprob(model, tokenizer, prefix, "true}", max_seq_len)
    logprob_false = _candidate_logprob(model, tokenizer, prefix, "false}", max_seq_len)
    max_logprob = max(logprob_true, logprob_false)
    exp_true = math.exp(logprob_true - max_logprob)
    exp_false = math.exp(logprob_false - max_logprob)
    denom = exp_true + exp_false
    p_true = exp_true / denom
    p_false = exp_false / denom
    return {
        "logprob_true": logprob_true,
        "logprob_false": logprob_false,
        "p_true": p_true,
        "p_false": p_false,
        "logit_margin": logprob_true - logprob_false,
        "prob_margin": p_true - p_false,
        "decision_confidence": max(p_true, p_false),
        "positive_relevance_score": p_true,
    }


def run_inference(cfg: dict[str, Any], split: str, model_variant: str, max_samples: int | None, resume: bool) -> Path:
    import torch  # type: ignore

    split_file = Path(str(cfg[f"{split}_file"]))
    output_dir = Path(str(cfg["output_dir"]))
    output_path = output_dir / f"{split}_raw.jsonl"
    rows = _read_jsonl(split_file, limit=max_samples)
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()

    model, tokenizer = _load_model(cfg, model_variant)
    adapter_path = str(cfg.get("adapter_path", "")) if model_variant == "lora" else ""
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    max_new_tokens = int(cfg.get("max_new_tokens", 8))
    max_history_items = int(cfg.get("max_history_items", 8))
    max_title_chars = int(cfg.get("max_title_chars", 160))
    max_history_text_chars = int(cfg.get("max_history_text_chars", 260))
    max_candidate_text_chars = int(cfg.get("max_candidate_text_chars", 360))
    pending: list[dict[str, Any]] = []

    for idx, sample in enumerate(rows):
        sid = _sample_id(split, idx, sample)
        if sid in finished:
            continue
        prompt = format_logit_prompt(
            sample,
            cfg["prompt_template"],
            max_history_items=max_history_items,
            max_title_chars=max_title_chars,
            max_history_text_chars=max_history_text_chars,
            max_candidate_text_chars=max_candidate_text_chars,
        )
        scores = _true_false_scores(model, tokenizer, prompt, max_seq_len)
        recommend = bool(scores["p_true"] >= scores["p_false"])

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to("cuda")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = generated[0][inputs["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_recommend = _extract_recommend(raw_text)
        correctness = int(recommend == (int(sample.get("label", 0)) == 1))
        pending.append(
            {
                "sample_id": sid,
                "domain": sample.get("domain", "beauty"),
                "split": split,
                "user_id": sample.get("user_id", ""),
                "candidate_item_id": sample.get("candidate_item_id", ""),
                "label": int(sample.get("label", 0)),
                "recommend": recommend,
                "generated_recommend": generated_recommend,
                "raw_response": raw_text,
                "parse_success": generated_recommend is not None,
                "schema_valid": True,
                "p_true": scores["p_true"],
                "p_false": scores["p_false"],
                "logprob_true": scores["logprob_true"],
                "logprob_false": scores["logprob_false"],
                "logit_margin": scores["logit_margin"],
                "prob_margin": scores["prob_margin"],
                "decision_confidence": scores["decision_confidence"],
                "positive_relevance_score": scores["positive_relevance_score"],
                "correctness": correctness,
                "model_variant": model_variant,
                "adapter_path": adapter_path,
                "inference_backend": "transformers",
                "confidence_source": "token_probability",
            }
        )
        if len(pending) >= 25:
            _write_jsonl(output_path, pending)
            pending = []
    _write_jsonl(output_path, pending)
    return output_path


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("schema_valid")
        and r.get("p_true") is not None
        and r.get("p_false") is not None
        and r.get("decision_confidence") is not None
    ]


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, score in enumerate(scores) if score >= lo and (score < hi or b == bins - 1)]
        if not idx:
            continue
        out += len(idx) / total * abs(mean([scores[i] for i in idx]) - mean([labels[i] for i in idx]))
    return out


def brier(scores: list[float], labels: list[int]) -> float:
    return mean([(score - label) ** 2 for score, label in zip(scores, labels)]) if scores else 0.0


def auroc(scores: list[float], labels: list[int]) -> float | str:
    pos = [score for score, label in zip(scores, labels) if label == 1]
    neg = [score for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return "NA"
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _unique_count(scores: list[float]) -> int:
    return len({round(score, 12) for score in scores})


def _metric_bundle(scores: list[float], labels: list[int]) -> dict[str, Any]:
    return {
        "ECE": ece(scores, labels),
        "Brier": brier(scores, labels),
        "AUROC": auroc(scores, labels),
        "mean": mean(scores) if scores else 0.0,
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores) if scores else "NA",
        "max": max(scores) if scores else "NA",
        "unique_count": _unique_count(scores),
    }


def diagnostics_row(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid_rows(rows)
    labels = [int(r.get("label", 0)) for r in valid]
    p_true = [float(r["positive_relevance_score"]) for r in valid]
    decisions = [1 if float(r["p_true"]) >= float(r["p_false"]) else 0 for r in valid]
    correctness = [int(r.get("correctness", int(decisions[i] == labels[i]))) for i, r in enumerate(valid)]
    decision_conf = [float(r["decision_confidence"]) for r in valid]
    relevance = _metric_bundle(p_true, labels)
    decision = _metric_bundle(decision_conf, correctness)
    high_conf_wrong = [1 for score, correct in zip(decision_conf, correctness) if score >= 0.8 and correct == 0]
    return {
        "split": split,
        "num_rows": len(rows),
        "num_valid_rows": len(valid),
        "parse_success_rate": sum(1 for r in rows if r.get("parse_success")) / len(rows) if rows else 0.0,
        "schema_valid_rate": len(valid) / len(rows) if rows else 0.0,
        "positive_label_rate": mean(labels) if labels else 0.0,
        "recommend_true_rate": mean(decisions) if decisions else 0.0,
        "accuracy_at_threshold_0.5": mean(correctness) if correctness else 0.0,
        "relevance_AUROC": relevance["AUROC"],
        "relevance_Brier": relevance["Brier"],
        "relevance_ECE": relevance["ECE"],
        "positive_relevance_score_mean": relevance["mean"],
        "positive_relevance_score_std": relevance["std"],
        "positive_relevance_score_min": relevance["min"],
        "positive_relevance_score_max": relevance["max"],
        "positive_relevance_score_unique_count": relevance["unique_count"],
        "AUROC_for_correctness": decision["AUROC"],
        "Brier_for_correctness": decision["Brier"],
        "ECE_for_correctness": decision["ECE"],
        "high_conf_error_rate": len(high_conf_wrong) / len(valid) if valid else 0.0,
        "confidence_mean": decision["mean"],
        "confidence_std": decision["std"],
        "confidence_min": decision["min"],
        "confidence_max": decision["max"],
        "confidence_unique_count": decision["unique_count"],
        "mean_logit_margin": mean([float(r["logit_margin"]) for r in valid]) if valid else 0.0,
        "std_logit_margin": pstdev([float(r["logit_margin"]) for r in valid]) if len(valid) > 1 else 0.0,
    }


def _fit_logistic(valid_scores: list[float], valid_labels: list[int]):
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(solver="lbfgs")
        model.fit([[score] for score in valid_scores], valid_labels)
        return model, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _fit_isotonic(valid_scores: list[float], valid_labels: list[int]):
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(valid_scores, valid_labels)
        return model, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"


def _predict_calibrator(model: Any, method: str, scores: list[float]) -> list[float]:
    if method == "logistic":
        return [float(x[1]) for x in model.predict_proba([[score] for score in scores])]
    if method == "isotonic":
        return [float(x) for x in model.predict(scores)]
    raise ValueError(method)


def _calibration_target_rows(
    target: str,
    valid_scores: list[float],
    valid_labels: list[int],
    test_scores: list[float],
    test_labels: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = _metric_bundle(test_scores, test_labels)
    rows.append(
        {
            "target": target,
            "split": "test",
            "score_type": "raw",
            "ECE": raw["ECE"],
            "Brier": raw["Brier"],
            "AUROC": raw["AUROC"],
            "mean_score": raw["mean"],
            "std_score": raw["std"],
            "calibration_method": "none",
            "status": "ok",
            "fallback_reason": "",
        }
    )
    if len(set(valid_labels)) < 2:
        for method in ["logistic", "isotonic"]:
            rows.append(
                {
                    "target": target,
                    "split": "test",
                    "score_type": f"{method}_calibrated",
                    "calibration_method": method,
                    "status": "fallback",
                    "fallback_reason": "valid_labels_have_single_class",
                }
            )
        return rows
    for method, fitter in [("logistic", _fit_logistic), ("isotonic", _fit_isotonic)]:
        model, err = fitter(valid_scores, valid_labels)
        if model is None:
            rows.append(
                {
                    "target": target,
                    "split": "test",
                    "score_type": f"{method}_calibrated",
                    "calibration_method": method,
                    "status": "fallback",
                    "fallback_reason": err,
                }
            )
            continue
        calibrated = _predict_calibrator(model, method, test_scores)
        metric = _metric_bundle(calibrated, test_labels)
        rows.append(
            {
                "target": target,
                "split": "test",
                "score_type": f"{method}_calibrated",
                "ECE": metric["ECE"],
                "Brier": metric["Brier"],
                "AUROC": metric["AUROC"],
                "mean_score": metric["mean"],
                "std_score": metric["std"],
                "calibration_method": method,
                "status": "ok",
                "fallback_reason": "",
                "delta_ECE_vs_raw": metric["ECE"] - raw["ECE"],
                "delta_Brier_vs_raw": metric["Brier"] - raw["Brier"],
            }
        )
    return rows


def calibration_rows(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = _valid_rows(valid_rows)
    test = _valid_rows(test_rows)
    valid_labels = [int(r.get("label", 0)) for r in valid]
    test_labels = [int(r.get("label", 0)) for r in test]
    valid_relevance_scores = [float(r["positive_relevance_score"]) for r in valid]
    test_relevance_scores = [float(r["positive_relevance_score"]) for r in test]
    valid_correctness = [
        int(r.get("correctness", int((float(r["p_true"]) >= float(r["p_false"])) == (int(r.get("label", 0)) == 1))))
        for r in valid
    ]
    test_correctness = [
        int(r.get("correctness", int((float(r["p_true"]) >= float(r["p_false"])) == (int(r.get("label", 0)) == 1))))
        for r in test
    ]
    valid_decision_scores = [float(r["decision_confidence"]) for r in valid]
    test_decision_scores = [float(r["decision_confidence"]) for r in test]
    return _calibration_target_rows(
        "positive_relevance_score_to_label",
        valid_relevance_scores,
        valid_labels,
        test_relevance_scores,
        test_labels,
    ) + _calibration_target_rows(
        "decision_confidence_to_correctness",
        valid_decision_scores,
        valid_correctness,
        test_decision_scores,
        test_correctness,
    )


def _recommendation(test_diag: dict[str, Any]) -> tuple[str, str]:
    if not test_diag or int(test_diag.get("num_valid_rows", 0)) == 0:
        return "missing_predictions", "needs_prompt_redesign"
    rel_auroc = test_diag.get("relevance_AUROC", "NA")
    corr_auroc = test_diag.get("AUROC_for_correctness", "NA")
    conf_std = float(test_diag.get("confidence_std", 0.0))
    if isinstance(rel_auroc, float) and isinstance(corr_auroc, float) and conf_std > 0.03:
        if rel_auroc > 0.55 or corr_auroc > 0.55:
            return "usable_miscalibrated_signal", "switch_to_logit_confidence"
    return "logit_signal_weak", "needs_prompt_redesign"


def write_report(diag_rows: list[dict[str, Any]], cal_rows: list[dict[str, Any]], pred_dir: Path) -> None:
    test = next((row for row in diag_rows if row["split"] == "test"), {})
    collapse_type, recommendation = _recommendation(test)
    rel_best = min(
        [r for r in cal_rows if r.get("target") == "positive_relevance_score_to_label" and r.get("status") == "ok"],
        key=lambda row: float(row.get("ECE", 999)),
        default={},
    )
    dec_best = min(
        [r for r in cal_rows if r.get("target") == "decision_confidence_to_correctness" and r.get("status") == "ok"],
        key=lambda row: float(row.get("ECE", 999)),
        default={},
    )
    report = f"""# Framework-Observation-Day1d Logit-Based Confidence Report

## Scope

Day1d is a local Qwen-LoRA Beauty 200/200 smoke. It does not train, use evidence, use CEP, call external APIs, or run four domains.

The model is prompted only for a binary `recommend` decision. Confidence is extracted from model token probabilities for `true` versus `false`, not from verbalized scalar confidence.

## Prediction Directory

`{pred_dir}`

## Relevance / Label Prediction

- test rows / valid rows: `{test.get('num_rows', 'NA')}` / `{test.get('num_valid_rows', 'NA')}`
- parse/schema: `{test.get('parse_success_rate', 'NA')}` / `{test.get('schema_valid_rate', 'NA')}`
- recommend true rate: `{test.get('recommend_true_rate', 'NA')}`
- accuracy at threshold 0.5: `{test.get('accuracy_at_threshold_0.5', 'NA')}`
- positive relevance AUROC: `{test.get('relevance_AUROC', 'NA')}`
- positive relevance Brier: `{test.get('relevance_Brier', 'NA')}`
- positive relevance ECE: `{test.get('relevance_ECE', 'NA')}`
- positive relevance score mean/std: `{test.get('positive_relevance_score_mean', 'NA')}` / `{test.get('positive_relevance_score_std', 'NA')}`

## Decision Correctness Confidence

- correctness AUROC: `{test.get('AUROC_for_correctness', 'NA')}`
- correctness Brier: `{test.get('Brier_for_correctness', 'NA')}`
- correctness ECE: `{test.get('ECE_for_correctness', 'NA')}`
- high-confidence error rate: `{test.get('high_conf_error_rate', 'NA')}`
- decision confidence mean/std/min/max: `{test.get('confidence_mean', 'NA')}` / `{test.get('confidence_std', 'NA')}` / `{test.get('confidence_min', 'NA')}` / `{test.get('confidence_max', 'NA')}`
- confidence unique count: `{test.get('confidence_unique_count', 'NA')}`

## Calibration

- best relevance calibrated score: `{rel_best.get('score_type', 'NA')}` with ECE `{rel_best.get('ECE', 'NA')}` and Brier `{rel_best.get('Brier', 'NA')}`
- best correctness calibrated score: `{dec_best.get('score_type', 'NA')}` with ECE `{dec_best.get('ECE', 'NA')}` and Brier `{dec_best.get('Brier', 'NA')}`

## Interpretation

- collapse_type: `{collapse_type}`
- recommendation: `{recommendation}`

If logit confidence has useful variance and AUROC/ECE improve over verbalized scalar confidence, keep the logit-confidence route. If the logit score is still weak, the current LoRA adapter's pointwise decision ability is insufficient; next steps should be pair/list context or self-consistency, not more scalar confidence wording.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def update_prompt_comparison(valid_diag: dict[str, Any], test_diag: dict[str, Any], cal_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    if PROMPT_COMPARISON_CSV.exists():
        with PROMPT_COMPARISON_CSV.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    for row in rows:
        if not row.get("confidence_source"):
            row["confidence_source"] = "verbalized_scalar"
        if row.get("prompt_variant") == "decision_forced":
            row["collapse_type"] = "low_variance_decision_anchored_confidence"
            row["recommendation"] = "switch_to_logit_confidence"
    collapse_type, recommendation = _recommendation(test_diag)
    best_dec = min(
        [r for r in cal_rows if r.get("target") == "decision_confidence_to_correctness" and r.get("status") == "ok"],
        key=lambda row: float(row.get("ECE", 999)),
        default={},
    )
    rows = [row for row in rows if row.get("prompt_variant") != "logit_decision_confidence"]
    rows.append(
        {
            "prompt_variant": "logit_decision_confidence",
            "backend": "transformers",
            "confidence_source": "token_probability",
            "num_valid_rows": valid_diag.get("num_valid_rows", 0),
            "num_test_rows": test_diag.get("num_valid_rows", 0),
            "parse_success_rate": test_diag.get("parse_success_rate", 0.0),
            "schema_valid_rate": test_diag.get("schema_valid_rate", 0.0),
            "recommend_true_rate": test_diag.get("recommend_true_rate", 0.0),
            "accuracy": test_diag.get("accuracy_at_threshold_0.5", 0.0),
            "raw_ECE": test_diag.get("ECE_for_correctness", 0.0),
            "calibrated_ECE": best_dec.get("ECE", "NA"),
            "Brier": test_diag.get("Brier_for_correctness", 0.0),
            "AUROC": test_diag.get("AUROC_for_correctness", "NA"),
            "confidence_mean": test_diag.get("confidence_mean", 0.0),
            "confidence_std": test_diag.get("confidence_std", 0.0),
            "confidence_unique_count": test_diag.get("confidence_unique_count", 0),
            "confidence_ge_0.9_rate": "",
            "confidence_ge_0.97_rate": "",
            "collapse_type": collapse_type,
            "recommendation": recommendation,
        }
    )
    _write_csv(PROMPT_COMPARISON_CSV, rows)


def analyze(pred_dir: Path) -> None:
    valid_rows = _read_jsonl(pred_dir / "valid_raw.jsonl")
    test_rows = _read_jsonl(pred_dir / "test_raw.jsonl")
    diag_rows = [diagnostics_row("valid", valid_rows), diagnostics_row("test", test_rows)]
    cal_rows = calibration_rows(valid_rows, test_rows) if valid_rows and test_rows else []
    _write_csv(DIAG_CSV, diag_rows)
    _write_csv(CAL_CSV, cal_rows)
    write_report(diag_rows, cal_rows, pred_dir)
    valid_diag = next((row for row in diag_rows if row["split"] == "valid"), {})
    test_diag = next((row for row in diag_rows if row["split"] == "test"), {})
    update_prompt_comparison(valid_diag, test_diag, cal_rows)
    print(
        json.dumps(
            {
                "diagnostics": str(DIAG_CSV),
                "calibration": str(CAL_CSV),
                "report": str(REPORT_MD),
                "prompt_comparison": str(PROMPT_COMPARISON_CSV),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1d logit-based confidence smoke.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["valid", "test"], default=None)
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    pred_dir = Path(str(cfg["output_dir"]))
    if args.analyze_only:
        analyze(pred_dir)
        return
    if args.split is None:
        raise ValueError("--split is required unless --analyze_only is set")
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    output_path = run_inference(cfg, args.split, variant, args.max_samples, args.resume)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "split": args.split,
                "model_variant": variant,
                "inference_backend": "transformers",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
