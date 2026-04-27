from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PRED_DIR = Path("output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions")
DIAG_CSV = Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv")
CAL_CSV = Path("data_done/framework_observation_day1_beauty_confidence_calibration.csv")
COLLAPSE_CSV = Path("data_done/framework_observation_day1_beauty_confidence_collapse_diagnostics.csv")
REPORT_MD = Path("data_done/framework_observation_day1_beauty_confidence_report.md")
DAY1C_DIAG_CSV = Path("data_done/framework_observation_day1c_decision_forced_confidence_diagnostics.csv")
DAY1C_CAL_CSV = Path("data_done/framework_observation_day1c_decision_forced_calibration.csv")
DAY1C_REPORT_MD = Path("data_done/framework_observation_day1c_decision_forced_report.md")
PROMPT_COMPARISON_CSV = Path("data_done/framework_observation_day1_prompt_comparison.csv")

ORIGINAL_PRED_DIR = Path("output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions")
REFINED_PRED_DIRS = [
    Path("output-repaired/framework_observation/beauty_qwen_lora_confidence_refined_vllm/predictions"),
    Path("output-repaired/framework_observation/beauty_qwen_lora_confidence_refined_transformers/predictions"),
]
DECISION_FORCED_PRED_DIR = Path("output-repaired/framework_observation/beauty_qwen_lora_confidence_decision_forced/predictions")
DECISION_BASIS_VALUES = ["strong_match", "weak_match", "unrelated", "insufficient_information"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("schema_valid") and r.get("confidence") is not None and r.get("recommend") is not None]


def _correctness(row: dict[str, Any]) -> int:
    return 1 if bool(row.get("recommend")) == (int(row.get("label", 0)) == 1) else 0


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (s >= lo and (s < hi or b == bins - 1))]
        if not idx:
            continue
        conf = mean([scores[i] for i in idx])
        acc = mean([labels[i] for i in idx])
        out += len(idx) / total * abs(conf - acc)
    return out


def brier(scores: list[float], labels: list[int]) -> float:
    return mean([(s - y) ** 2 for s, y in zip(scores, labels)]) if scores else 0.0


def auroc(scores: list[float], labels: list[int]) -> float | str:
    pos = [(s, y) for s, y in zip(scores, labels) if y == 1]
    neg = [(s, y) for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return "NA"
    wins = 0.0
    total = len(pos) * len(neg)
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                wins += 0.5
    return wins / total


def histogram(scores: list[float], bins: int = 10) -> str:
    counts = [0] * bins
    for s in scores:
        idx = min(bins - 1, max(0, int(s * bins)))
        counts[idx] += 1
    return json.dumps(counts)


def _safe_mean(values: list[float]) -> float | str:
    return mean(values) if values else "NA"


def _json_counts(values: list[str], keys: list[str] | None = None) -> str:
    counts: dict[str, int] = {k: 0 for k in keys or []}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return json.dumps(counts, sort_keys=True)


def _expected_confidence_level(score: float) -> str:
    if 0.50 <= score < 0.60:
        return "low"
    if 0.60 <= score < 0.75:
        return "medium"
    if 0.75 <= score < 0.90:
        return "high"
    if 0.90 <= score <= 0.97:
        return "very_high"
    return "out_of_range"


def collapse_row(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid_rows(rows)
    scores = [float(r["confidence"]) for r in valid]
    correct_flags = [_correctness(r) for r in valid]
    recommend_true = [float(r["confidence"]) for r in valid if r.get("recommend") is True]
    recommend_false = [float(r["confidence"]) for r in valid if r.get("recommend") is False]
    correct_scores = [s for s, c in zip(scores, correct_flags) if c == 1]
    wrong_scores = [s for s, c in zip(scores, correct_flags) if c == 0]
    confidence_levels = [str(r.get("confidence_level", "")).strip().lower() for r in valid if r.get("confidence_level")]
    valid_level_rows = [
        r
        for r in valid
        if str(r.get("confidence_level", "")).strip().lower() in {"low", "medium", "high", "very_high"}
    ]
    level_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
    for level in confidence_levels:
        if level in level_counts:
            level_counts[level] += 1
    consistent = 0
    for r in valid_level_rows:
        level = str(r.get("confidence_level", "")).strip().lower()
        expected = _expected_confidence_level(float(r["confidence"]))
        if level == expected:
            consistent += 1
    recommend_true_rate = sum(1 for r in valid if r.get("recommend") is True) / len(valid) if valid else 0.0
    recommend_false_rate = sum(1 for r in valid if r.get("recommend") is False) / len(valid) if valid else 0.0
    correct_mean = _safe_mean(correct_scores)
    wrong_mean = _safe_mean(wrong_scores)
    if isinstance(correct_mean, float) and isinstance(wrong_mean, float):
        gap = correct_mean - wrong_mean
    else:
        gap = "NA"
    return {
        "split": split,
        "num_rows": len(rows),
        "num_valid_rows": len(valid),
        "parse_success_rate": sum(1 for r in rows if r.get("parse_success")) / len(rows) if rows else 0.0,
        "schema_valid_rate": len(valid) / len(rows) if rows else 0.0,
        "confidence_mean": mean(scores) if scores else 0.0,
        "confidence_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "confidence_min": min(scores) if scores else "NA",
        "confidence_max": max(scores) if scores else "NA",
        "confidence_unique_count": len(set(scores)),
        "confidence_at_1_rate": sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0.0,
        "confidence_ge_0.9_rate": sum(1 for s in scores if s >= 0.9) / len(scores) if scores else 0.0,
        "confidence_ge_0.97_rate": sum(1 for s in scores if s >= 0.97) / len(scores) if scores else 0.0,
        "confidence_by_recommend_true_mean": _safe_mean(recommend_true),
        "confidence_by_recommend_false_mean": _safe_mean(recommend_false),
        "confidence_by_correct_mean": correct_mean,
        "confidence_by_wrong_mean": wrong_mean,
        "confidence_gap_correct_minus_wrong": gap,
        "recommend_true_rate": recommend_true_rate,
        "recommend_false_rate": recommend_false_rate,
        "confidence_level_valid_rate": len(valid_level_rows) / len(valid) if valid else 0.0,
        "confidence_level_distribution": json.dumps(level_counts, sort_keys=True),
        "confidence_level_score_consistency_rate": consistent / len(valid_level_rows) if valid_level_rows else "NA",
    }


def metric_row(split: str, score_type: str, rows: list[dict[str, Any]], scores: list[float] | None = None) -> dict[str, Any]:
    valid = _valid_rows(rows)
    if scores is None:
        scores = [float(r["confidence"]) for r in valid]
    labels = [_correctness(r) for r in valid]
    all_rows = rows
    rec_yes = [1 if r.get("recommend") is True else 0 for r in valid]
    gold = [int(r.get("label", 0)) for r in valid]
    high_conf_wrong = [1 for s, y in zip(scores, labels) if s >= 0.8 and y == 0]
    return {
        "split": split,
        "score_type": score_type,
        "num_rows": len(all_rows),
        "num_valid_rows": len(valid),
        "parse_success_rate": sum(1 for r in all_rows if r.get("parse_success")) / len(all_rows) if all_rows else 0.0,
        "schema_valid_rate": len(valid) / len(all_rows) if all_rows else 0.0,
        "accuracy": mean(labels) if labels else 0.0,
        "ECE": ece(scores, labels),
        "Brier": brier(scores, labels),
        "AUROC": auroc(scores, labels),
        "high_conf_error_rate": len(high_conf_wrong) / len(valid) if valid else 0.0,
        "mean_confidence": mean(scores) if scores else 0.0,
        "std_confidence": pstdev(scores) if len(scores) > 1 else 0.0,
        "confidence_histogram": histogram(scores),
        "positive_label_rate": mean(gold) if gold else 0.0,
        "recommend_yes_rate": mean(rec_yes) if rec_yes else 0.0,
    }


def decision_forced_row(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = _valid_rows(rows)
    metric = metric_row(split, "raw_confidence", rows)
    collapse = collapse_row(split, rows)
    labels = [int(r.get("label", 0)) for r in valid]
    recommend_true = [1 if r.get("recommend") is True else 0 for r in valid]
    recommend_false = [1 if r.get("recommend") is False else 0 for r in valid]
    basis_rows = [str(r.get("decision_basis", "")).strip().lower() for r in valid]
    valid_basis_rows = [b for b in basis_rows if b in DECISION_BASIS_VALUES]

    def rate_for_label(values: list[int], label_value: int) -> float | str:
        idx = [i for i, y in enumerate(labels) if y == label_value]
        return mean([values[i] for i in idx]) if idx else "NA"

    def basis_by(group_key: str) -> str:
        grouped: dict[str, dict[str, int]] = {}
        for r in valid:
            basis = str(r.get("decision_basis", "")).strip().lower()
            if basis not in DECISION_BASIS_VALUES:
                basis = "invalid_or_missing"
            if group_key == "label":
                group = str(int(r.get("label", 0)))
            elif group_key == "recommend":
                group = str(bool(r.get("recommend")))
            elif group_key == "correctness":
                group = str(_correctness(r))
            else:
                group = "all"
            grouped.setdefault(group, {})
            grouped[group][basis] = grouped[group].get(basis, 0) + 1
        return json.dumps(grouped, sort_keys=True)

    row = {
        "split": split,
        "num_rows": len(rows),
        "num_valid_rows": len(valid),
        "parse_success_rate": metric["parse_success_rate"],
        "schema_valid_rate": metric["schema_valid_rate"],
        "accuracy": metric["accuracy"],
        "ECE": metric["ECE"],
        "Brier": metric["Brier"],
        "AUROC": metric["AUROC"],
        "recommend_true_rate": collapse["recommend_true_rate"],
        "recommend_false_rate": collapse["recommend_false_rate"],
        "recommend_true_rate_label_1": rate_for_label(recommend_true, 1),
        "recommend_true_rate_label_0": rate_for_label(recommend_true, 0),
        "recommend_false_rate_label_1": rate_for_label(recommend_false, 1),
        "recommend_false_rate_label_0": rate_for_label(recommend_false, 0),
        "confidence_mean": collapse["confidence_mean"],
        "confidence_std": collapse["confidence_std"],
        "confidence_unique_count": collapse["confidence_unique_count"],
        "confidence_min": collapse["confidence_min"],
        "confidence_max": collapse["confidence_max"],
        "confidence_at_1_rate": collapse["confidence_at_1_rate"],
        "confidence_ge_0.9_rate": collapse["confidence_ge_0.9_rate"],
        "confidence_ge_0.97_rate": collapse["confidence_ge_0.97_rate"],
        "confidence_by_recommend_true_mean": collapse["confidence_by_recommend_true_mean"],
        "confidence_by_recommend_false_mean": collapse["confidence_by_recommend_false_mean"],
        "confidence_by_correct_mean": collapse["confidence_by_correct_mean"],
        "confidence_by_wrong_mean": collapse["confidence_by_wrong_mean"],
        "confidence_gap_correct_minus_wrong": collapse["confidence_gap_correct_minus_wrong"],
        "decision_basis_valid_rate": len(valid_basis_rows) / len(valid) if valid else 0.0,
        "decision_basis_distribution": _json_counts(basis_rows, DECISION_BASIS_VALUES),
        "decision_basis_by_label": basis_by("label"),
        "decision_basis_by_recommend": basis_by("recommend"),
        "decision_basis_by_correctness": basis_by("correctness"),
    }
    return row


def _fit_logistic(valid_scores: list[float], valid_labels: list[int]):
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(solver="lbfgs")
        model.fit([[s] for s in valid_scores], valid_labels)
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
        return [float(x[1]) for x in model.predict_proba([[s] for s in scores])]
    if method == "isotonic":
        return [float(x) for x in model.predict(scores)]
    raise ValueError(method)


def calibration_rows(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = _valid_rows(valid_rows)
    test = _valid_rows(test_rows)
    valid_scores = [float(r["confidence"]) for r in valid]
    valid_labels = [_correctness(r) for r in valid]
    test_scores = [float(r["confidence"]) for r in test]
    test_labels = [_correctness(r) for r in test]
    rows = []
    raw = metric_row("test", "raw_confidence", test_rows)
    raw.update({"calibration_method": "none", "status": "ok", "fallback_reason": ""})
    rows.append(raw)
    if len(set(valid_labels)) < 2:
        for method in ["logistic", "isotonic"]:
            rows.append(
                {
                    "split": "test",
                    "score_type": f"{method}_calibrated_confidence",
                    "calibration_method": method,
                    "status": "fallback",
                    "fallback_reason": "valid_correctness_has_single_class",
                }
            )
        return rows
    for method, fitter in [("logistic", _fit_logistic), ("isotonic", _fit_isotonic)]:
        model, err = fitter(valid_scores, valid_labels)
        if model is None:
            rows.append(
                {
                    "split": "test",
                    "score_type": f"{method}_calibrated_confidence",
                    "calibration_method": method,
                    "status": "fallback",
                    "fallback_reason": err,
                }
            )
            continue
        calibrated = _predict_calibrator(model, method, test_scores)
        row = metric_row("test", f"{method}_calibrated_confidence", test_rows, scores=calibrated)
        row.update({"calibration_method": method, "status": "ok", "fallback_reason": ""})
        row["delta_ECE_vs_raw"] = row["ECE"] - raw["ECE"]
        row["delta_Brier_vs_raw"] = row["Brier"] - raw["Brier"]
        rows.append(row)
    return rows


def _collapse_interpretation(row: dict[str, Any]) -> str:
    if not row or row.get("num_valid_rows", 0) == 0:
        return "pending_predictions"
    if float(row.get("confidence_at_1_rate", 0.0)) >= 0.2 or float(row.get("confidence_ge_0.97_rate", 0.0)) >= 0.5:
        return "confidence_collapse_or_saturation"
    if int(row.get("confidence_unique_count", 0)) <= 3 and int(row.get("num_valid_rows", 0)) >= 50:
        return "low_unique_confidence_values"
    return "no_obvious_saturation"


def _prompt_variant_from_pred_dir(pred_dir: Path) -> str:
    text = str(pred_dir).replace("\\", "/").lower()
    if "decision_forced" in text:
        return "decision_forced"
    if "refined" in text:
        return "refined"
    return "original"


def write_report(
    diag_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    collapse_rows: list[dict[str, Any]],
    pred_dir: Path,
) -> None:
    test_raw = next((r for r in diag_rows if r["split"] == "test" and r["score_type"] == "raw_confidence"), {})
    test_collapse = next((r for r in collapse_rows if r["split"] == "test"), {})
    best_cal = min(
        [r for r in cal_rows if r.get("status") == "ok"],
        key=lambda r: float(r.get("ECE", 999)),
        default={},
    )
    status = "pending_predictions"
    if test_raw and test_raw.get("num_rows", 0) == 0:
        status = "pending_predictions"
    elif test_raw:
        status = "ready_for_interpretation" if test_raw.get("schema_valid_rate", 0) >= 0.9 else "needs_parse_review"
    collapse_status = _collapse_interpretation(test_collapse)
    raw_ece = test_raw.get("ECE", "NA")
    best_ece = best_cal.get("ECE", "NA")
    if isinstance(raw_ece, float) and isinstance(best_ece, float) and best_ece < raw_ece:
        calibration_note = "valid-set calibration reduces ECE/Brier and makes the raw confidence signal more usable."
    else:
        calibration_note = "calibration benefit is pending or not yet established; inspect ECE/Brier after full predictions."
    report = f"""# Framework-Observation-Day1 Beauty Local Qwen-LoRA Confidence Report

## Scope

This is a local Qwen/Qwen-LoRA confidence observation on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or ranking-baseline repair.

## Prediction Directory

`{pred_dir}`

## Raw Confidence Diagnostics

- status: `{status}`
- test parse success: `{test_raw.get('parse_success_rate', 'NA')}`
- test schema valid: `{test_raw.get('schema_valid_rate', 'NA')}`
- test accuracy: `{test_raw.get('accuracy', 'NA')}`
- test ECE: `{test_raw.get('ECE', 'NA')}`
- test Brier: `{test_raw.get('Brier', 'NA')}`
- test AUROC: `{test_raw.get('AUROC', 'NA')}`
- test high-confidence error rate: `{test_raw.get('high_conf_error_rate', 'NA')}`

## Confidence Collapse / Saturation Diagnostics

- collapse status: `{collapse_status}`
- test confidence mean: `{test_collapse.get('confidence_mean', 'NA')}`
- test confidence std: `{test_collapse.get('confidence_std', 'NA')}`
- test confidence min/max: `{test_collapse.get('confidence_min', 'NA')}` / `{test_collapse.get('confidence_max', 'NA')}`
- test confidence unique count: `{test_collapse.get('confidence_unique_count', 'NA')}`
- test confidence at 1.0 rate: `{test_collapse.get('confidence_at_1_rate', 'NA')}`
- test confidence >= 0.90 rate: `{test_collapse.get('confidence_ge_0.9_rate', 'NA')}`
- test confidence >= 0.97 rate: `{test_collapse.get('confidence_ge_0.97_rate', 'NA')}`
- recommend true/false rate: `{test_collapse.get('recommend_true_rate', 'NA')}` / `{test_collapse.get('recommend_false_rate', 'NA')}`
- confidence correct/wrong mean: `{test_collapse.get('confidence_by_correct_mean', 'NA')}` / `{test_collapse.get('confidence_by_wrong_mean', 'NA')}`
- confidence gap correct-minus-wrong: `{test_collapse.get('confidence_gap_correct_minus_wrong', 'NA')}`

If confidence mass concentrates near `0.97` or `1.0`, this should be interpreted as confidence collapse/saturation, not as method success. If confidence has meaningful variance but ECE/Brier are poor, the signal is informative but miscalibrated.

## Calibration

- best available calibrated score: `{best_cal.get('score_type', 'NA')}`
- best calibrated ECE: `{best_cal.get('ECE', 'NA')}`
- best calibrated Brier: `{best_cal.get('Brier', 'NA')}`
- interpretation: {calibration_note}

Calibration is fit on valid and evaluated on test. Raw confidence is verbalized confidence, not a calibrated probability.

Metrics calibrate confidence as confidence in the model's binary decision being correct (`recommend == label`), not as direct `P(relevant)`.

## Relation To Week1-Week4

This stage is a cleaner local-framework continuation of the earlier confidence observation. If raw confidence is informative but miscalibrated, it supports bringing confidence calibration into the later framework design.

## Recommendation

If the original prompt shows confidence collapse/saturation, run the optional Day1b refined 200/200 smoke before spending more full-runtime. If parse/schema are stable, confidence is not collapsed, and calibration reduces ECE/Brier, proceed to broader local confidence observation. If parsing is weak, repair the confidence prompt/parser before scaling.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def _day1c_recommendation(test_row: dict[str, Any]) -> tuple[str, str]:
    if not test_row or test_row.get("num_valid_rows", 0) == 0:
        return "format_failure", "needs_prompt_redesign"
    rec_true = float(test_row.get("recommend_true_rate", 0.0))
    conf_std = float(test_row.get("confidence_std", 0.0))
    ece_value = float(test_row.get("ECE", 0.0))
    if rec_true > 0.9:
        return "decision_true_collapse", "needs_prompt_redesign"
    if 0.2 <= rec_true <= 0.7 and conf_std > 0.03 and ece_value > 0.05:
        return "usable_miscalibrated_signal", "run_full_beauty"
    if 0.2 <= rec_true <= 0.7 and conf_std <= 0.03:
        return "medium_constant_collapse", "switch_to_logit_confidence"
    return "format_failure", "needs_prompt_redesign"


def write_day1c_report(
    diag_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
    pred_dir: Path,
) -> None:
    test_row = next((r for r in diag_rows if r["split"] == "test"), {})
    best_cal = min(
        [r for r in cal_rows if r.get("status") == "ok"],
        key=lambda r: float(r.get("ECE", 999)),
        default={},
    )
    collapse_type, recommendation = _day1c_recommendation(test_row)
    report = f"""# Framework-Observation-Day1c Decision-Forced Confidence Report

## Scope

Day1c is a local Qwen-LoRA confidence elicitation smoke on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or training.

## Prediction Directory

`{pred_dir}`

## Decision / Confidence Diagnostics

- test rows / valid rows: `{test_row.get('num_rows', 'NA')}` / `{test_row.get('num_valid_rows', 'NA')}`
- test parse success: `{test_row.get('parse_success_rate', 'NA')}`
- test schema valid: `{test_row.get('schema_valid_rate', 'NA')}`
- test accuracy: `{test_row.get('accuracy', 'NA')}`
- test AUROC: `{test_row.get('AUROC', 'NA')}`
- test ECE: `{test_row.get('ECE', 'NA')}`
- test Brier: `{test_row.get('Brier', 'NA')}`
- recommend true / false rate: `{test_row.get('recommend_true_rate', 'NA')}` / `{test_row.get('recommend_false_rate', 'NA')}`
- confidence mean / std: `{test_row.get('confidence_mean', 'NA')}` / `{test_row.get('confidence_std', 'NA')}`
- confidence unique count: `{test_row.get('confidence_unique_count', 'NA')}`
- decision_basis valid rate: `{test_row.get('decision_basis_valid_rate', 'NA')}`
- decision_basis distribution: `{test_row.get('decision_basis_distribution', 'NA')}`

## Calibration

- best calibrated score: `{best_cal.get('score_type', 'NA')}`
- best calibrated ECE: `{best_cal.get('ECE', 'NA')}`
- best calibrated Brier: `{best_cal.get('Brier', 'NA')}`

## Interpretation

- collapse_type: `{collapse_type}`
- recommendation: `{recommendation}`

If `recommend_true_rate` remains above 0.9, decision collapse is still unresolved and Day1c should not be scaled to full Beauty. If the decision rate becomes reasonable but confidence remains nearly constant, the decision prompt may be usable but verbalized scalar confidence is not; the next route should be logit/probability confidence or self-consistency rather than more confidence wording.
"""
    DAY1C_REPORT_MD.write_text(report, encoding="utf-8")


def _read_first_existing_pred_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if (path / "test_raw.jsonl").exists() or (path / "valid_raw.jsonl").exists():
            return path
    return None


def _comparison_row(prompt_variant: str, backend: str, pred_dir: Path | None) -> dict[str, Any]:
    if pred_dir is None:
        return {
            "prompt_variant": prompt_variant,
            "backend": backend,
            "num_valid_rows": 0,
            "num_test_rows": 0,
            "parse_success_rate": 0.0,
            "schema_valid_rate": 0.0,
            "collapse_type": "missing_predictions",
            "recommendation": "needs_prompt_redesign",
        }
    valid_rows = _read_jsonl(pred_dir / "valid_raw.jsonl")
    test_rows = _read_jsonl(pred_dir / "test_raw.jsonl")
    test_metric = metric_row("test", "raw_confidence", test_rows)
    test_collapse = collapse_row("test", test_rows)
    cal_rows = calibration_rows(valid_rows, test_rows) if valid_rows and test_rows else []
    best_cal = min([r for r in cal_rows if r.get("status") == "ok"], key=lambda r: float(r.get("ECE", 999)), default={})
    collapse_type = "format_failure"
    recommendation = "needs_prompt_redesign"
    rec_true = float(test_collapse.get("recommend_true_rate", 0.0))
    conf_std = float(test_collapse.get("confidence_std", 0.0))
    if prompt_variant == "original":
        collapse_type = "high_confidence_saturation"
        recommendation = "do_not_full_run"
    elif prompt_variant == "refined":
        collapse_type = "medium_constant_collapse" if int(test_collapse.get("confidence_unique_count", 0)) <= 3 else "usable_miscalibrated_signal"
        recommendation = "do_not_full_run" if collapse_type == "medium_constant_collapse" else "run_full_beauty"
    elif prompt_variant == "decision_forced":
        collapse_type, recommendation = _day1c_recommendation(decision_forced_row("test", test_rows))
    if rec_true > 0.9 and prompt_variant != "original":
        collapse_type = "decision_true_collapse" if collapse_type != "medium_constant_collapse" else collapse_type
    return {
        "prompt_variant": prompt_variant,
        "backend": backend,
        "num_valid_rows": len(_valid_rows(valid_rows)),
        "num_test_rows": len(_valid_rows(test_rows)),
        "parse_success_rate": test_metric["parse_success_rate"],
        "schema_valid_rate": test_metric["schema_valid_rate"],
        "recommend_true_rate": test_collapse["recommend_true_rate"],
        "accuracy": test_metric["accuracy"],
        "raw_ECE": test_metric["ECE"],
        "calibrated_ECE": best_cal.get("ECE", "NA"),
        "Brier": test_metric["Brier"],
        "AUROC": test_metric["AUROC"],
        "confidence_mean": test_collapse["confidence_mean"],
        "confidence_std": test_collapse["confidence_std"],
        "confidence_unique_count": test_collapse["confidence_unique_count"],
        "confidence_ge_0.9_rate": test_collapse["confidence_ge_0.9_rate"],
        "confidence_ge_0.97_rate": test_collapse["confidence_ge_0.97_rate"],
        "collapse_type": collapse_type,
        "recommendation": recommendation,
    }


def write_prompt_comparison() -> None:
    refined_dir = _read_first_existing_pred_dir(REFINED_PRED_DIRS)
    rows = [
        _comparison_row("original", "transformers", ORIGINAL_PRED_DIR if ORIGINAL_PRED_DIR.exists() else None),
        _comparison_row("refined", "vllm" if refined_dir and "vllm" in str(refined_dir) else "transformers", refined_dir),
        _comparison_row("decision_forced", "vllm", DECISION_FORCED_PRED_DIR if DECISION_FORCED_PRED_DIR.exists() else None),
    ]
    _write_csv(PROMPT_COMPARISON_CSV, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Framework-Observation-Day1 raw confidence predictions.")
    parser.add_argument("--pred_dir", default=str(PRED_DIR))
    parser.add_argument("--variant", choices=["auto", "original", "refined", "decision_forced"], default="auto")
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    variant = _prompt_variant_from_pred_dir(pred_dir) if args.variant == "auto" else args.variant
    valid_rows = _read_jsonl(pred_dir / "valid_raw.jsonl")
    test_rows = _read_jsonl(pred_dir / "test_raw.jsonl")
    diag_rows = [
        metric_row("valid", "raw_confidence", valid_rows),
        metric_row("test", "raw_confidence", test_rows),
    ]
    collapse_rows = [
        collapse_row("valid", valid_rows),
        collapse_row("test", test_rows),
    ]
    cal_rows = calibration_rows(valid_rows, test_rows) if valid_rows and test_rows else [
        {"split": "test", "score_type": "raw_confidence", "status": "missing_predictions", "fallback_reason": "valid_or_test_prediction_missing"}
    ]
    if variant == "decision_forced":
        day1c_rows = [
            decision_forced_row("valid", valid_rows),
            decision_forced_row("test", test_rows),
        ]
        _write_csv(DAY1C_DIAG_CSV, day1c_rows)
        _write_csv(DAY1C_CAL_CSV, cal_rows)
        write_day1c_report(day1c_rows, cal_rows, pred_dir)
        write_prompt_comparison()
        payload = {
            "diagnostics": str(DAY1C_DIAG_CSV),
            "calibration": str(DAY1C_CAL_CSV),
            "report": str(DAY1C_REPORT_MD),
            "prompt_comparison": str(PROMPT_COMPARISON_CSV),
        }
    else:
        _write_csv(DIAG_CSV, diag_rows)
        _write_csv(CAL_CSV, cal_rows)
        _write_csv(COLLAPSE_CSV, collapse_rows)
        write_report(diag_rows, cal_rows, collapse_rows, pred_dir)
        write_prompt_comparison()
        payload = {
            "diagnostics": str(DIAG_CSV),
            "calibration": str(CAL_CSV),
            "collapse": str(COLLAPSE_CSV),
            "report": str(REPORT_MD),
            "prompt_comparison": str(PROMPT_COMPARISON_CSV),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
