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
REPORT_MD = Path("data_done/framework_observation_day1_beauty_confidence_report.md")


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


def write_report(diag_rows: list[dict[str, Any]], cal_rows: list[dict[str, Any]], pred_dir: Path) -> None:
    test_raw = next((r for r in diag_rows if r["split"] == "test" and r["score_type"] == "raw_confidence"), {})
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

## Calibration

- best available calibrated score: `{best_cal.get('score_type', 'NA')}`
- best calibrated ECE: `{best_cal.get('ECE', 'NA')}`
- best calibrated Brier: `{best_cal.get('Brier', 'NA')}`

Calibration is fit on valid and evaluated on test. Raw confidence is verbalized confidence, not a calibrated probability.

Metrics calibrate confidence as confidence in the model's binary decision being correct (`recommend == label`), not as direct `P(relevant)`.

## Relation To Week1-Week4

This stage is a cleaner local-framework continuation of the earlier confidence observation. If raw confidence is informative but miscalibrated, it supports bringing confidence calibration into the later framework design.

## Recommendation

If parse/schema are stable and calibration reduces ECE/Brier, proceed to four-domain local confidence observation. If parsing is weak, repair the confidence prompt/parser before scaling.
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Framework-Observation-Day1 raw confidence predictions.")
    parser.add_argument("--pred_dir", default=str(PRED_DIR))
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    valid_rows = _read_jsonl(pred_dir / "valid_raw.jsonl")
    test_rows = _read_jsonl(pred_dir / "test_raw.jsonl")
    diag_rows = [
        metric_row("valid", "raw_confidence", valid_rows),
        metric_row("test", "raw_confidence", test_rows),
    ]
    cal_rows = calibration_rows(valid_rows, test_rows) if valid_rows and test_rows else [
        {"split": "test", "score_type": "raw_confidence", "status": "missing_predictions", "fallback_reason": "valid_or_test_prediction_missing"}
    ]
    _write_csv(DIAG_CSV, diag_rows)
    _write_csv(CAL_CSV, cal_rows)
    write_report(diag_rows, cal_rows, pred_dir)
    print(json.dumps({"diagnostics": str(DIAG_CSV), "calibration": str(CAL_CSV), "report": str(REPORT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
