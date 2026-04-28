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
from main_framework_observation_day1_local_confidence_infer import _read_jsonl


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_base_generative_candidate_grounded.yaml"
DIAG_CSV = Path("data_done/framework_observation_day2_generative_candidate_grounded_diagnostics.csv")
CAL_CSV = Path("data_done/framework_observation_day2_generative_candidate_grounded_calibration.csv")
REPORT_MD = Path("data_done/framework_observation_day2_generative_recommendation_report.md")
COMPARISON_CSV = Path("data_done/framework_observation_day2_generative_vs_binary_observation_comparison.csv")
DAY2B_COMPARE_CSV = Path("data_done/framework_observation_day2b_day2_vs_day2b_comparison.csv")

PLACEHOLDER_TITLES = {"", "...", "n/a", "na", "unknown", "none", "null"}


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None, "NA"}:
            return default
        return float(value)
    except Exception:
        return default


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def normalize_title(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_placeholder_title(text: Any) -> bool:
    value = str(text or "").strip().lower()
    normalized = normalize_title(value)
    return value in PLACEHOLDER_TITLES or normalized in PLACEHOLDER_TITLES


def explanatory_text_after_json(raw_text: Any) -> bool:
    text = str(raw_text or "").strip()
    if not text:
        return False
    start = text.find("{")
    if start < 0:
        return False
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return bool(text[i + 1 :].strip())
    return False


def _tokens(text: Any) -> set[str]:
    return {tok for tok in normalize_title(text).split() if len(tok) > 1}


def token_similarity(a: Any, b: Any) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    # A title-generation match should reward covering generated tokens while
    # still penalizing loose matches to very long product titles.
    recall = inter / len(ta)
    jaccard = inter / len(ta | tb)
    return 0.7 * recall + 0.3 * jaccard


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            title = row.get("title", "")
            rows.append(
                {
                    "item_id": str(row.get("item_id", "")),
                    "title": title,
                    "normalized_title": normalize_title(title),
                    "tokens": _tokens(title),
                }
            )
    return rows


def _candidate_ranking(pred: dict[str, Any]) -> list[dict[str, Any]]:
    generated = pred.get("recommended_title", "")
    if is_placeholder_title(generated):
        return [
            {
                "item_id": str(cand.get("candidate_item_id", "")),
                "title": str(cand.get("candidate_title", "")),
                "score": 0.0,
                "label": int(cand.get("label", 0)),
            }
            for cand in pred.get("candidate_pool", [])
        ]
    ranking = []
    for cand in pred.get("candidate_pool", []):
        title = cand.get("candidate_title", "")
        score = 1.0 if normalize_title(generated) == normalize_title(title) and normalize_title(title) else token_similarity(generated, title)
        ranking.append(
            {
                "item_id": str(cand.get("candidate_item_id", "")),
                "title": str(title),
                "score": score,
                "label": int(cand.get("label", 0)),
            }
        )
    ranking.sort(key=lambda row: (-float(row["score"]), str(row["item_id"])))
    return ranking


def ground_prediction(pred: dict[str, Any], catalog: list[dict[str, Any]], threshold: float = 0.35) -> dict[str, Any]:
    generated = pred.get("recommended_title", "")
    normalized_generated = normalize_title(generated)
    placeholder_title = is_placeholder_title(generated)
    empty_title = str(generated or "").strip() == ""
    explanatory_after_json = explanatory_text_after_json(pred.get("raw_response", ""))
    candidate_pool = pred.get("candidate_pool", [])
    target_item_id = str(pred.get("target_item_id", ""))

    exact_candidates = [
        cand
        for cand in candidate_pool
        if (not placeholder_title) and normalized_generated and normalized_generated == normalize_title(cand.get("candidate_title", ""))
    ]
    candidate_ranking = _candidate_ranking(pred)
    best_candidate = candidate_ranking[0] if candidate_ranking else {"item_id": "", "title": "", "score": 0.0}
    target_rank = 0
    for idx, row in enumerate(candidate_ranking, start=1):
        if str(row.get("item_id", "")) == target_item_id:
            target_rank = idx
            break

    method = "none"
    matched_candidate_item_id = ""
    matched_candidate_title = ""
    candidate_match_score = 0.0
    if exact_candidates:
        method = "candidate_normalized_exact"
        matched_candidate_item_id = str(exact_candidates[0].get("candidate_item_id", ""))
        matched_candidate_title = str(exact_candidates[0].get("candidate_title", ""))
        candidate_match_score = 1.0
    elif (not placeholder_title) and float(best_candidate.get("score", 0.0)) >= threshold:
        method = "candidate_token_retrieval"
        matched_candidate_item_id = str(best_candidate.get("item_id", ""))
        matched_candidate_title = str(best_candidate.get("title", ""))
        candidate_match_score = float(best_candidate.get("score", 0.0))

    catalog_match = {"item_id": "", "title": "", "score": 0.0, "rank": 0}
    if catalog and not placeholder_title:
        scored = []
        for item in catalog:
            score = 1.0 if normalized_generated and normalized_generated == item["normalized_title"] else token_similarity(generated, item["title"])
            if score > 0:
                scored.append({"item_id": item["item_id"], "title": item["title"], "score": score})
        scored.sort(key=lambda row: (-float(row["score"]), str(row["item_id"])))
        if scored:
            catalog_match = {**scored[0], "rank": 1}

    candidate_title_exact_match = method == "candidate_normalized_exact"
    is_valid_candidate_title = bool(matched_candidate_item_id)
    is_valid_catalog_item = (not placeholder_title) and (is_valid_candidate_title or float(catalog_match.get("score", 0.0)) >= threshold)
    generation_valid = is_valid_catalog_item
    matched_item_id = matched_candidate_item_id or (str(catalog_match.get("item_id", "")) if is_valid_catalog_item else "")
    matched_title = matched_candidate_title or (str(catalog_match.get("title", "")) if is_valid_catalog_item else "")
    match_score = candidate_match_score if matched_candidate_item_id else float(catalog_match.get("score", 0.0))
    hit_target = matched_item_id == target_item_id and bool(matched_item_id)
    top1_from_candidate_retrieval = candidate_ranking[0]["item_id"] if candidate_ranking else ""
    return {
        "normalized_recommended_title": normalized_generated,
        "placeholder_title": placeholder_title,
        "empty_title": empty_title,
        "explanatory_text_after_json": explanatory_after_json,
        "matched_candidate_item_id": matched_candidate_item_id,
        "matched_candidate_title": matched_candidate_title,
        "candidate_match_method": method,
        "candidate_match_score": candidate_match_score,
        "candidate_title_exact_match": candidate_title_exact_match,
        "matched_item_id": matched_item_id,
        "matched_title": matched_title,
        "match_score": match_score,
        "match_rank": int(catalog_match.get("rank", 0)),
        "is_valid_candidate_title": is_valid_candidate_title,
        "is_valid_catalog_item": is_valid_catalog_item,
        "generation_valid": generation_valid,
        "invalid_title": not generation_valid,
        "hallucination": not is_valid_catalog_item,
        "hit_target": hit_target,
        "target_rank_if_candidate_pool_available": target_rank,
        "candidate_retrieval_top1_item_id": top1_from_candidate_retrieval,
        "candidate_retrieval_top1_hit": top1_from_candidate_retrieval == target_item_id,
    }


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    error = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        idxs = [
            j
            for j, score in enumerate(scores)
            if (score >= lo and (score < hi or (i == bins - 1 and score <= hi)))
        ]
        if not idxs:
            continue
        conf = mean(scores[j] for j in idxs)
        acc = mean(labels[j] for j in idxs)
        error += len(idxs) / total * abs(conf - acc)
    return error


def brier(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    return mean((score - label) ** 2 for score, label in zip(scores, labels))


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [(score, label) for score, label in zip(scores, labels) if label == 1]
    neg = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for ps, _ in pos:
        for ns, _ in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _dcg(rank: int) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks = [int(row.get("target_rank_if_candidate_pool_available", 0)) for row in rows]
    valid_ranks = [rank for rank in ranks if rank > 0]
    n = len(ranks)
    if n == 0:
        return {"HR@1": 0.0, "MRR": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0}
    return {
        "HR@1": sum(1 for rank in ranks if rank == 1) / n,
        "MRR": sum((1.0 / rank) if rank > 0 else 0.0 for rank in ranks) / n,
        "NDCG@3": sum(_dcg(rank) if 0 < rank <= 3 else 0.0 for rank in ranks) / n,
        "NDCG@5": sum(_dcg(rank) if 0 < rank <= 5 else 0.0 for rank in ranks) / n,
        "NDCG@10": sum(_dcg(rank) if 0 < rank <= 10 else 0.0 for rank in ranks) / n,
        "target_rank_mean": _safe_mean(valid_ranks),
    }


def _fit_bin_calibrator(scores: list[float], labels: list[int], bins: int = 10) -> list[dict[str, float]]:
    mapping = []
    global_rate = mean(labels) if labels else 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        idxs = [
            j
            for j, score in enumerate(scores)
            if score >= lo and (score < hi or (i == bins - 1 and score <= hi))
        ]
        empirical = mean(labels[j] for j in idxs) if idxs else global_rate
        mapping.append({"lo": lo, "hi": hi, "value": empirical})
    return mapping


def _apply_bin_calibrator(scores: list[float], mapping: list[dict[str, float]]) -> list[float]:
    out = []
    for score in scores:
        value = mapping[-1]["value"] if mapping else score
        for row in mapping:
            if score >= row["lo"] and (score < row["hi"] or row["hi"] >= 1.0):
                value = row["value"]
                break
        out.append(max(0.0, min(1.0, value)))
    return out


def _summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    n = len(rows)
    confidences = [float(row["confidence"]) for row in rows if row.get("confidence") is not None]
    correctness = [1 if row.get("hit_target") else 0 for row in rows]
    conf_for_rows = [float(row.get("confidence") or 0.0) for row in rows]
    ranking = _ranking_metrics(rows)
    diag = {
        "split": split,
        "num_users": n,
        "parse_success_rate": _safe_mean([1.0 if row.get("parse_success") else 0.0 for row in rows]),
        "schema_valid_rate": _safe_mean([1.0 if row.get("schema_valid") else 0.0 for row in rows]),
        "generation_valid_rate": _safe_mean([1.0 if row.get("generation_valid") else 0.0 for row in rows]),
        "placeholder_title_rate": _safe_mean([1.0 if row.get("placeholder_title") else 0.0 for row in rows]),
        "empty_title_rate": _safe_mean([1.0 if row.get("empty_title") else 0.0 for row in rows]),
        "invalid_title_rate": _safe_mean([1.0 if row.get("invalid_title") else 0.0 for row in rows]),
        "explanatory_text_after_json_rate": _safe_mean([1.0 if row.get("explanatory_text_after_json") else 0.0 for row in rows]),
        "candidate_title_exact_match_rate": _safe_mean([1.0 if row.get("candidate_title_exact_match") else 0.0 for row in rows]),
        "valid_candidate_title_rate": _safe_mean([1.0 if row.get("is_valid_candidate_title") else 0.0 for row in rows]),
        "catalog_match_rate": _safe_mean([1.0 if row.get("is_valid_catalog_item") else 0.0 for row in rows]),
        "matched_title_hit_rate": _safe_mean([1.0 if row.get("hit_target") else 0.0 for row in rows]),
        "matched_title_hit_rate_given_generation_valid": _safe_mean(
            [1.0 if row.get("hit_target") else 0.0 for row in rows if row.get("generation_valid")]
        ),
        "hallucination_rate": _safe_mean([1.0 if row.get("hallucination") else 0.0 for row in rows]),
        "HR@1": ranking["HR@1"],
        "MRR": ranking["MRR"],
        "NDCG@3": ranking["NDCG@3"],
        "NDCG@5": ranking["NDCG@5"],
        "NDCG@10": ranking["NDCG@10"],
        "target_rank_mean": ranking.get("target_rank_mean", 0.0),
        "confidence_mean": _safe_mean(confidences),
        "confidence_std": _safe_std(confidences),
        "confidence_unique_count": len(set(round(x, 6) for x in confidences)),
        "confidence_ge_0.9_rate": _safe_mean([1.0 if x >= 0.9 else 0.0 for x in confidences]),
        "ECE_for_generation_correctness": ece(conf_for_rows, correctness),
        "Brier_for_generation_correctness": brier(conf_for_rows, correctness),
        "AUROC_for_generation_correctness": auroc(conf_for_rows, correctness),
        "high_conf_wrong_rate": _safe_mean(
            [1.0 if (row.get("confidence") is not None and float(row["confidence"]) >= 0.9 and not row.get("hit_target")) else 0.0 for row in rows]
        ),
    }
    return diag


def _calibration_rows(enriched_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    valid = enriched_by_split.get("valid", [])
    test = enriched_by_split.get("test", [])
    rows = []
    for split, split_rows in enriched_by_split.items():
        labels = [1 if row.get("hit_target") else 0 for row in split_rows]
        scores = [float(row.get("confidence") or 0.0) for row in split_rows]
        rows.append(
            {
                "split": split,
                "score_type": "raw_verbalized_confidence",
                "fit_split": "none",
                "ECE": ece(scores, labels),
                "Brier": brier(scores, labels),
                "AUROC": auroc(scores, labels),
                "note": "diagnostic_only" if not valid or split == "valid" else "",
            }
        )
    if valid and test:
        valid_scores = [float(row.get("confidence") or 0.0) for row in valid]
        valid_labels = [1 if row.get("hit_target") else 0 for row in valid]
        mapping = _fit_bin_calibrator(valid_scores, valid_labels)
        for split, split_rows in enriched_by_split.items():
            labels = [1 if row.get("hit_target") else 0 for row in split_rows]
            scores = [float(row.get("confidence") or 0.0) for row in split_rows]
            calibrated = _apply_bin_calibrator(scores, mapping)
            rows.append(
                {
                    "split": split,
                    "score_type": "calibrated_verbalized_confidence",
                    "fit_split": "valid",
                    "ECE": ece(calibrated, labels),
                    "Brier": brier(calibrated, labels),
                    "AUROC": auroc(calibrated, labels),
                    "note": "valid_fit_test_evaluate",
                }
            )
    return rows


def _analysis_paths(cfg: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    if cfg.get("analysis_diagnostics_csv"):
        return (
            Path(str(cfg["analysis_diagnostics_csv"])),
            Path(str(cfg["analysis_calibration_csv"])),
            Path(str(cfg["analysis_report_md"])),
            Path(str(cfg.get("analysis_comparison_csv", COMPARISON_CSV))),
        )
    return DIAG_CSV, CAL_CSV, REPORT_MD, COMPARISON_CSV


def _first_value(path: Path, candidates: list[str]) -> str:
    rows = _read_csv(path)
    if not rows:
        return "NA"
    row = rows[-1]
    for key in candidates:
        if key in row and row[key] not in {"", None}:
            return str(row[key])
    return "NA"


def _comparison_rows(diag_rows: list[dict[str, Any]], cal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_diag = next((row for row in diag_rows if row.get("split") == "test"), diag_rows[-1] if diag_rows else {})
    raw_cal = next(
        (row for row in cal_rows if row.get("split") == "test" and row.get("score_type") == "raw_verbalized_confidence"),
        {},
    )
    old_rows = [
        {
            "observation_type": "pointwise verbalized confidence",
            "task_form": "binary yes/no",
            "output_type": "recommend + confidence",
            "confidence_source": "verbalized_scalar",
            "parse_success": _first_value(Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv"), ["parse_success_rate"]),
            "valid_output_rate": _first_value(Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv"), ["schema_valid_rate"]),
            "recommendation_metric": _first_value(Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv"), ["accuracy"]),
            "confidence_ECE": _first_value(Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv"), ["ECE"]),
            "confidence_AUROC": _first_value(Path("data_done/framework_observation_day1_beauty_confidence_diagnostics.csv"), ["AUROC"]),
            "hallucination_rate": "NA",
            "claim_level": "baseline_observation",
            "notes": "high-confidence saturation in Day1; not a generative task",
        },
        {
            "observation_type": "pointwise logit P(true)",
            "task_form": "binary yes/no",
            "output_type": "token probability",
            "confidence_source": "token_probability",
            "parse_success": "NA",
            "valid_output_rate": "NA",
            "recommendation_metric": _first_value(Path("data_done/framework_observation_day1e_logit_score_ranking_eval.csv"), ["MRR"]),
            "confidence_ECE": _first_value(Path("data_done/framework_observation_day1d_logit_confidence_calibration.csv"), ["raw_ECE", "ECE"]),
            "confidence_AUROC": _first_value(Path("data_done/framework_observation_day1d_logit_confidence_diagnostics.csv"), ["AUROC"]),
            "hallucination_rate": "NA",
            "claim_level": "weak_usable_signal",
            "notes": "usable but weak; still tied to yes/no formulation",
        },
        {
            "observation_type": "listwise ranking shuffled",
            "task_form": "relative candidate ranking",
            "output_type": "ranked item ids",
            "confidence_source": "behavioral_rank_score",
            "parse_success": _first_value(Path("data_done/framework_observation_day1i_shuffled_behavioral_diagnostics.csv"), ["parse_success_rate"]),
            "valid_output_rate": _first_value(Path("data_done/framework_observation_day1i_shuffled_behavioral_diagnostics.csv"), ["complete_ranking_rate", "schema_valid_rate"]),
            "recommendation_metric": _first_value(Path("data_done/framework_observation_day1i_order_bias_control_comparison.csv"), ["MRR"]),
            "confidence_ECE": _first_value(Path("data_done/framework_observation_day1i_shuffled_behavioral_uncertainty_calibration.csv"), ["ECE", "raw_ECE"]),
            "confidence_AUROC": _first_value(Path("data_done/framework_observation_day1i_order_bias_control_comparison.csv"), ["top1_confidence_AUROC"]),
            "hallucination_rate": "NA",
            "claim_level": "order_bias_control_observation",
            "notes": "ranking remains useful; behavioral uncertainty was order-bias confounded",
        },
    ]
    day2_rows = [
        {
            "observation_type": "generative title candidate-grounded confidence",
            "task_form": "candidate-grounded generative recommendation",
            "output_type": "recommended_title + confidence",
            "confidence_source": "verbalized_scalar",
            "parse_success": test_diag.get("parse_success_rate", "NA"),
            "valid_output_rate": test_diag.get("generation_valid_rate", "NA"),
            "recommendation_metric": test_diag.get("matched_title_hit_rate", "NA"),
            "confidence_ECE": raw_cal.get("ECE", "NA"),
            "confidence_AUROC": raw_cal.get("AUROC", "NA"),
            "hallucination_rate": test_diag.get("hallucination_rate", "NA"),
            "claim_level": "day2_smoke_observation",
            "notes": "tests whether generated title confidence predicts catalog-grounded correctness",
        },
        {
            "observation_type": "generative title candidate-grounded validity",
            "task_form": "candidate-grounded generative recommendation",
            "output_type": "catalog-grounded title",
            "confidence_source": "catalog_match_score",
            "parse_success": test_diag.get("parse_success_rate", "NA"),
            "valid_output_rate": test_diag.get("generation_valid_rate", "NA"),
            "recommendation_metric": test_diag.get("matched_title_hit_rate", "NA"),
            "confidence_ECE": "diagnostic_after_run",
            "confidence_AUROC": "diagnostic_after_run",
            "hallucination_rate": test_diag.get("hallucination_rate", "NA"),
            "claim_level": "day2_smoke_observation",
            "notes": "validity/matchability is treated as an uncertainty signal candidate",
        },
    ]
    return old_rows + day2_rows


def _write_report(diag_rows: list[dict[str, Any]], cal_rows: list[dict[str, Any]], report_path: Path, day2b: bool) -> None:
    title = (
        "# Framework-Observation-Day2b Generative Recommendation Prompt/Parser Repair Report"
        if day2b
        else "# Framework-Observation-Day2 Generative Recommendation Smoke Report"
    )
    lines = [
        title,
        "",
        "Status: observation only. This is not training, CEP, evidence decomposition, or a continuation of yes/no confidence prompting.",
        "",
        "## Setup",
        "",
        "- Task: Beauty candidate-grounded title generation.",
        "- Model: base Qwen3-8B first; LoRA remains optional.",
        "- Output schema: `recommended_title` plus raw verbalized `confidence`.",
        "- Evaluation: generated title is grounded back to the 6-item candidate pool and then to the catalog when needed.",
        "- Placeholder titles such as `...`, empty strings, `N/A`, `unknown`, and `none` are generation-invalid.",
        "",
        "## Day2b Interpretation",
        "",
        "- Day2 exposed a placeholder/schema-following failure: parse/schema success can be superficial.",
        "- Day2b tests whether removing placeholder examples and forcing exact candidate-title copying fixes output control.",
        "- Main metrics are generation validity, exact candidate-title matching, matched-title hit rate, hallucination, and placeholder rate.",
        "",
        "## Diagnostics",
        "",
    ]
    if diag_rows:
        headers = list(diag_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in diag_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend(["", "## Calibration", ""])
    if cal_rows:
        headers = list(cal_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in cal_rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- If candidate-grounded title validity and HR@1 are reasonable, Day3 can expand to valid/test 500 or Beauty full.",
            "- If raw verbalized confidence collapses again, Day3 should add generation logprob, title retrieval margin, and title self-consistency agreement.",
            "- If hallucination is high, keep candidate-grounded generation and defer open-title full runs.",
            "- If placeholder outputs persist, switch to the Day2c label-first generation fallback before expanding sample size.",
            "- If candidate-grounded generation is strong, later evidence observation or CEP can be considered, but Day2 itself is still observation.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _day2b_comparison(day2_diag_path: Path, day2b_diag_rows: list[dict[str, Any]], day2b_cal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    day2_rows = _read_csv(day2_diag_path)
    if day2_rows:
        day2_test = next((row for row in day2_rows if row.get("split") == "test"), day2_rows[-1])
        rows.append(
            {
                "method": "base_qwen_candidate_grounded",
                "prompt_version": "day2_placeholder_schema",
                "num_users": day2_test.get("num_users", "NA"),
                "parse_success_rate": day2_test.get("parse_success_rate", "NA"),
                "schema_valid_rate": day2_test.get("schema_valid_rate", "NA"),
                "generation_valid_rate": day2_test.get("generation_valid_rate", day2_test.get("catalog_match_rate", "NA")),
                "placeholder_title_rate": day2_test.get("placeholder_title_rate", "0.87_observed"),
                "candidate_title_exact_match_rate": day2_test.get("candidate_title_exact_match_rate", day2_test.get("valid_candidate_title_rate", "NA")),
                "catalog_match_rate": day2_test.get("catalog_match_rate", "NA"),
                "matched_title_hit_rate": day2_test.get("matched_title_hit_rate", day2_test.get("HR@1", "NA")),
                "hallucination_rate": day2_test.get("hallucination_rate", "NA"),
                "confidence_mean": day2_test.get("confidence_mean", "NA"),
                "confidence_std": day2_test.get("confidence_std", "NA"),
                "confidence_unique_count": day2_test.get("confidence_unique_count", "NA"),
                "confidence_ECE_for_generation_correctness": day2_test.get("ECE_for_generation_correctness", "NA"),
                "confidence_AUROC_for_generation_correctness": day2_test.get("AUROC_for_generation_correctness", "NA"),
                "notes": "Day2 parse/schema looked good but most outputs were placeholder titles.",
            }
        )
    if day2b_diag_rows:
        test = next((row for row in day2b_diag_rows if row.get("split") == "test"), day2b_diag_rows[-1])
        raw_cal = next(
            (row for row in day2b_cal_rows if row.get("split") == "test" and row.get("score_type") == "raw_verbalized_confidence"),
            {},
        )
        rows.append(
            {
                "method": "base_qwen_candidate_grounded",
                "prompt_version": "day2b_no_placeholder_exact_title",
                "num_users": test.get("num_users", "NA"),
                "parse_success_rate": test.get("parse_success_rate", "NA"),
                "schema_valid_rate": test.get("schema_valid_rate", "NA"),
                "generation_valid_rate": test.get("generation_valid_rate", "NA"),
                "placeholder_title_rate": test.get("placeholder_title_rate", "NA"),
                "candidate_title_exact_match_rate": test.get("candidate_title_exact_match_rate", "NA"),
                "catalog_match_rate": test.get("catalog_match_rate", "NA"),
                "matched_title_hit_rate": test.get("matched_title_hit_rate", "NA"),
                "hallucination_rate": test.get("hallucination_rate", "NA"),
                "confidence_mean": test.get("confidence_mean", "NA"),
                "confidence_std": test.get("confidence_std", "NA"),
                "confidence_unique_count": test.get("confidence_unique_count", "NA"),
                "confidence_ECE_for_generation_correctness": raw_cal.get("ECE", test.get("ECE_for_generation_correctness", "NA")),
                "confidence_AUROC_for_generation_correctness": raw_cal.get("AUROC", test.get("AUROC_for_generation_correctness", "NA")),
                "notes": "Day2b fixes output control if placeholder rate drops and generation validity rises.",
            }
        )
    return rows


def analyze(cfg: dict[str, Any], pred_dir: Path) -> dict[str, str]:
    diag_path, cal_path, report_path, comparison_path = _analysis_paths(cfg)
    catalog = _load_catalog(Path(str(cfg.get("catalog_file", ""))))
    enriched_by_split: dict[str, list[dict[str, Any]]] = {}
    diag_rows = []
    for split in ["valid", "test"]:
        path = pred_dir / f"{split}_raw.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        enriched = []
        for row in rows:
            enriched.append({**row, **ground_prediction(row, catalog)})
        enriched_by_split[split] = enriched
        diag_rows.append(_summarize_split(enriched, split))
    cal_rows = _calibration_rows(enriched_by_split)
    _write_csv(diag_path, diag_rows)
    _write_csv(cal_path, cal_rows)
    _write_csv(comparison_path, _comparison_rows(diag_rows, cal_rows))
    is_day2b = "day2b" in str(diag_path).lower() or "day2b" in str(report_path).lower()
    _write_report(diag_rows, cal_rows, report_path, is_day2b)
    if is_day2b:
        day2_diag_path = Path(str(cfg.get("day2_diagnostics_csv", DIAG_CSV)))
        _write_csv(DAY2B_COMPARE_CSV, _day2b_comparison(day2_diag_path, diag_rows, cal_rows))
    return {
        "diagnostics": str(diag_path),
        "calibration": str(cal_path),
        "report": str(report_path),
        "comparison": str(comparison_path),
        "day2b_comparison": str(DAY2B_COMPARE_CSV) if is_day2b else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pred_dir", default=None)
    args = parser.parse_args()
    cfg = _read_config(args.config)
    pred_dir = Path(args.pred_dir or str(cfg["output_dir"]))
    print(json.dumps(analyze(cfg, pred_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
