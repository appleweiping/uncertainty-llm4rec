from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import _compact_json, _read_jsonl, _truncate_text
from main_framework_observation_day1f_self_consistency import (
    _calibrate,
    _group_by_user,
    _label,
    _load_vllm,
    _lora_request,
    _read_pred,
    _score,
    _selected_rows,
    _subset_entries,
    _write_csv,
    _write_jsonl,
    auroc,
    brier,
    ece,
    load_subset,
    random_ranking_metrics,
    ranking_metrics,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_day1g_context.yaml"
LIST_DIAG_CSV = Path("data_done/framework_observation_day1g_listwise_context_diagnostics.csv")
LIST_CAL_CSV = Path("data_done/framework_observation_day1g_listwise_context_calibration.csv")
LIST_REPORT_MD = Path("data_done/framework_observation_day1g_listwise_context_report.md")
PAIR_DIAG_CSV = Path("data_done/framework_observation_day1g_pairwise_context_diagnostics.csv")
PAIR_CAL_CSV = Path("data_done/framework_observation_day1g_pairwise_context_calibration.csv")
PAIR_RANK_CSV = Path("data_done/framework_observation_day1g_pairwise_context_ranking_eval.csv")
COMPARISON_CSV = Path("data_done/framework_observation_day1g_context_comparison.csv")
GO_NO_GO_MD = Path("data_done/framework_observation_day1g_go_no_go_decision.md")


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in _read_jsonl(path)}


def _history_payload(sample: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    history = sample.get("history", [])
    max_history_items = int(cfg.get("max_history_items", 8))
    if max_history_items > 0:
        history = history[-max_history_items:]
    return [
        {
            "item_id": str(item.get("item_id", "")),
            "title": _truncate_text(item.get("title", ""), int(cfg.get("max_title_chars", 160))),
            "text": _truncate_text(item.get("text", ""), int(cfg.get("max_history_text_chars", 220))),
            "text_missing": bool(item.get("text_missing", False)),
        }
        for item in history
    ]


def _candidate_payload(sample: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_item_id": str(sample.get("candidate_item_id", "")),
        "title": _truncate_text(sample.get("candidate_title", ""), int(cfg.get("max_title_chars", 160))),
        "text": _truncate_text(sample.get("candidate_text", ""), int(cfg.get("max_candidate_text_chars", 280))),
        "candidate_text_missing": bool(sample.get("candidate_text_missing", False)),
    }


def _listwise_prompt(user_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    template = Path(str(cfg["listwise_prompt_template"])).read_text(encoding="utf-8").strip()
    payload = {
        "user_history": _history_payload(user_rows[0], cfg),
        "candidate_pool": [_candidate_payload(row, cfg) for row in user_rows],
    }
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


def _pairwise_prompt(pair: dict[str, Any], cfg: dict[str, Any]) -> str:
    template = Path(str(cfg["pairwise_prompt_template"])).read_text(encoding="utf-8").strip()
    payload = {
        "user_history": _history_payload(pair["a_row"], cfg),
        "candidate_A": _candidate_payload(pair["a_row"], cfg),
        "candidate_B": _candidate_payload(pair["b_row"], cfg),
    }
    return f"{template}\n\nInput JSON:\n{_compact_json(payload)}\n\nOutput JSON:\n"


def _json_obj(raw_text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", raw_text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _clip01(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        return None
    return x


def _split_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    subset = load_subset(cfg)
    user_ids = _subset_entries(subset)[split]["sampled_user_ids"]
    return _selected_rows(Path(str(cfg[f"{split}_file"])), user_ids)


def _parse_listwise(raw_text: str, candidate_ids: list[str]) -> dict[str, Any]:
    obj = _json_obj(raw_text)
    if obj is None:
        return {
            "parse_success": False,
            "schema_valid": False,
            "ranked_item_ids": [],
            "top1_confidence": None,
            "rank_margin": None,
            "missing_ranked_item_ids_count": len(candidate_ids),
            "duplicate_item_count": 0,
            "invalid_item_count": 0,
            "complete_ranking": False,
        }
    ranked = obj.get("ranked_item_ids")
    ranked_ids = [str(x) for x in ranked] if isinstance(ranked, list) else []
    top1_conf = _clip01(obj.get("top1_confidence"))
    rank_margin = _clip01(obj.get("rank_margin"))
    candidate_set = set(candidate_ids)
    ranked_set = set(ranked_ids)
    missing = [item_id for item_id in candidate_ids if item_id not in ranked_set]
    invalid = [item_id for item_id in ranked_ids if item_id not in candidate_set]
    duplicate_count = len(ranked_ids) - len(ranked_set)
    complete = len(ranked_ids) == len(candidate_ids) and not missing and not invalid and duplicate_count == 0
    schema_valid = complete and top1_conf is not None and rank_margin is not None
    return {
        "parse_success": True,
        "schema_valid": schema_valid,
        "ranked_item_ids": ranked_ids,
        "top1_confidence": top1_conf,
        "rank_margin": rank_margin,
        "missing_ranked_item_ids_count": len(missing),
        "duplicate_item_count": duplicate_count,
        "invalid_item_count": len(invalid),
        "complete_ranking": complete,
    }


def _parse_pairwise(raw_text: str) -> dict[str, Any]:
    obj = _json_obj(raw_text)
    if obj is None:
        return {"parse_success": False, "schema_valid": False, "preferred": "", "confidence": None, "preference_margin": None}
    preferred = obj.get("preferred")
    preferred = preferred.strip().upper() if isinstance(preferred, str) else ""
    confidence = _clip01(obj.get("confidence"))
    margin = _clip01(obj.get("preference_margin"))
    schema_valid = preferred in {"A", "B"} and confidence is not None and margin is not None
    return {
        "parse_success": True,
        "schema_valid": schema_valid,
        "preferred": preferred,
        "confidence": confidence,
        "preference_margin": margin,
    }


def run_listwise(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore

    rows = _split_rows(cfg, split)
    grouped = _group_by_user(rows)
    output_path = Path(str(cfg["listwise_output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    batch_size = int(cfg.get("vllm_batch_size", 16))
    pending: list[tuple[str, list[dict[str, Any]], str]] = []
    for user_id, user_rows in grouped.items():
        sid = f"{split}_{user_id}"
        if sid in finished:
            continue
        pending.append((sid, user_rows, _listwise_prompt(user_rows, cfg)))
    sampling_params = SamplingParams(
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_new_tokens_listwise", 192)),
        seed=int(cfg.get("seed", 42)),
    )
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = llm.generate([item[2] for item in batch], sampling_params, lora_request=lora_request)
        out_rows = []
        for (sid, user_rows, _), output in zip(batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            candidate_ids = [str(row.get("candidate_item_id", "")) for row in user_rows]
            labels_by_item = {str(row.get("candidate_item_id", "")): int(row.get("label", 0)) for row in user_rows}
            parsed = _parse_listwise(raw_text, candidate_ids)
            ranked_ids = parsed["ranked_item_ids"]
            top1_id = ranked_ids[0] if parsed["schema_valid"] and ranked_ids else ""
            out_rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "user_id": user_rows[0].get("user_id", ""),
                    "candidate_item_ids": candidate_ids,
                    "labels_by_item": labels_by_item,
                    "positive_item_id": next((item_id for item_id, label in labels_by_item.items() if label == 1), ""),
                    "raw_text": raw_text,
                    "ranked_item_ids": ranked_ids,
                    "top1_item_id": top1_id,
                    "top1_correct": int(labels_by_item.get(top1_id, 0) == 1) if top1_id else 0,
                    "top1_confidence": parsed["top1_confidence"],
                    "rank_margin": parsed["rank_margin"],
                    "parse_success": parsed["parse_success"],
                    "schema_valid": parsed["schema_valid"],
                    "missing_ranked_item_ids_count": parsed["missing_ranked_item_ids_count"],
                    "duplicate_item_count": parsed["duplicate_item_count"],
                    "invalid_item_count": parsed["invalid_item_count"],
                    "complete_ranking": parsed["complete_ranking"],
                    "inference_backend": "vllm",
                    "context_type": "listwise",
                }
            )
        _write_jsonl(output_path, out_rows)
    return output_path


def _pair_rows_for_user(user_rows: list[dict[str, Any]], cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    positives = [row for row in user_rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in user_rows if int(row.get("label", 0)) == 0]
    if not positives:
        return []
    pos = positives[0]
    max_neg = int(cfg.get("pairwise_negatives_per_user", 5))
    pairs = []
    for idx, neg in enumerate(negatives[:max_neg]):
        if idx % 2 == 0:
            a_row, b_row = pos, neg
        else:
            a_row, b_row = neg, pos
        pairs.append(
            {
                "sample_id": f"{split}_{pos.get('user_id', '')}_{a_row.get('candidate_item_id', '')}_{b_row.get('candidate_item_id', '')}",
                "user_id": str(pos.get("user_id", "")),
                "a_row": a_row,
                "b_row": b_row,
            }
        )
    return pairs


def run_pairwise(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore

    rows = _split_rows(cfg, split)
    grouped = _group_by_user(rows)
    output_path = Path(str(cfg["pairwise_output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    pairs = [pair for user_rows in grouped.values() for pair in _pair_rows_for_user(user_rows, cfg, split)]
    pending = [(pair, _pairwise_prompt(pair, cfg)) for pair in pairs if pair["sample_id"] not in finished]
    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    batch_size = int(cfg.get("vllm_batch_size", 16))
    sampling_params = SamplingParams(
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_tokens=int(cfg.get("max_new_tokens_pairwise", 96)),
        seed=int(cfg.get("seed", 42)),
    )
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = llm.generate([item[1] for item in batch], sampling_params, lora_request=lora_request)
        out_rows = []
        for (pair, _), output in zip(batch, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed = _parse_pairwise(raw_text)
            preferred_id = ""
            preferred_label = 0
            if parsed["schema_valid"]:
                preferred_row = pair["a_row"] if parsed["preferred"] == "A" else pair["b_row"]
                preferred_id = str(preferred_row.get("candidate_item_id", ""))
                preferred_label = int(preferred_row.get("label", 0))
            out_rows.append(
                {
                    "sample_id": pair["sample_id"],
                    "split": split,
                    "user_id": pair["user_id"],
                    "candidate_A_item_id": pair["a_row"].get("candidate_item_id", ""),
                    "candidate_B_item_id": pair["b_row"].get("candidate_item_id", ""),
                    "candidate_A_label": int(pair["a_row"].get("label", 0)),
                    "candidate_B_label": int(pair["b_row"].get("label", 0)),
                    "raw_text": raw_text,
                    "preferred": parsed["preferred"],
                    "preferred_item_id": preferred_id,
                    "preferred_label": preferred_label,
                    "pairwise_correct": int(preferred_label == max(int(pair["a_row"].get("label", 0)), int(pair["b_row"].get("label", 0)))) if parsed["schema_valid"] else 0,
                    "confidence": parsed["confidence"],
                    "preference_margin": parsed["preference_margin"],
                    "parse_success": parsed["parse_success"],
                    "schema_valid": parsed["schema_valid"],
                    "inference_backend": "vllm",
                    "context_type": "pairwise",
                }
            )
        _write_jsonl(output_path, out_rows)
    return output_path


def _listwise_candidate_rows(list_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in list_rows:
        labels = row.get("labels_by_item", {})
        candidate_ids = [str(x) for x in row.get("candidate_item_ids", [])]
        if bool(row.get("schema_valid", False)):
            ranked = [str(x) for x in row.get("ranked_item_ids", [])]
            rank_score = {item_id: len(candidate_ids) - rank for rank, item_id in enumerate(ranked)}
        else:
            # Invalid rankings should not inherit JSONL candidate order.
            rank_score = {}
        for item_id in candidate_ids:
            out.append(
                {
                    "user_id": row.get("user_id", ""),
                    "candidate_item_id": item_id,
                    "label": int(labels.get(item_id, 0)),
                    "listwise_rank_score": float(rank_score.get(item_id, 0.0)),
                }
            )
    return out


def _pairwise_candidate_rows(pair_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = {
        (str(row.get("user_id", "")), str(row.get("candidate_item_id", ""))): {
            "user_id": row.get("user_id", ""),
            "candidate_item_id": row.get("candidate_item_id", ""),
            "label": int(row.get("label", 0)),
            "wins": 0.0,
            "comparisons": 0.0,
            "margin_sum": 0.0,
            "weighted_sum": 0.0,
        }
        for row in split_rows
    }
    for row in pair_rows:
        if not bool(row.get("schema_valid", False)):
            continue
        a = (str(row.get("user_id", "")), str(row.get("candidate_A_item_id", "")))
        b = (str(row.get("user_id", "")), str(row.get("candidate_B_item_id", "")))
        pref = str(row.get("preferred", ""))
        margin = _score(row, "preference_margin")
        conf = _score(row, "confidence")
        for key in [a, b]:
            if key in base:
                base[key]["comparisons"] += 1.0
        winner, loser = (a, b) if pref == "A" else (b, a)
        if winner in base:
            base[winner]["wins"] += 1.0
            base[winner]["margin_sum"] += margin
            base[winner]["weighted_sum"] += margin * conf
        if loser in base:
            base[loser]["margin_sum"] -= margin
            base[loser]["weighted_sum"] -= margin * conf
    out = []
    for row in base.values():
        comps = row["comparisons"]
        row["pairwise_win_rate"] = row["wins"] / comps if comps else 0.0
        row["pairwise_margin_score"] = row["margin_sum"] / comps if comps else 0.0
        row["pairwise_confidence_weighted_score"] = row["weighted_sum"] / comps if comps else 0.0
        out.append(row)
    return out


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if bool(row.get(key, False)) else 0.0 for row in rows]) if rows else 0.0


def _uncertainty_metrics(rows: list[dict[str, Any]], score_key: str, label_key: str) -> dict[str, Any]:
    scores = [_score(row, score_key) for row in rows if row.get(score_key) not in [None, ""]]
    labels = [int(row.get(label_key, 0)) for row in rows if row.get(score_key) not in [None, ""]]
    return {
        f"{score_key}_ECE": ece(scores, labels),
        f"{score_key}_Brier": brier(scores, labels),
        f"{score_key}_AUROC": auroc(scores, labels),
    }


def listwise_diagnostics(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        candidate_rows = _listwise_candidate_rows(rows)
        rank = ranking_metrics(candidate_rows, "listwise_rank_score")
        top_conf = [_score(row, "top1_confidence") for row in rows if row.get("top1_confidence") not in [None, ""]]
        margins = [_score(row, "rank_margin") for row in rows if row.get("rank_margin") not in [None, ""]]
        top_correct = [int(row.get("top1_correct", 0)) for row in rows if row.get("top1_confidence") not in [None, ""]]
        margin_correct = [int(row.get("top1_correct", 0)) for row in rows if row.get("rank_margin") not in [None, ""]]
        out.append(
            {
                "split": split,
                "num_users": len(rows),
                "parse_success_rate": _rate(rows, "parse_success"),
                "schema_valid_rate": _rate(rows, "schema_valid"),
                "missing_ranked_item_ids_rate": mean([1.0 if int(row.get("missing_ranked_item_ids_count", 0)) > 0 else 0.0 for row in rows]) if rows else 0.0,
                "duplicate_item_rate": mean([1.0 if int(row.get("duplicate_item_count", 0)) > 0 else 0.0 for row in rows]) if rows else 0.0,
                "invalid_item_rate": mean([1.0 if int(row.get("invalid_item_count", 0)) > 0 else 0.0 for row in rows]) if rows else 0.0,
                "complete_ranking_rate": _rate(rows, "complete_ranking"),
                **rank,
                "top1_confidence_mean": mean(top_conf) if top_conf else 0.0,
                "top1_confidence_std": pstdev(top_conf) if len(top_conf) > 1 else 0.0,
                "top1_confidence_unique_count": len(set(top_conf)),
                "top1_confidence_ECE_for_top1_correctness": ece(top_conf, top_correct),
                "top1_confidence_Brier_for_top1_correctness": brier(top_conf, top_correct),
                "top1_confidence_AUROC_for_top1_correctness": auroc(top_conf, top_correct),
                "rank_margin_mean": mean(margins) if margins else 0.0,
                "rank_margin_std": pstdev(margins) if len(margins) > 1 else 0.0,
                "rank_margin_AUROC_for_top1_correctness": auroc(margins, margin_correct),
            }
        )
    return out


def listwise_calibration(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for score_key in ["top1_confidence", "rank_margin"]:
        valid_scores = [_score(row, score_key) for row in valid_rows if row.get(score_key) not in [None, ""]]
        valid_labels = [int(row.get("top1_correct", 0)) for row in valid_rows if row.get(score_key) not in [None, ""]]
        test_scores = [_score(row, score_key) for row in test_rows if row.get(score_key) not in [None, ""]]
        test_labels = [int(row.get("top1_correct", 0)) for row in test_rows if row.get(score_key) not in [None, ""]]
        cal = _calibrate(valid_scores, valid_labels, test_scores)
        raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
        calibrated = {"ECE": ece(cal, test_labels), "Brier": brier(cal, test_labels), "AUROC": auroc(cal, test_labels)}
        out.append({"score": score_key, "score_type": "raw", **raw, "calibration_method": "none"})
        out.append({"score": score_key, "score_type": "logistic_calibrated", **calibrated, "calibration_method": "valid_logistic", "delta_ECE_vs_raw": calibrated["ECE"] - raw["ECE"], "delta_Brier_vs_raw": calibrated["Brier"] - raw["Brier"]})
    return out


def pairwise_diagnostics(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        conf = [_score(row, "confidence") for row in rows if row.get("confidence") not in [None, ""]]
        margins = [_score(row, "preference_margin") for row in rows if row.get("preference_margin") not in [None, ""]]
        correctness_conf = [int(row.get("pairwise_correct", 0)) for row in rows if row.get("confidence") not in [None, ""]]
        correctness_margin = [int(row.get("pairwise_correct", 0)) for row in rows if row.get("preference_margin") not in [None, ""]]
        out.append(
            {
                "split": split,
                "num_pairs": len(rows),
                "parse_success_rate": _rate(rows, "parse_success"),
                "schema_valid_rate": _rate(rows, "schema_valid"),
                "pairwise_accuracy": mean([int(row.get("pairwise_correct", 0)) for row in rows]) if rows else 0.0,
                "pairwise_confidence_mean": mean(conf) if conf else 0.0,
                "pairwise_confidence_std": pstdev(conf) if len(conf) > 1 else 0.0,
                "pairwise_confidence_ECE_for_correctness": ece(conf, correctness_conf),
                "pairwise_confidence_Brier_for_correctness": brier(conf, correctness_conf),
                "pairwise_confidence_AUROC_for_correctness": auroc(conf, correctness_conf),
                "preference_margin_mean": mean(margins) if margins else 0.0,
                "preference_margin_std": pstdev(margins) if len(margins) > 1 else 0.0,
                "preference_margin_ECE_for_correctness": ece(margins, correctness_margin),
                "preference_margin_Brier_for_correctness": brier(margins, correctness_margin),
                "preference_margin_AUROC_for_correctness": auroc(margins, correctness_margin),
            }
        )
    return out


def pairwise_calibration(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for score_key in ["confidence", "preference_margin"]:
        valid_scores = [_score(row, score_key) for row in valid_rows if row.get(score_key) not in [None, ""]]
        valid_labels = [int(row.get("pairwise_correct", 0)) for row in valid_rows if row.get(score_key) not in [None, ""]]
        test_scores = [_score(row, score_key) for row in test_rows if row.get(score_key) not in [None, ""]]
        test_labels = [int(row.get("pairwise_correct", 0)) for row in test_rows if row.get(score_key) not in [None, ""]]
        cal = _calibrate(valid_scores, valid_labels, test_scores)
        raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
        calibrated = {"ECE": ece(cal, test_labels), "Brier": brier(cal, test_labels), "AUROC": auroc(cal, test_labels)}
        out.append({"score": score_key, "score_type": "raw", **raw, "calibration_method": "none"})
        out.append({"score": score_key, "score_type": "logistic_calibrated", **calibrated, "calibration_method": "valid_logistic", "delta_ECE_vs_raw": calibrated["ECE"] - raw["ECE"], "delta_Brier_vs_raw": calibrated["Brier"] - raw["Brier"]})
    return out


def pairwise_ranking_rows(pair_rows_by_split: dict[str, list[dict[str, Any]]], split_rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    out = []
    candidates_by_split = {}
    for split, pair_rows in pair_rows_by_split.items():
        candidate_rows = _pairwise_candidate_rows(pair_rows, split_rows_by_split[split])
        candidates_by_split[split] = candidate_rows
        random_metrics = random_ranking_metrics(candidate_rows)
        for method in ["pairwise_win_rate", "pairwise_margin_score", "pairwise_confidence_weighted_score"]:
            out.append({"split": split, "ranking_method": method, **ranking_metrics(candidate_rows, method), "random_baseline": json.dumps(random_metrics, sort_keys=True), "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "random", **random_metrics, "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "oracle", "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "oracle_upper_bound": 1.0})
    return out, candidates_by_split


def _metric_bundle(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    scores = [_score(row, score_key) for row in rows]
    labels = [_label(row) for row in rows]
    return {
        "num_users": len(_group_by_user(rows)),
        "num_rows_or_pairs": len(rows),
        **ranking_metrics(rows, score_key),
        "AUROC": auroc(scores, labels),
        "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
    }


def context_comparison(
    cfg: dict[str, Any],
    split_rows_by_split: dict[str, list[dict[str, Any]]],
    list_rows_by_split: dict[str, list[dict[str, Any]]],
    pair_candidates_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    logit_dir = Path(str(cfg["pointwise_logit_output_dir"]))
    self_dir = Path(str(cfg["self_consistency_output_dir"]))
    out = []
    for split in ["valid", "test"]:
        logit_rows = _read_pred(logit_dir / f"{split}_raw.jsonl")
        self_rows = _read_pred(self_dir / f"{split}_raw.jsonl")
        list_candidate_rows = _listwise_candidate_rows(list_rows_by_split[split])
        pair_candidate_rows = pair_candidates_by_split.get(split, [])
        methods: list[dict[str, Any]] = []
        if logit_rows:
            valid_logit = _read_pred(logit_dir / "valid_raw.jsonl")
            cal = _calibrate([_score(row, "positive_relevance_score") for row in valid_logit], [_label(row) for row in valid_logit], [_score(row, "positive_relevance_score") for row in logit_rows])
            for row, score in zip(logit_rows, cal):
                row["calibrated_pointwise_logit_ptrue"] = score
            methods.extend(
                [
                    {"method": "pointwise_logit_ptrue", "context_type": "pointwise", "rows": logit_rows, "score": "positive_relevance_score", "raw_ece": ece([_score(r, "positive_relevance_score") for r in logit_rows], [_label(r) for r in logit_rows]), "cal_ece": "", "cost": 1.0},
                    {"method": "calibrated_pointwise_logit_ptrue", "context_type": "pointwise", "rows": logit_rows, "score": "calibrated_pointwise_logit_ptrue", "raw_ece": "", "cal_ece": ece(cal, [_label(r) for r in logit_rows]), "cost": 1.0},
                ]
            )
        if self_rows:
            methods.append({"method": "self_consistency_true_frequency", "context_type": "pointwise", "rows": self_rows, "score": "recommend_true_frequency", "raw_ece": ece([_score(r, "recommend_true_frequency") for r in self_rows], [_label(r) for r in self_rows]), "cal_ece": "", "cost": 5.0})
        if list_candidate_rows:
            list_diag = listwise_diagnostics({split: list_rows_by_split[split]})[0]
            list_cal = listwise_calibration(list_rows_by_split["valid"], list_rows_by_split[split])
            cal_ece = next((row["ECE"] for row in list_cal if row["score"] == "top1_confidence" and row["score_type"] == "logistic_calibrated"), "")
            methods.extend(
                [
                    {"method": "listwise_ranking", "context_type": "listwise", "rows": list_candidate_rows, "score": "listwise_rank_score", "raw_ece": list_diag["top1_confidence_ECE_for_top1_correctness"], "cal_ece": "", "cost": 1.0 / 6.0},
                    {"method": "listwise_top1_confidence_calibrated", "context_type": "listwise", "rows": list_candidate_rows, "score": "listwise_rank_score", "raw_ece": "", "cal_ece": cal_ece, "cost": 1.0 / 6.0},
                ]
            )
        if pair_candidate_rows:
            pair_diag = pairwise_diagnostics({split: _read_pred(Path(str(cfg["pairwise_output_dir"])) / f"{split}_raw.jsonl")})[0]
            pair_cal = pairwise_calibration(_read_pred(Path(str(cfg["pairwise_output_dir"])) / "valid_raw.jsonl"), _read_pred(Path(str(cfg["pairwise_output_dir"])) / f"{split}_raw.jsonl"))
            cal_ece = next((row["ECE"] for row in pair_cal if row["score"] == "confidence" and row["score_type"] == "logistic_calibrated"), "")
            methods.extend(
                [
                    {"method": "pairwise_win_rate", "context_type": "pairwise", "rows": pair_candidate_rows, "score": "pairwise_win_rate", "raw_ece": pair_diag["pairwise_confidence_ECE_for_correctness"], "cal_ece": cal_ece, "cost": 5.0},
                    {"method": "pairwise_margin_score", "context_type": "pairwise", "rows": pair_candidate_rows, "score": "pairwise_margin_score", "raw_ece": pair_diag["preference_margin_ECE_for_correctness"], "cal_ece": cal_ece, "cost": 5.0},
                    {"method": "pairwise_confidence_weighted_score", "context_type": "pairwise", "rows": pair_candidate_rows, "score": "pairwise_confidence_weighted_score", "raw_ece": pair_diag["pairwise_confidence_ECE_for_correctness"], "cal_ece": cal_ece, "cost": 5.0},
                ]
            )
        for spec in methods:
            metric = _metric_bundle(spec["rows"], spec["score"])
            parse_rate = ""
            schema_rate = ""
            if spec["context_type"] == "listwise":
                parse_rate = _rate(list_rows_by_split[split], "parse_success")
                schema_rate = _rate(list_rows_by_split[split], "schema_valid")
            elif spec["context_type"] == "pairwise":
                pair_rows = _read_pred(Path(str(cfg["pairwise_output_dir"])) / f"{split}_raw.jsonl")
                parse_rate = _rate(pair_rows, "parse_success")
                schema_rate = _rate(pair_rows, "schema_valid")
            out.append({"method": spec["method"], "split": split, "context_type": spec["context_type"], **metric, "parse_success_rate": parse_rate, "schema_valid_rate": schema_rate, "raw_uncertainty_ECE": spec["raw_ece"], "calibrated_uncertainty_ECE": spec["cal_ece"], "cost_relative": spec["cost"], "recommendation": "candidate"})
        random_metrics = random_ranking_metrics(split_rows_by_split[split])
        out.append({"method": "random", "split": split, "context_type": "baseline", "num_users": len(_group_by_user(split_rows_by_split[split])), "num_rows_or_pairs": len(split_rows_by_split[split]), **random_metrics, "parse_success_rate": "", "schema_valid_rate": "", "raw_uncertainty_ECE": "", "calibrated_uncertainty_ECE": "", "AUROC": "NA", "score_std": "NA", "cost_relative": 0.0, "recommendation": "baseline"})
        out.append({"method": "oracle", "split": split, "context_type": "baseline", "num_users": len(_group_by_user(split_rows_by_split[split])), "num_rows_or_pairs": len(split_rows_by_split[split]), "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "parse_success_rate": "", "schema_valid_rate": "", "raw_uncertainty_ECE": "", "calibrated_uncertainty_ECE": "", "AUROC": 1.0, "score_std": "NA", "cost_relative": 0.0, "recommendation": "upper_bound"})
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def write_go_no_go(comp: list[dict[str, Any]]) -> str:
    test = [row for row in comp if row.get("split") == "test"]
    pointwise = next((row for row in test if row.get("method") == "pointwise_logit_ptrue"), {})
    candidates = [row for row in test if row.get("context_type") in {"listwise", "pairwise"} and row.get("method") not in {"listwise_top1_confidence_calibrated"}]
    best = max(candidates, key=lambda row: (_as_float(row.get("MRR")), _as_float(row.get("HR@1")), _as_float(row.get("NDCG@3"))), default={})
    recommendation = "keep_logit_ptrue_as_weak_primary_signal_and_move_to_evidence_observation"
    if best and pointwise:
        beats = (
            _as_float(best.get("MRR")) > _as_float(pointwise.get("MRR"))
            and _as_float(best.get("HR@1")) > _as_float(pointwise.get("HR@1"))
            and _as_float(best.get("NDCG@3")) > _as_float(pointwise.get("NDCG@3"))
        )
        cal_ece = _as_float(best.get("calibrated_uncertainty_ECE"), 1.0)
        if beats and cal_ece <= 0.08:
            recommendation = "use_relative_context_for_confidence_framework"
        elif beats:
            recommendation = "use_relative_ranking_score_but_not_raw_uncertainty"
        elif _as_float(pointwise.get("AUROC")) < 0.55 and (not best or _as_float(best.get("MRR")) <= _as_float(pointwise.get("MRR"))):
            recommendation = "current_lora_adapter_not_strong_enough_for_confidence_observation"
    text = f"""# Framework-Observation-Day1g Go/No-Go Decision

## Recommendation

`{recommendation}`

## Interpretation

Day1g compares pointwise, listwise, and pairwise signals on the same Beauty 100-user subset where possible. This remains observation only: no training, no evidence fields, no CEP, no external API, and no four-domain run.

## Test Snapshot

- pointwise logit P(true) MRR/HR@1/NDCG@3/AUROC: `{pointwise.get('MRR', 'NA')}` / `{pointwise.get('HR@1', 'NA')}` / `{pointwise.get('NDCG@3', 'NA')}` / `{pointwise.get('AUROC', 'NA')}`
- best relative-context method: `{best.get('method', 'NA')}`
- best relative-context MRR/HR@1/NDCG@3/AUROC: `{best.get('MRR', 'NA')}` / `{best.get('HR@1', 'NA')}` / `{best.get('NDCG@3', 'NA')}` / `{best.get('AUROC', 'NA')}`
"""
    GO_NO_GO_MD.write_text(text, encoding="utf-8")
    return recommendation


def write_report(comp: list[dict[str, Any]], recommendation: str) -> None:
    test = [row for row in comp if row.get("split") == "test"]
    def row(method: str) -> dict[str, Any]:
        return next((item for item in test if item.get("method") == method), {})
    report = f"""# Framework-Observation-Day1g Relative Context Audit Report

## Scope

Day1g tests whether relative candidate context improves local Qwen-LoRA confidence/relevance signals. It uses Beauty only, no training, no evidence fields, no CEP, no external API, and no four-domain run.

## Prior Observations

- Pointwise verbalized confidence collapsed.
- Pointwise logit P(true) is usable but weak.
- Self-consistency is not the primary confidence line. After tie-aware ranking fix, self-consistency no longer beats logit P(true). It is more expensive and weaker than logit on the same subset.

## Test Comparison

- pointwise logit P(true): MRR `{row('pointwise_logit_ptrue').get('MRR', 'NA')}`, HR@1 `{row('pointwise_logit_ptrue').get('HR@1', 'NA')}`, NDCG@3 `{row('pointwise_logit_ptrue').get('NDCG@3', 'NA')}`
- listwise ranking: MRR `{row('listwise_ranking').get('MRR', 'NA')}`, HR@1 `{row('listwise_ranking').get('HR@1', 'NA')}`, NDCG@3 `{row('listwise_ranking').get('NDCG@3', 'NA')}`
- pairwise win rate: MRR `{row('pairwise_win_rate').get('MRR', 'NA')}`, HR@1 `{row('pairwise_win_rate').get('HR@1', 'NA')}`, NDCG@3 `{row('pairwise_win_rate').get('NDCG@3', 'NA')}`

## Recommendation

`{recommendation}`
"""
    LIST_REPORT_MD.write_text(report, encoding="utf-8")


def analyze(cfg: dict[str, Any]) -> None:
    split_rows_by_split = {split: _split_rows(cfg, split) for split in ["valid", "test"]}
    list_dir = Path(str(cfg["listwise_output_dir"]))
    pair_dir = Path(str(cfg["pairwise_output_dir"]))
    list_rows_by_split = {split: _read_pred(list_dir / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    pair_rows_by_split = {split: _read_pred(pair_dir / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    if not all(list_rows_by_split.values()):
        raise FileNotFoundError("Listwise predictions are required for Day1g analysis.")

    list_diag = listwise_diagnostics(list_rows_by_split)
    list_cal = listwise_calibration(list_rows_by_split["valid"], list_rows_by_split["test"])
    _write_csv(LIST_DIAG_CSV, list_diag)
    _write_csv(LIST_CAL_CSV, list_cal)

    pair_candidates_by_split: dict[str, list[dict[str, Any]]] = {}
    if all(pair_rows_by_split.values()):
        pair_diag = pairwise_diagnostics(pair_rows_by_split)
        pair_cal = pairwise_calibration(pair_rows_by_split["valid"], pair_rows_by_split["test"])
        pair_rank, pair_candidates_by_split = pairwise_ranking_rows(pair_rows_by_split, split_rows_by_split)
        _write_csv(PAIR_DIAG_CSV, pair_diag)
        _write_csv(PAIR_CAL_CSV, pair_cal)
        _write_csv(PAIR_RANK_CSV, pair_rank)
    else:
        _write_csv(PAIR_DIAG_CSV, [{"status": "pending_pairwise_predictions"}])
        _write_csv(PAIR_CAL_CSV, [{"status": "pending_pairwise_predictions"}])
        _write_csv(PAIR_RANK_CSV, [{"status": "pending_pairwise_predictions"}])

    comp = context_comparison(cfg, split_rows_by_split, list_rows_by_split, pair_candidates_by_split)
    _write_csv(COMPARISON_CSV, comp)
    recommendation = write_go_no_go(comp)
    write_report(comp, recommendation)
    print(json.dumps({"listwise_diagnostics": str(LIST_DIAG_CSV), "listwise_calibration": str(LIST_CAL_CSV), "pairwise_diagnostics": str(PAIR_DIAG_CSV), "pairwise_calibration": str(PAIR_CAL_CSV), "pairwise_ranking": str(PAIR_RANK_CSV), "comparison": str(COMPARISON_CSV), "report": str(LIST_REPORT_MD), "go_no_go": str(GO_NO_GO_MD)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1g list/pair context audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run_listwise", choices=["valid", "test"], default=None)
    parser.add_argument("--run_pairwise", choices=["valid", "test"], default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _read_config(args.config)
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    if args.run_listwise:
        path = run_listwise(cfg, args.run_listwise, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_listwise, "mode": "listwise"}, ensure_ascii=False, indent=2))
    if args.run_pairwise:
        path = run_pairwise(cfg, args.run_pairwise, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_pairwise, "mode": "pairwise"}, ensure_ascii=False, indent=2))
    if args.analyze_only:
        analyze(cfg)


if __name__ == "__main__":
    main()
