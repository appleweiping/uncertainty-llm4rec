from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
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


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_day1h_behavioral_uncertainty.yaml"
DIAG_CSV = Path("data_done/framework_observation_day1h_listwise_behavioral_uncertainty_diagnostics.csv")
CAL_CSV = Path("data_done/framework_observation_day1h_listwise_behavioral_uncertainty_calibration.csv")
RANK_CSV = Path("data_done/framework_observation_day1h_listwise_behavioral_uncertainty_ranking_eval.csv")
REPORT_MD = Path("data_done/framework_observation_day1h_listwise_behavioral_uncertainty_report.md")
COMPARISON_CSV = Path("data_done/framework_observation_day1h_context_comparison.csv")
GO_NO_GO_MD = Path("data_done/framework_observation_day1h_go_no_go_decision.md")


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


def _split_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    subset = load_subset(cfg)
    user_ids = _subset_entries(subset)[split]["sampled_user_ids"]
    return _selected_rows(Path(str(cfg[f"{split}_file"])), user_ids)


def _prompt(user_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    template = Path(str(cfg["prompt_template"])).read_text(encoding="utf-8").strip()
    payload = {
        "user_history": _history_payload(user_rows[0], cfg),
        "candidate_pool": [_candidate_payload(row, cfg) for row in user_rows],
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


def _parse_ranking(raw_text: str, candidate_ids: list[str]) -> dict[str, Any]:
    obj = _json_obj(raw_text)
    if obj is None:
        return {"parse_success": False, "schema_valid": False, "ranked_item_ids": []}
    ranked = obj.get("ranked_item_ids")
    ranked_ids = [str(x) for x in ranked] if isinstance(ranked, list) else []
    candidate_set = set(candidate_ids)
    complete = (
        len(ranked_ids) == len(candidate_ids)
        and len(set(ranked_ids)) == len(ranked_ids)
        and set(ranked_ids) == candidate_set
    )
    return {"parse_success": True, "schema_valid": complete, "ranked_item_ids": ranked_ids}


def run_inference(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore

    rows = _split_rows(cfg, split)
    grouped = _group_by_user(rows)
    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()

    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    batch_size = int(cfg.get("vllm_batch_size", 12))
    n = int(cfg.get("num_samples", 5))
    seed = int(cfg.get("seed", 42))
    pending: list[tuple[str, list[dict[str, Any]], str]] = []
    for user_id, user_rows in grouped.items():
        sid = f"{split}_{user_id}"
        if sid not in finished:
            pending.append((sid, user_rows, _prompt(user_rows, cfg)))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        raw_by_user: list[list[str]] = [[] for _ in batch]
        parsed_by_user: list[list[dict[str, Any]]] = [[] for _ in batch]
        for sample_idx in range(n):
            sampling_params = SamplingParams(
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.9)),
                max_tokens=int(cfg.get("max_new_tokens", 160)),
                seed=seed + sample_idx,
            )
            outputs = llm.generate([item[2] for item in batch], sampling_params, lora_request=lora_request)
            for idx, ((_, user_rows, _), output) in enumerate(zip(batch, outputs)):
                raw_text = output.outputs[0].text if output.outputs else ""
                candidate_ids = [str(row.get("candidate_item_id", "")) for row in user_rows]
                raw_by_user[idx].append(raw_text)
                parsed_by_user[idx].append(_parse_ranking(raw_text, candidate_ids))

        out_rows = []
        for (sid, user_rows, _), raw_generations, parsed in zip(batch, raw_by_user, parsed_by_user):
            candidate_ids = [str(row.get("candidate_item_id", "")) for row in user_rows]
            labels_by_item = {str(row.get("candidate_item_id", "")): int(row.get("label", 0)) for row in user_rows}
            positive_item_id = next((item_id for item_id, label in labels_by_item.items() if label == 1), "")
            valid_rankings = [row["ranked_item_ids"] for row in parsed if row["schema_valid"]]
            out_rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "user_id": user_rows[0].get("user_id", ""),
                    "candidate_item_ids": candidate_ids,
                    "labels_by_item": labels_by_item,
                    "positive_item_id": positive_item_id,
                    "num_samples": n,
                    "parse_success_count": sum(1 for row in parsed if row["parse_success"]),
                    "schema_valid_count": len(valid_rankings),
                    "parse_success_rate": sum(1 for row in parsed if row["parse_success"]) / n if n else 0.0,
                    "schema_valid_rate": len(valid_rankings) / n if n else 0.0,
                    "rankings": valid_rankings,
                    "raw_generations": raw_generations,
                    "context_type": "listwise_behavioral",
                    "inference_backend": "vllm",
                }
            )
        _write_jsonl(output_path, out_rows)
    return output_path


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return -sum(p * math.log(p, 2) for p in probs)


def _behavior_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        candidate_ids = [str(x) for x in row.get("candidate_item_ids", [])]
        labels = {str(k): int(v) for k, v in dict(row.get("labels_by_item", {})).items()}
        rankings = [[str(x) for x in ranking] for ranking in row.get("rankings", [])]
        n_valid = len(rankings)
        top1_counts = Counter(ranking[0] for ranking in rankings if ranking)
        majority_top1, top1_count = ("", 0)
        if top1_counts:
            majority_top1, top1_count = top1_counts.most_common(1)[0]
        majority_top1_confidence = top1_count / n_valid if n_valid else 0.0
        top1_vote_entropy = _entropy(list(top1_counts.values()))
        positive_item_id = str(row.get("positive_item_id", ""))
        positive_ranks = []
        ranks_by_candidate: dict[str, list[int]] = {item_id: [] for item_id in candidate_ids}
        for ranking in rankings:
            for idx, item_id in enumerate(ranking, 1):
                ranks_by_candidate.setdefault(item_id, []).append(idx)
            if positive_item_id in ranking:
                positive_ranks.append(ranking.index(positive_item_id) + 1)
        all_rank_stds = [pstdev(values) for values in ranks_by_candidate.values() if len(values) > 1]
        rank_entropy_values = []
        for item_id in candidate_ids:
            rank_counts = Counter(ranks_by_candidate.get(item_id, []))
            rank_entropy_values.append(_entropy(list(rank_counts.values())))
        rank_entropy = mean(rank_entropy_values) if rank_entropy_values else 0.0
        rank_variance = mean([std**2 for std in all_rank_stds]) if all_rank_stds else 0.0
        positive_rank_mean = mean(positive_ranks) if positive_ranks else 0.0
        positive_rank_std = pstdev(positive_ranks) if len(positive_ranks) > 1 else 0.0
        top1_correct = int(labels.get(majority_top1, 0) == 1) if majority_top1 else 0
        out.append(
            {
                "sample_id": row.get("sample_id", ""),
                "split": row.get("split", ""),
                "user_id": row.get("user_id", ""),
                "candidate_item_ids": candidate_ids,
                "labels_by_item": labels,
                "positive_item_id": positive_item_id,
                "num_samples": int(row.get("num_samples", 0)),
                "parse_success_rate": _score(row, "parse_success_rate"),
                "schema_valid_rate": _score(row, "schema_valid_rate"),
                "majority_top1_item_id": majority_top1,
                "majority_top1_correct": top1_correct,
                "top1_frequency": majority_top1_confidence,
                "majority_top1_confidence": majority_top1_confidence,
                "rank_stability_uncertainty": 1.0 - majority_top1_confidence,
                "top1_vote_entropy": top1_vote_entropy,
                "rank_entropy": rank_entropy,
                "rank_variance": rank_variance,
                "positive_rank_mean": positive_rank_mean,
                "positive_rank_std": positive_rank_std,
                "candidate_rank_means": {item_id: (mean(v) if v else 0.0) for item_id, v in ranks_by_candidate.items()},
                "candidate_rank_stds": {item_id: (pstdev(v) if len(v) > 1 else 0.0) for item_id, v in ranks_by_candidate.items()},
            }
        )
    return out


def _candidate_score_rows(behavior_rows: list[dict[str, Any]], use_stability: bool = True) -> list[dict[str, Any]]:
    out = []
    for row in behavior_rows:
        labels = dict(row.get("labels_by_item", {}))
        rank_means = dict(row.get("candidate_rank_means", {}))
        rank_stds = dict(row.get("candidate_rank_stds", {}))
        for item_id in row.get("candidate_item_ids", []):
            rank_mean = float(rank_means.get(item_id, 0.0))
            rank_std = float(rank_stds.get(item_id, 0.0))
            rank_score = 1.0 / rank_mean if rank_mean > 0 else 0.0
            stability_weighted = rank_score / (1.0 + rank_std) if use_stability else rank_score
            out.append(
                {
                    "user_id": row.get("user_id", ""),
                    "candidate_item_id": item_id,
                    "label": int(labels.get(item_id, 0)),
                    "rank_score": rank_score,
                    "stability_weighted_rank_score": stability_weighted,
                    "rank_std": rank_std,
                }
            )
    return out


def _inverse_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    if max_value <= 0:
        return [1.0 for _ in values]
    return [1.0 - min(max(v / max_value, 0.0), 1.0) for v in values]


def diagnostics(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    diag = []
    behavior_by_split = {split: _behavior_rows(rows) for split, rows in rows_by_split.items()}
    candidate_by_split = {split: _candidate_score_rows(rows) for split, rows in behavior_by_split.items()}
    for split, behavior in behavior_by_split.items():
        candidates = candidate_by_split[split]
        top_correct = [int(row.get("majority_top1_correct", 0)) for row in behavior]
        top_conf = [_score(row, "majority_top1_confidence") for row in behavior]
        entropy = [_score(row, "top1_vote_entropy") for row in behavior]
        rank_entropy = [_score(row, "rank_entropy") for row in behavior]
        rank_variance = [_score(row, "rank_variance") for row in behavior]
        inv_entropy = _inverse_scores(entropy)
        inv_rank_entropy = _inverse_scores(rank_entropy)
        inv_rank_variance = _inverse_scores(rank_variance)
        rank = ranking_metrics(candidates, "rank_score")
        diag.append(
            {
                "split": split,
                "num_users": len(behavior),
                "parse_success_rate": mean([_score(row, "parse_success_rate") for row in behavior]) if behavior else 0.0,
                "schema_valid_rate": mean([_score(row, "schema_valid_rate") for row in behavior]) if behavior else 0.0,
                **rank,
                "top1_frequency_mean": mean(top_conf) if top_conf else 0.0,
                "top1_frequency_std": pstdev(top_conf) if len(top_conf) > 1 else 0.0,
                "majority_top1_confidence_AUROC": auroc(top_conf, top_correct),
                "majority_top1_confidence_ECE": ece(top_conf, top_correct),
                "majority_top1_confidence_Brier": brier(top_conf, top_correct),
                "top1_vote_entropy_mean": mean(entropy) if entropy else 0.0,
                "top1_vote_entropy_AUROC_for_error_risk": auroc(entropy, [1 - y for y in top_correct]),
                "top1_vote_entropy_inverse_AUROC_for_correctness": auroc(inv_entropy, top_correct),
                "rank_entropy_mean": mean(rank_entropy) if rank_entropy else 0.0,
                "rank_entropy_inverse_AUROC_for_correctness": auroc(inv_rank_entropy, top_correct),
                "rank_variance_mean": mean(rank_variance) if rank_variance else 0.0,
                "rank_variance_inverse_AUROC_for_correctness": auroc(inv_rank_variance, top_correct),
                "positive_rank_mean": mean([_score(row, "positive_rank_mean") for row in behavior]) if behavior else 0.0,
                "positive_rank_std_mean": mean([_score(row, "positive_rank_std") for row in behavior]) if behavior else 0.0,
            }
        )
    return diag, behavior_by_split, candidate_by_split


def calibration_rows(behavior_by_split: dict[str, list[dict[str, Any]]], candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    valid_candidates = candidate_by_split["valid"]
    test_candidates = candidate_by_split["test"]
    valid_scores = [_score(row, "rank_score") for row in valid_candidates]
    valid_labels = [_label(row) for row in valid_candidates]
    test_scores = [_score(row, "rank_score") for row in test_candidates]
    test_labels = [_label(row) for row in test_candidates]
    calibrated = _calibrate(valid_scores, valid_labels, test_scores)
    raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
    cal = {"ECE": ece(calibrated, test_labels), "Brier": brier(calibrated, test_labels), "AUROC": auroc(calibrated, test_labels)}
    out.append({"target": "rank_score_to_relevance_label", "score_type": "raw", **raw, "calibration_method": "none"})
    out.append({"target": "rank_score_to_relevance_label", "score_type": "logistic_calibrated", **cal, "calibration_method": "valid_logistic", "delta_ECE_vs_raw": cal["ECE"] - raw["ECE"], "delta_Brier_vs_raw": cal["Brier"] - raw["Brier"]})

    uncertainty_specs = [
        ("majority_top1_confidence_to_top1_correctness", "majority_top1_confidence", False),
        ("top1_vote_entropy_to_error_risk", "top1_vote_entropy", True),
        ("rank_entropy_to_error_risk", "rank_entropy", True),
        ("rank_variance_to_error_risk", "rank_variance", True),
    ]
    for target, key, error_target in uncertainty_specs:
        valid_scores = [_score(row, key) for row in behavior_by_split["valid"]]
        test_scores = [_score(row, key) for row in behavior_by_split["test"]]
        if error_target:
            valid_labels = [1 - int(row.get("majority_top1_correct", 0)) for row in behavior_by_split["valid"]]
            test_labels = [1 - int(row.get("majority_top1_correct", 0)) for row in behavior_by_split["test"]]
        else:
            valid_labels = [int(row.get("majority_top1_correct", 0)) for row in behavior_by_split["valid"]]
            test_labels = [int(row.get("majority_top1_correct", 0)) for row in behavior_by_split["test"]]
        calibrated = _calibrate(valid_scores, valid_labels, test_scores)
        raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
        cal = {"ECE": ece(calibrated, test_labels), "Brier": brier(calibrated, test_labels), "AUROC": auroc(calibrated, test_labels)}
        out.append({"target": target, "score_type": "raw", **raw, "calibration_method": "none"})
        out.append({"target": target, "score_type": "logistic_calibrated", **cal, "calibration_method": "valid_logistic", "delta_ECE_vs_raw": cal["ECE"] - raw["ECE"], "delta_Brier_vs_raw": cal["Brier"] - raw["Brier"]})
    return out


def ranking_rows(candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in candidate_by_split.items():
        random_metrics = random_ranking_metrics(rows)
        for method in ["rank_score", "stability_weighted_rank_score"]:
            out.append({"split": split, "ranking_method": method, **ranking_metrics(rows, method), "random_baseline": json.dumps(random_metrics, sort_keys=True), "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "random", **random_metrics, "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "oracle", "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "oracle_upper_bound": 1.0})
    return out


def comparison_rows(cfg: dict[str, Any], candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    point_dir = Path(str(cfg["pointwise_logit_output_dir"]))
    day1g_dir = Path(str(cfg["day1g_listwise_output_dir"]))
    for split in ["valid", "test"]:
        point_rows = _read_pred(point_dir / f"{split}_raw.jsonl")
        day1h_rows = candidate_by_split[split]
        day1g_rows = _read_pred(day1g_dir / f"{split}_raw.jsonl")
        day1g_candidates = []
        for row in day1g_rows:
            labels = dict(row.get("labels_by_item", {}))
            ranked = [str(x) for x in row.get("ranked_item_ids", [])]
            candidate_ids = [str(x) for x in row.get("candidate_item_ids", [])]
            scores = {item_id: len(candidate_ids) - idx for idx, item_id in enumerate(ranked)} if row.get("schema_valid") else {}
            for item_id in candidate_ids:
                day1g_candidates.append({"user_id": row.get("user_id", ""), "candidate_item_id": item_id, "label": int(labels.get(item_id, 0)), "listwise_rank_score": float(scores.get(item_id, 0.0))})
        methods = [
            ("pointwise_logit_ptrue", "pointwise", point_rows, "positive_relevance_score"),
            ("day1g_single_listwise_rank_score", "listwise", day1g_candidates, "listwise_rank_score"),
            ("day1h_behavioral_rank_score", "listwise_behavioral", day1h_rows, "rank_score"),
            ("day1h_stability_weighted_rank_score", "listwise_behavioral", day1h_rows, "stability_weighted_rank_score"),
        ]
        for method, context_type, rows, score_key in methods:
            if not rows:
                continue
            scores = [_score(row, score_key) for row in rows]
            labels = [_label(row) for row in rows]
            out.append({"split": split, "method": method, "context_type": context_type, "num_users": len(_group_by_user(rows)), "num_rows": len(rows), **ranking_metrics(rows, score_key), "AUROC": auroc(scores, labels), "score_std": pstdev(scores) if len(scores) > 1 else 0.0})
        random_metrics = random_ranking_metrics(day1h_rows)
        out.append({"split": split, "method": "random", "context_type": "baseline", "num_users": len(_group_by_user(day1h_rows)), "num_rows": len(day1h_rows), **random_metrics, "AUROC": "NA", "score_std": "NA"})
        out.append({"split": split, "method": "oracle", "context_type": "baseline", "num_users": len(_group_by_user(day1h_rows)), "num_rows": len(day1h_rows), "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "AUROC": 1.0, "score_std": "NA"})
    return out


def write_report(diag: list[dict[str, Any]], cal: list[dict[str, Any]], comp: list[dict[str, Any]]) -> str:
    test_diag = next((row for row in diag if row.get("split") == "test"), {})
    test_comp = [row for row in comp if row.get("split") == "test"]
    day1h = next((row for row in test_comp if row.get("method") == "day1h_behavioral_rank_score"), {})
    day1g = next((row for row in test_comp if row.get("method") == "day1g_single_listwise_rank_score"), {})
    point = next((row for row in test_comp if row.get("method") == "pointwise_logit_ptrue"), {})
    stability_auc = test_diag.get("majority_top1_confidence_AUROC", "NA")
    recommendation = "keep_listwise_ranking_score_not_behavioral_uncertainty"
    if isinstance(stability_auc, float) and stability_auc > 0.6:
        recommendation = "use_rank_stability_as_behavioral_uncertainty_candidate"
    text = f"""# Framework-Observation-Day1h Listwise Behavioral Uncertainty Report

## Scope

This is still Framework Observation. Day1h does not train, use evidence, implement CEP, call external APIs, or run four domains.

## Motivation

Relative listwise context improves recommendation ranking, while raw verbalized uncertainty collapses. Day1h tests whether behavioral uncertainty from repeated listwise rankings can replace self-reported confidence.

## Test Ranking Snapshot

- pointwise logit P(true) MRR/HR@1/NDCG@3: `{point.get('MRR', 'NA')}` / `{point.get('HR@1', 'NA')}` / `{point.get('NDCG@3', 'NA')}`
- Day1g single listwise MRR/HR@1/NDCG@3: `{day1g.get('MRR', 'NA')}` / `{day1g.get('HR@1', 'NA')}` / `{day1g.get('NDCG@3', 'NA')}`
- Day1h behavioral rank score MRR/HR@1/NDCG@3: `{day1h.get('MRR', 'NA')}` / `{day1h.get('HR@1', 'NA')}` / `{day1h.get('NDCG@3', 'NA')}`

## Behavioral Uncertainty

- majority_top1_confidence AUROC for top1 correctness: `{test_diag.get('majority_top1_confidence_AUROC', 'NA')}`
- top1 vote entropy AUROC for error risk: `{test_diag.get('top1_vote_entropy_AUROC_for_error_risk', 'NA')}`
- rank entropy inverse AUROC for correctness: `{test_diag.get('rank_entropy_inverse_AUROC_for_correctness', 'NA')}`
- rank variance inverse AUROC for correctness: `{test_diag.get('rank_variance_inverse_AUROC_for_correctness', 'NA')}`

## Recommendation

`{recommendation}`
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    GO_NO_GO_MD.write_text(f"# Framework-Observation-Day1h Go/No-Go Decision\n\n`{recommendation}`\n\nThis remains observation only. Do not enter CEP/evidence from Day1h alone.\n", encoding="utf-8")
    return recommendation


def analyze(cfg: dict[str, Any]) -> None:
    pred_dir = Path(str(cfg["output_dir"]))
    rows_by_split = {split: _read_pred(pred_dir / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    if not all(rows_by_split.values()):
        raise FileNotFoundError("Day1h valid/test raw predictions are required for analysis.")
    diag, behavior_by_split, candidate_by_split = diagnostics(rows_by_split)
    cal = calibration_rows(behavior_by_split, candidate_by_split)
    rank = ranking_rows(candidate_by_split)
    comp = comparison_rows(cfg, candidate_by_split)
    _write_csv(DIAG_CSV, diag)
    _write_csv(CAL_CSV, cal)
    _write_csv(RANK_CSV, rank)
    _write_csv(COMPARISON_CSV, comp)
    recommendation = write_report(diag, cal, comp)
    print(json.dumps({"diagnostics": str(DIAG_CSV), "calibration": str(CAL_CSV), "ranking": str(RANK_CSV), "comparison": str(COMPARISON_CSV), "report": str(REPORT_MD), "go_no_go": str(GO_NO_GO_MD), "recommendation": recommendation}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1h listwise behavioral uncertainty audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run_inference", choices=["valid", "test"], default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _read_config(args.config)
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    if args.run_inference:
        path = run_inference(cfg, args.run_inference, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_inference, "mode": "listwise_behavioral_uncertainty"}, ensure_ascii=False, indent=2))
    if args.analyze_only:
        analyze(cfg)


if __name__ == "__main__":
    main()
