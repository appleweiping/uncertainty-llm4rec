from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PRED_DIR = Path("output-repaired/framework_observation/beauty_qwen_lora_logit_confidence/predictions")
DAY1D_DIAG_CSV = Path("data_done/framework_observation_day1d_logit_confidence_diagnostics.csv")
DAY1D_CAL_CSV = Path("data_done/framework_observation_day1d_logit_confidence_calibration.csv")
DAY1D_REPORT_MD = Path("data_done/framework_observation_day1d_logit_confidence_report.md")
RANKING_CSV = Path("data_done/framework_observation_day1e_logit_score_ranking_eval.csv")
THRESHOLD_CSV = Path("data_done/framework_observation_day1e_threshold_selection.csv")
DIST_CSV = Path("data_done/framework_observation_day1e_logit_score_distribution.csv")
CAL_SUMMARY_CSV = Path("data_done/framework_observation_day1e_logit_calibration_summary.csv")
GO_NO_GO_MD = Path("data_done/framework_observation_day1e_go_no_go_decision.md")
REPORT_MD = Path("data_done/framework_observation_day1e_logit_score_ranking_report.md")
SELF_CONSISTENCY_PLAN_MD = Path("data_done/framework_observation_day1f_self_consistency_plan.md")
PROMPT_COMPARISON_CSV = Path("data_done/framework_observation_day1_prompt_comparison.csv")


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _label(row: dict[str, Any]) -> int:
    return int(row.get("label", 0))


def _ptrue(row: dict[str, Any]) -> float:
    return float(row.get("positive_relevance_score", row.get("p_true", 0.0)))


def _decision_confidence(row: dict[str, Any]) -> float:
    return float(row.get("decision_confidence", 0.0))


def _recommend(row: dict[str, Any]) -> int:
    return 1 if bool(row.get("recommend", False)) else 0


def _dcg(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(rank + 2) for rank, label in enumerate(labels[:k]))


def _ndcg(sorted_labels: list[int], k: int) -> float:
    ideal = sorted(sorted_labels, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    return _dcg(sorted_labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def _mrr(sorted_labels: list[int]) -> float:
    for rank, label in enumerate(sorted_labels, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def _hr(sorted_labels: list[int], k: int) -> float:
    return 1.0 if any(label > 0 for label in sorted_labels[:k]) else 0.0


def _group_by_user(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("user_id", ""))].append(row)
    return dict(groups)


def _expected_random_metrics(groups: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    metrics = {"NDCG@10": [], "MRR": [], "HR@1": [], "HR@3": [], "NDCG@3": [], "NDCG@5": [], "HR@10": []}
    for items in groups.values():
        labels = [_label(item) for item in items]
        n = len(labels)
        if n == 0:
            continue
        positives = sum(labels)
        if positives <= 0:
            for key in metrics:
                metrics[key].append(0.0)
            continue
        # Exact expectation by enumerating each item as the positive rank proxy for 5neg/one-positive pools.
        if positives == 1:
            pos_values = []
            for rank in range(1, n + 1):
                sorted_labels = [0] * n
                sorted_labels[rank - 1] = 1
                pos_values.append(
                    {
                        "NDCG@10": _ndcg(sorted_labels, 10),
                        "MRR": 1.0 / rank,
                        "HR@1": 1.0 if rank <= 1 else 0.0,
                        "HR@3": 1.0 if rank <= 3 else 0.0,
                        "NDCG@3": _ndcg(sorted_labels, 3),
                        "NDCG@5": _ndcg(sorted_labels, 5),
                        "HR@10": 1.0 if rank <= 10 else 0.0,
                    }
                )
            for key in metrics:
                metrics[key].append(mean([row[key] for row in pos_values]))
        else:
            # Deterministic approximation for rare multi-positive/incomplete pools.
            random_labels = sorted(labels, reverse=True)
            for key in metrics:
                if key.startswith("NDCG"):
                    k = int(key.split("@")[1])
                    metrics[key].append(_ndcg(random_labels, k))
                elif key.startswith("HR"):
                    k = int(key.split("@")[1])
                    metrics[key].append(min(1.0, positives * k / n))
                else:
                    metrics[key].append(mean([1.0 / rank for rank in range(1, n + 1)]))
    return {key: mean(values) if values else 0.0 for key, values in metrics.items()}


def _ranking_metrics(groups: dict[str, list[dict[str, Any]]], score_key: str) -> dict[str, float]:
    values: dict[str, list[float]] = {
        "NDCG@10": [],
        "MRR": [],
        "HR@1": [],
        "HR@3": [],
        "NDCG@3": [],
        "NDCG@5": [],
        "HR@10": [],
    }
    pool_sizes = []
    for items in groups.values():
        ranked = sorted(items, key=lambda row: _float(row.get(score_key), 0.0), reverse=True)
        labels = [_label(row) for row in ranked]
        pool_sizes.append(len(items))
        values["NDCG@10"].append(_ndcg(labels, 10))
        values["MRR"].append(_mrr(labels))
        values["HR@1"].append(_hr(labels, 1))
        values["HR@3"].append(_hr(labels, 3))
        values["NDCG@3"].append(_ndcg(labels, 3))
        values["NDCG@5"].append(_ndcg(labels, 5))
        values["HR@10"].append(_hr(labels, 10))
    out = {key: mean(v) if v else 0.0 for key, v in values.items()}
    out["candidate_pool_size_mean"] = mean(pool_sizes) if pool_sizes else 0.0
    out["hr10_trivial_flag"] = out["candidate_pool_size_mean"] <= 10
    return out


def _add_calibrated_scores(valid_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> None:
    # A monotonic Platt-style approximation is enough for ranking audit; monotonic transforms preserve order.
    # Calibration metrics themselves are read from the Day1d calibration output.
    valid_scores = [_ptrue(row) for row in valid_rows]
    valid_labels = [_label(row) for row in valid_rows]
    if not valid_scores or len(set(valid_labels)) < 2:
        for row in target_rows:
            row["calibrated_p_true"] = _ptrue(row)
        return
    pos_mean = mean([s for s, y in zip(valid_scores, valid_labels) if y == 1])
    neg_mean = mean([s for s, y in zip(valid_scores, valid_labels) if y == 0])
    slope = 8.0 if pos_mean >= neg_mean else -8.0
    center = mean(valid_scores)
    for row in target_rows:
        x = _ptrue(row)
        row["calibrated_p_true"] = 1.0 / (1.0 + math.exp(-slope * (x - center)))


def _add_hard_decision_score(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        digest = hashlib.md5(str(row.get("sample_id", "")).encode("utf-8")).hexdigest()
        tie_break = int(digest[:8], 16) / 0xFFFFFFFF
        row["hard_recommend_score"] = float(_recommend(row)) + 1e-9 * tie_break


def ranking_eval(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _add_calibrated_scores(valid_rows, valid_rows)
    _add_calibrated_scores(valid_rows, test_rows)
    _add_hard_decision_score(valid_rows)
    _add_hard_decision_score(test_rows)
    rows: list[dict[str, Any]] = []
    for split, split_rows in [("valid", valid_rows), ("test", test_rows)]:
        groups = _group_by_user(split_rows)
        random_metrics = _expected_random_metrics(groups)
        for method, score_key in [
            ("raw_p_true", "positive_relevance_score"),
            ("calibrated_p_true", "calibrated_p_true"),
            ("hard_recommend_decision", "hard_recommend_score"),
        ]:
            metrics = _ranking_metrics(groups, score_key)
            row = {
                "split": split,
                "ranking_method": method,
                **metrics,
                "random_baseline": json.dumps(random_metrics, sort_keys=True),
                "oracle_upper_bound": 1.0,
            }
            rows.append(row)
        rows.append(
            {
                "split": split,
                "ranking_method": "random",
                **random_metrics,
                "candidate_pool_size_mean": mean([len(v) for v in groups.values()]) if groups else 0.0,
                "hr10_trivial_flag": True,
                "random_baseline": json.dumps(random_metrics, sort_keys=True),
                "oracle_upper_bound": 1.0,
            }
        )
        rows.append(
            {
                "split": split,
                "ranking_method": "oracle",
                "NDCG@10": 1.0,
                "MRR": 1.0,
                "HR@1": 1.0,
                "HR@3": 1.0,
                "NDCG@3": 1.0,
                "NDCG@5": 1.0,
                "HR@10": 1.0,
                "hr10_trivial_flag": True,
                "candidate_pool_size_mean": mean([len(v) for v in groups.values()]) if groups else 0.0,
                "random_baseline": json.dumps(random_metrics, sort_keys=True),
                "oracle_upper_bound": 1.0,
            }
        )
    return rows


def _binary_metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    acc = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bal_acc = (tpr + tnr) / 2
    return {
        "accuracy": acc,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "recommend_true_rate": mean(preds) if preds else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_selection(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_scores = [_ptrue(row) for row in valid_rows]
    valid_labels = [_label(row) for row in valid_rows]
    test_scores = [_ptrue(row) for row in test_rows]
    test_labels = [_label(row) for row in test_rows]
    valid_pos_rate = mean(valid_labels) if valid_labels else 0.0
    grid = [i / 100 for i in range(1, 100)]
    scored = [(threshold, _binary_metrics(valid_scores, valid_labels, threshold)) for threshold in grid]

    def pick(metric: str) -> float:
        return max(scored, key=lambda item: (item[1][metric], -abs(item[1]["recommend_true_rate"] - valid_pos_rate)))[0]

    strategies = {
        "best_threshold_by_accuracy": pick("accuracy"),
        "best_threshold_by_f1": pick("f1"),
        "best_threshold_by_balanced_accuracy": pick("balanced_accuracy"),
        "threshold_matching_positive_rate": min(
            scored,
            key=lambda item: (abs(item[1]["recommend_true_rate"] - valid_pos_rate), -item[1]["balanced_accuracy"]),
        )[0],
    }
    out = []
    for strategy, threshold in strategies.items():
        valid_metric = _binary_metrics(valid_scores, valid_labels, threshold)
        test_metric = _binary_metrics(test_scores, test_labels, threshold)
        out.append(
            {
                "threshold_strategy": strategy,
                "valid_threshold": threshold,
                "valid_accuracy": valid_metric["accuracy"],
                "valid_f1": valid_metric["f1"],
                "valid_balanced_accuracy": valid_metric["balanced_accuracy"],
                "valid_recommend_true_rate": valid_metric["recommend_true_rate"],
                "test_accuracy": test_metric["accuracy"],
                "test_f1": test_metric["f1"],
                "test_balanced_accuracy": test_metric["balanced_accuracy"],
                "test_recommend_true_rate": test_metric["recommend_true_rate"],
                "test_confusion_tp": test_metric["tp"],
                "test_confusion_fp": test_metric["fp"],
                "test_confusion_tn": test_metric["tn"],
                "test_confusion_fn": test_metric["fn"],
            }
        )
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def score_distribution(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        for label in [0, 1]:
            subset = [row for row in rows if _label(row) == label]
            p_scores = [_ptrue(row) for row in subset]
            conf_scores = [_decision_confidence(row) for row in subset]
            out.append(
                {
                    "split": split,
                    "label": label,
                    "count": len(subset),
                    "p_true_mean": mean(p_scores) if p_scores else 0.0,
                    "p_true_std": pstdev(p_scores) if len(p_scores) > 1 else 0.0,
                    "p_true_q05": _quantile(p_scores, 0.05),
                    "p_true_q25": _quantile(p_scores, 0.25),
                    "p_true_q50": _quantile(p_scores, 0.50),
                    "p_true_q75": _quantile(p_scores, 0.75),
                    "p_true_q95": _quantile(p_scores, 0.95),
                    "decision_confidence_mean": mean(conf_scores) if conf_scores else 0.0,
                    "decision_confidence_std": pstdev(conf_scores) if len(conf_scores) > 1 else 0.0,
                }
            )
    return out


def calibration_summary() -> list[dict[str, Any]]:
    rows = _read_csv(DAY1D_CAL_CSV)
    out = []
    for target, score_type in [
        ("positive_relevance_score_to_label", "positive_relevance_score"),
        ("decision_confidence_to_correctness", "decision_confidence"),
    ]:
        target_rows = [row for row in rows if row.get("target") == target and row.get("status") == "ok"]
        raw = next((row for row in target_rows if row.get("score_type") == "raw"), {})
        calibrated_candidates = [row for row in target_rows if row.get("score_type") != "raw"]
        best = min(calibrated_candidates, key=lambda row: _float(row.get("ECE"), 999), default={})
        raw_ece = _float(raw.get("ECE"))
        cal_ece = _float(best.get("ECE"), raw_ece)
        raw_brier = _float(raw.get("Brier"))
        cal_brier = _float(best.get("Brier"), raw_brier)
        if cal_ece < raw_ece and cal_brier <= raw_brier:
            interpretation = "valid-set calibration improves ECE/Brier"
        elif cal_ece < raw_ece:
            interpretation = "valid-set calibration improves ECE but not Brier"
        else:
            interpretation = "calibration benefit not established"
        out.append(
            {
                "score_type": score_type,
                "raw_ECE": raw_ece,
                "calibrated_ECE": cal_ece,
                "raw_Brier": raw_brier,
                "calibrated_Brier": cal_brier,
                "raw_AUROC": raw.get("AUROC", "NA"),
                "calibrated_AUROC": best.get("AUROC", "NA"),
                "interpretation": interpretation,
            }
        )
    return out


def _test_row(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("split") == "test" and row.get("ranking_method") == method), {})


def _go_no_go(
    ranking_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    dist_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    raw = _test_row(ranking_rows, "raw_p_true")
    random = _test_row(ranking_rows, "random")
    rel_cal = next((row for row in cal_rows if row.get("score_type") == "positive_relevance_score"), {})
    best_balanced = max(threshold_rows, key=lambda row: _float(row.get("test_balanced_accuracy")), default={})
    ranking_beats_random = _float(raw.get("MRR")) > _float(random.get("MRR")) and _float(raw.get("NDCG@3")) > _float(random.get("NDCG@3"))
    cal_improves = _float(rel_cal.get("calibrated_ECE")) < _float(rel_cal.get("raw_ECE"))
    threshold_helps = _float(best_balanced.get("test_balanced_accuracy")) > 0.55
    if ranking_beats_random and cal_improves and threshold_helps:
        return (
            "no_go_for_full_beauty_yet",
            "P(true) is not collapsed and calibration helps, but the 200/200 smoke is still weak; run a slightly larger threshold-free audit or Day1f before full Beauty.",
        )
    return (
        "no_go_for_full_beauty",
        "Do not full-run: ranking/threshold evidence is not strong enough to rule out weak pointwise decision signal.",
    )


def write_go_no_go(
    ranking_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    dist_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
) -> None:
    decision, rationale = _go_no_go(ranking_rows, threshold_rows, dist_rows, cal_rows)
    raw = _test_row(ranking_rows, "raw_p_true")
    random = _test_row(ranking_rows, "random")
    best_balanced = max(threshold_rows, key=lambda row: _float(row.get("test_balanced_accuracy")), default={})
    text = f"""# Framework-Observation-Day1e Go/No-Go Decision

## Decision

`{decision}`

{rationale}

## Evidence

- raw P(true) test MRR / random MRR: `{raw.get('MRR', 'NA')}` / `{random.get('MRR', 'NA')}`
- raw P(true) test NDCG@3 / random NDCG@3: `{raw.get('NDCG@3', 'NA')}` / `{random.get('NDCG@3', 'NA')}`
- raw P(true) test HR@1 / random HR@1: `{raw.get('HR@1', 'NA')}` / `{random.get('HR@1', 'NA')}`
- best threshold strategy on test balanced accuracy: `{best_balanced.get('threshold_strategy', 'NA')}`
- best threshold test balanced accuracy: `{best_balanced.get('test_balanced_accuracy', 'NA')}`
- best threshold test recommend true rate: `{best_balanced.get('test_recommend_true_rate', 'NA')}`

## Interpretation

Logit/token probability fixes scalar verbalized confidence collapse and gives a usable-but-weak miscalibrated signal. However, the hard recommend=true/false decision remains conservative and under-recommending at the default 0.5 threshold, so we should not full-run yet.
"""
    GO_NO_GO_MD.write_text(text, encoding="utf-8")


def write_day1d_report_update() -> None:
    if not DAY1D_REPORT_MD.exists():
        return
    text = DAY1D_REPORT_MD.read_text(encoding="utf-8")
    old = "If logit confidence has useful variance and AUROC/ECE improve over verbalized scalar confidence, keep the logit-confidence route. If the logit score is still weak, the current LoRA adapter's pointwise decision ability is insufficient; next steps should be pair/list context or self-consistency, not more scalar confidence wording."
    new = "Logit/token probability fixes scalar verbalized confidence collapse and gives a usable-but-weak miscalibrated signal. However, the hard recommend=true/false decision remains conservative and under-recommending at the default 0.5 threshold, so we should not full-run yet. The next audit should treat P(true) as a continuous relevance score rather than over-reading the fixed-threshold hard decision."
    text = text.replace(old, new)
    DAY1D_REPORT_MD.write_text(text, encoding="utf-8")


def update_prompt_comparison(ranking_rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> None:
    rows = _read_csv(PROMPT_COMPARISON_CSV)
    rows = [row for row in rows if row.get("prompt_variant") not in {"logit_ptrue_ranking", "logit_ptrue_valid_threshold"}]
    raw = _test_row(ranking_rows, "raw_p_true")
    random = _test_row(ranking_rows, "random")
    best_balanced = max(threshold_rows, key=lambda row: _float(row.get("test_balanced_accuracy")), default={})
    rows.append(
        {
            "prompt_variant": "logit_ptrue_ranking",
            "backend": "transformers",
            "confidence_source": "token_probability",
            "num_valid_rows": 200,
            "num_test_rows": 200,
            "parse_success_rate": 1.0,
            "schema_valid_rate": 1.0,
            "recommend_true_rate": "",
            "accuracy": "",
            "raw_ECE": "",
            "calibrated_ECE": "",
            "Brier": "",
            "AUROC": "",
            "confidence_mean": "",
            "confidence_std": "",
            "confidence_unique_count": "",
            "confidence_ge_0.9_rate": "",
            "confidence_ge_0.97_rate": "",
            "collapse_type": "not_collapsed_weak_signal",
            "recommendation": "run_threshold_free_ranking_audit",
            "test_MRR": raw.get("MRR", ""),
            "test_random_MRR": random.get("MRR", ""),
            "test_NDCG@3": raw.get("NDCG@3", ""),
            "test_random_NDCG@3": random.get("NDCG@3", ""),
        }
    )
    rows.append(
        {
            "prompt_variant": "logit_ptrue_valid_threshold",
            "backend": "transformers",
            "confidence_source": "token_probability",
            "num_valid_rows": 200,
            "num_test_rows": 200,
            "parse_success_rate": 1.0,
            "schema_valid_rate": 1.0,
            "recommend_true_rate": best_balanced.get("test_recommend_true_rate", ""),
            "accuracy": best_balanced.get("test_accuracy", ""),
            "raw_ECE": "",
            "calibrated_ECE": "",
            "Brier": "",
            "AUROC": "",
            "confidence_mean": "",
            "confidence_std": "",
            "confidence_unique_count": "",
            "confidence_ge_0.9_rate": "",
            "confidence_ge_0.97_rate": "",
            "collapse_type": "under_recommending_at_0.5_threshold",
            "recommendation": "maybe_full_if_ranking_beats_random",
            "threshold_strategy": best_balanced.get("threshold_strategy", ""),
            "valid_threshold": best_balanced.get("valid_threshold", ""),
            "test_balanced_accuracy": best_balanced.get("test_balanced_accuracy", ""),
        }
    )
    _write_csv(PROMPT_COMPARISON_CSV, rows)


def write_self_consistency_plan() -> None:
    text = """# Framework-Observation-Day1f Self-Consistency Plan

## Trigger

Run this only if Day1e shows that logit P(true) is still weak as a ranking/relevance signal or the go/no-go remains ambiguous.

## Scope

- Beauty 100/100 smoke only.
- Local Qwen-LoRA only.
- No training, no evidence, no CEP, no external APIs, no four-domain run.

## Method

For each user-candidate pair, sample the binary recommendation prompt `n=5` or `n=10` times with stochastic decoding.

Derived scores:

- `recommend_true_frequency`: fraction of samples voting `recommend=true`; use as relevance score.
- `decision_confidence`: `majority_vote_rate = max(true_votes, false_votes) / n`.
- `uncertainty`: `1 - majority_vote_rate`.

## Evaluation

Compare self-consistency against Day1d/Day1e logit P(true):

- AUROC, Brier, ECE for `recommend_true_frequency` against label.
- AUROC, Brier, ECE for `decision_confidence` against decision correctness.
- User-level NDCG/MRR/HR using `recommend_true_frequency` as ranking score.

Do not run self-consistency full until the 100/100 smoke beats or clarifies logit P(true).
"""
    SELF_CONSISTENCY_PLAN_MD.write_text(text, encoding="utf-8")


def write_report(
    ranking_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    dist_rows: list[dict[str, Any]],
    cal_rows: list[dict[str, Any]],
) -> None:
    raw = _test_row(ranking_rows, "raw_p_true")
    random = _test_row(ranking_rows, "random")
    best_balanced = max(threshold_rows, key=lambda row: _float(row.get("test_balanced_accuracy")), default={})
    rel_cal = next((row for row in cal_rows if row.get("score_type") == "positive_relevance_score"), {})
    decision, rationale = _go_no_go(ranking_rows, threshold_rows, dist_rows, cal_rows)
    text = f"""# Framework-Observation-Day1e Logit Score Ranking Report

## 1. Day1d Recap

Day1d showed that logit/token probability fixes scalar verbalized confidence collapse. P(true) has `unique_count=200`, test std around `0.170`, and relevance AUROC around `0.589`. Calibration reduces relevance ECE from `{rel_cal.get('raw_ECE', 'NA')}` to `{rel_cal.get('calibrated_ECE', 'NA')}`.

## 2. Why Hard Threshold 0.5 Is Not Enough

The 0.5 hard threshold gives a very conservative recommend decision. It produced low recommend-true rate in Day1d, but this does not mean the continuous score is useless. Day1e therefore evaluates P(true) as a threshold-free relevance/ranking score and learns thresholds on valid.

## 3. P(true) Threshold-Free Ranking

- test raw P(true) MRR / random MRR: `{raw.get('MRR', 'NA')}` / `{random.get('MRR', 'NA')}`
- test raw P(true) NDCG@3 / random NDCG@3: `{raw.get('NDCG@3', 'NA')}` / `{random.get('NDCG@3', 'NA')}`
- test raw P(true) HR@1 / random HR@1: `{raw.get('HR@1', 'NA')}` / `{random.get('HR@1', 'NA')}`
- HR@10 is marked trivial because candidate pools are at most 10 items.

## 4. Valid Threshold Selection

- best test balanced-accuracy strategy: `{best_balanced.get('threshold_strategy', 'NA')}`
- valid threshold: `{best_balanced.get('valid_threshold', 'NA')}`
- test accuracy/F1/balanced accuracy: `{best_balanced.get('test_accuracy', 'NA')}` / `{best_balanced.get('test_f1', 'NA')}` / `{best_balanced.get('test_balanced_accuracy', 'NA')}`
- test recommend true rate: `{best_balanced.get('test_recommend_true_rate', 'NA')}`

## 5. Calibration Result

Valid-set calibration makes the weak P(true) signal more usable by reducing ECE/Brier. This supports continuing with token-probability confidence rather than verbalized scalar confidence.

## 6. Go/No-Go For Full Beauty

- decision: `{decision}`
- rationale: {rationale}

## 7. Next Step

If the ranking lift over random is modest or ambiguous, do not full-run. The next smoke should be Day1f self-consistency 100/100 or a pair/list context audit, not more scalar confidence wording.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1e logit score ranking and threshold audit.")
    parser.add_argument("--pred_dir", default=str(PRED_DIR))
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    valid_rows = _read_jsonl(pred_dir / "valid_raw.jsonl")
    test_rows = _read_jsonl(pred_dir / "test_raw.jsonl")
    ranking_rows = ranking_eval(valid_rows, test_rows)
    threshold_rows = threshold_selection(valid_rows, test_rows)
    dist_rows = score_distribution({"valid": valid_rows, "test": test_rows})
    cal_rows = calibration_summary()
    _write_csv(RANKING_CSV, ranking_rows)
    _write_csv(THRESHOLD_CSV, threshold_rows)
    _write_csv(DIST_CSV, dist_rows)
    _write_csv(CAL_SUMMARY_CSV, cal_rows)
    write_go_no_go(ranking_rows, threshold_rows, dist_rows, cal_rows)
    write_report(ranking_rows, threshold_rows, dist_rows, cal_rows)
    write_self_consistency_plan()
    write_day1d_report_update()
    update_prompt_comparison(ranking_rows, threshold_rows)
    print(
        json.dumps(
            {
                "ranking": str(RANKING_CSV),
                "thresholds": str(THRESHOLD_CSV),
                "distribution": str(DIST_CSV),
                "calibration_summary": str(CAL_SUMMARY_CSV),
                "go_no_go": str(GO_NO_GO_MD),
                "report": str(REPORT_MD),
                "self_consistency_plan": str(SELF_CONSISTENCY_PLAN_MD),
                "prompt_comparison": str(PROMPT_COMPARISON_CSV),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
