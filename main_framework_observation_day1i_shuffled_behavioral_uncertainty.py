from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import _read_jsonl
from main_framework_observation_day1f_self_consistency import (
    _calibrate,
    _group_by_user,
    _label,
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
from main_framework_observation_day1h_listwise_behavioral_uncertainty import (
    _behavior_rows,
    _candidate_score_rows,
    _prompt,
    _parse_ranking,
    run_inference as _run_day1h_inference,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_day1i_shuffled_behavioral_uncertainty.yaml"
SHUFFLE_JSON = Path("data_done/framework_observation_day1i_shuffled_candidate_order_subset.json")
ORDER_DIAG_CSV = Path("data_done/framework_observation_day1i_candidate_order_diagnostics.csv")
RANK_CSV = Path("data_done/framework_observation_day1i_shuffled_behavioral_ranking_eval.csv")
CAL_CSV = Path("data_done/framework_observation_day1i_shuffled_behavioral_uncertainty_calibration.csv")
DIAG_CSV = Path("data_done/framework_observation_day1i_shuffled_behavioral_diagnostics.csv")
COMPARISON_CSV = Path("data_done/framework_observation_day1i_order_bias_control_comparison.csv")
REPORT_MD = Path("data_done/framework_observation_day1i_order_shuffled_behavioral_uncertainty_report.md")
GO_NO_GO_MD = Path("data_done/framework_observation_day1i_go_no_go_decision.md")


def _split_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    subset = load_subset(cfg)
    user_ids = _subset_entries(subset)[split]["sampled_user_ids"]
    return _selected_rows(Path(str(cfg[f"{split}_file"])), user_ids)


def _shuffle_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(cfg.get("shuffle_subset_path", SHUFFLE_JSON)))
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("entries", []))
    seed = int(cfg.get("seed", 42))
    entries = []
    for split in ["valid", "test"]:
        rows = _split_rows(cfg, split)
        grouped = _group_by_user(rows)
        for user_idx, user_id in enumerate(sorted(grouped)):
            user_rows = grouped[user_id]
            original = [str(row.get("candidate_item_id", "")) for row in user_rows]
            labels = {str(row.get("candidate_item_id", "")): int(row.get("label", 0)) for row in user_rows}
            target = next((item_id for item_id, label in labels.items() if label == 1), "")
            shuffled = original[:]
            rng = random.Random(seed + user_idx + (0 if split == "valid" else 100000))
            rng.shuffle(shuffled)
            entries.append(
                {
                    "split": split,
                    "user_id": user_id,
                    "original_candidate_order": original,
                    "shuffled_candidate_order": shuffled,
                    "target_item_id": target,
                    "positive_original_position": original.index(target) + 1 if target in original else 0,
                    "positive_shuffled_position": shuffled.index(target) + 1 if target in shuffled else 0,
                    "seed": seed,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries


def create_shuffle_subset(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _shuffle_entries(cfg)
    rows = []
    for split in ["valid", "test"]:
        split_entries = [entry for entry in entries if entry["split"] == split]
        counts = Counter(int(entry["positive_shuffled_position"]) for entry in split_entries)
        total = len(split_entries)
        for position in range(1, 7):
            rows.append(
                {
                    "split": split,
                    "position": position,
                    "positive_count": counts.get(position, 0),
                    "positive_rate": counts.get(position, 0) / total if total else 0.0,
                }
            )
    _write_csv(ORDER_DIAG_CSV, rows)
    return entries


def _shuffled_rows(cfg: dict[str, Any], split: str) -> list[list[dict[str, Any]]]:
    entries = {entry["user_id"]: entry for entry in _shuffle_entries(cfg) if entry["split"] == split}
    grouped = _group_by_user(_split_rows(cfg, split))
    out = []
    for user_id in sorted(grouped):
        by_item = {str(row.get("candidate_item_id", "")): row for row in grouped[user_id]}
        order = entries[user_id]["shuffled_candidate_order"]
        out.append([by_item[item_id] for item_id in order])
    return out


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in _read_jsonl(path)}


def run_inference(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore
    from main_framework_observation_day1f_self_consistency import _load_vllm, _lora_request

    create_shuffle_subset(cfg)
    user_groups = _shuffled_rows(cfg, split)
    output_path = Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()

    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    batch_size = int(cfg.get("vllm_batch_size", 12))
    n = int(cfg.get("num_samples", 5))
    seed = int(cfg.get("seed", 42))
    pending = []
    for user_rows in user_groups:
        sid = f"{split}_{user_rows[0].get('user_id', '')}"
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
                    "context_type": "listwise_behavioral_shuffled",
                    "candidate_order": "shuffled",
                    "tie_break_policy": str(cfg.get("tie_break_policy", "order_neutral_expected_tie_metric")),
                    "inference_backend": "vllm",
                }
            )
        _write_jsonl(output_path, out_rows)
    return output_path


def _diag_and_candidates(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    behavior_by_split = {split: _behavior_rows(rows) for split, rows in rows_by_split.items()}
    candidate_by_split = {split: _candidate_score_rows(rows) for split, rows in behavior_by_split.items()}
    diag = []
    for split, behavior in behavior_by_split.items():
        candidates = candidate_by_split[split]
        top_correct = [int(row.get("majority_top1_correct", 0)) for row in behavior]
        top_conf = [_score(row, "majority_top1_confidence") for row in behavior]
        entropy = [_score(row, "top1_vote_entropy") for row in behavior]
        rank_variance = [_score(row, "rank_variance") for row in behavior]
        diag.append(
            {
                "split": split,
                "candidate_order": "shuffled",
                "tie_break_policy": "order_neutral_expected_tie_metric",
                "num_users": len(behavior),
                "parse_success_rate": mean([_score(row, "parse_success_rate") for row in behavior]) if behavior else 0.0,
                "schema_valid_rate": mean([_score(row, "schema_valid_rate") for row in behavior]) if behavior else 0.0,
                **ranking_metrics(candidates, "rank_score"),
                "majority_top1_confidence_mean": mean(top_conf) if top_conf else 0.0,
                "majority_top1_confidence_AUROC": auroc(top_conf, top_correct),
                "majority_top1_confidence_ECE": ece(top_conf, top_correct),
                "majority_top1_confidence_Brier": brier(top_conf, top_correct),
                "top1_vote_entropy_mean": mean(entropy) if entropy else 0.0,
                "top1_vote_entropy_AUROC_for_error_risk": auroc(entropy, [1 - y for y in top_correct]),
                "rank_variance_mean": mean(rank_variance) if rank_variance else 0.0,
                "rank_variance_inverse_AUROC_for_correctness": auroc([1.0 - min(v, 1.0) for v in rank_variance], top_correct),
                "positive_rank_mean": mean([_score(row, "positive_rank_mean") for row in behavior]) if behavior else 0.0,
                "positive_rank_std_mean": mean([_score(row, "positive_rank_std") for row in behavior]) if behavior else 0.0,
            }
        )
    return diag, behavior_by_split, candidate_by_split


def _calibration_rows(behavior_by_split: dict[str, list[dict[str, Any]]], candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    specs = [
        ("rank_score_to_relevance_label", candidate_by_split["valid"], candidate_by_split["test"], "rank_score", lambda row: _label(row)),
        ("stability_weighted_rank_score_to_relevance_label", candidate_by_split["valid"], candidate_by_split["test"], "stability_weighted_rank_score", lambda row: _label(row)),
        ("majority_top1_confidence_to_top1_correctness", behavior_by_split["valid"], behavior_by_split["test"], "majority_top1_confidence", lambda row: int(row.get("majority_top1_correct", 0))),
        ("top1_vote_entropy_to_error_risk", behavior_by_split["valid"], behavior_by_split["test"], "top1_vote_entropy", lambda row: 1 - int(row.get("majority_top1_correct", 0))),
        ("rank_variance_to_error_risk", behavior_by_split["valid"], behavior_by_split["test"], "rank_variance", lambda row: 1 - int(row.get("majority_top1_correct", 0))),
    ]
    for target, valid_rows, test_rows, key, label_fn in specs:
        valid_scores = [_score(row, key) for row in valid_rows]
        valid_labels = [label_fn(row) for row in valid_rows]
        test_scores = [_score(row, key) for row in test_rows]
        test_labels = [label_fn(row) for row in test_rows]
        cal = _calibrate(valid_scores, valid_labels, test_scores)
        raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
        calibrated = {"ECE": ece(cal, test_labels), "Brier": brier(cal, test_labels), "AUROC": auroc(cal, test_labels)}
        out.append({"target": target, "score_type": "raw", **raw, "calibration_method": "none"})
        out.append({"target": target, "score_type": "logistic_calibrated", **calibrated, "calibration_method": "valid_logistic", "delta_ECE_vs_raw": calibrated["ECE"] - raw["ECE"], "delta_Brier_vs_raw": calibrated["Brier"] - raw["Brier"]})
    return out


def _ranking_rows(candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in candidate_by_split.items():
        random_metrics = random_ranking_metrics(rows)
        for method in ["rank_score", "stability_weighted_rank_score"]:
            out.append({"split": split, "ranking_method": method, "candidate_order": "shuffled", "tie_break_policy": "order_neutral_expected_tie_metric", **ranking_metrics(rows, method), "random_baseline": json.dumps(random_metrics, sort_keys=True), "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "random", "candidate_order": "shuffled", **random_metrics, "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "oracle", "candidate_order": "oracle", "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "oracle_upper_bound": 1.0})
    return out


def _day1h_candidate_rows(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows_by_split = {split: _read_pred(Path(str(cfg["day1h_output_dir"])) / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    return {split: _candidate_score_rows(_behavior_rows(rows)) for split, rows in rows_by_split.items()}


def _pointwise_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return _read_pred(Path(str(cfg["pointwise_logit_output_dir"])) / f"{split}_raw.jsonl")


def _day1g_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = _read_pred(Path(str(cfg["day1g_listwise_output_dir"])) / f"{split}_raw.jsonl")
    out = []
    for row in rows:
        labels = dict(row.get("labels_by_item", {}))
        candidate_ids = [str(x) for x in row.get("candidate_item_ids", [])]
        ranked = [str(x) for x in row.get("ranked_item_ids", [])]
        scores = {item_id: len(candidate_ids) - idx for idx, item_id in enumerate(ranked)} if row.get("schema_valid") else {}
        for item_id in candidate_ids:
            out.append({"user_id": row.get("user_id", ""), "candidate_item_id": item_id, "label": int(labels.get(item_id, 0)), "listwise_rank_score": float(scores.get(item_id, 0.0))})
    return out


def _metric_row(method: str, order: str, rows: list[dict[str, Any]], score_key: str, day1h_ref: dict[str, Any] | None = None) -> dict[str, Any]:
    scores = [_score(row, score_key) for row in rows]
    labels = [_label(row) for row in rows]
    rank = ranking_metrics(rows, score_key)
    drop = ""
    interpretation = ""
    if day1h_ref:
        drop = float(day1h_ref.get("MRR", 0.0)) - float(rank.get("MRR", 0.0))
        if drop > 0.2:
            interpretation = "order_bias_confirmed"
        elif drop > 0.05:
            interpretation = "order_bias_reduced_but_signal_remains"
        else:
            interpretation = "robust_behavioral_uncertainty_signal"
    return {"method": method, "candidate_order": order, **rank, "AUROC": auroc(scores, labels), "top1_confidence_AUROC": "", "entropy_error_AUROC": "", "rank_variance_AUROC": "", "drop_vs_day1h": drop, "interpretation": interpretation}


def _comparison_rows(cfg: dict[str, Any], diag: list[dict[str, Any]], candidate_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    split = "test"
    day1h_candidates = _day1h_candidate_rows(cfg)
    day1h_rank = ranking_metrics(day1h_candidates[split], "rank_score") if day1h_candidates.get(split) else {}
    day1h_stability = ranking_metrics(day1h_candidates[split], "stability_weighted_rank_score") if day1h_candidates.get(split) else {}
    shuffled_diag = next((row for row in diag if row.get("split") == split), {})
    rows = []
    if day1h_candidates.get(split):
        rows.append(_metric_row("Day1h original-order behavioral rank_score", "original_positive_position_1", day1h_candidates[split], "rank_score"))
        rows.append(_metric_row("Day1h original-order stability_weighted_rank_score", "original_positive_position_1", day1h_candidates[split], "stability_weighted_rank_score"))
    rows.append(_metric_row("Day1i shuffled-order behavioral rank_score", "shuffled", candidate_by_split[split], "rank_score", day1h_rank))
    rows.append(_metric_row("Day1i shuffled-order stability_weighted_rank_score", "shuffled", candidate_by_split[split], "stability_weighted_rank_score", day1h_stability))
    if _pointwise_rows(cfg, split):
        rows.append(_metric_row("pointwise logit P(true)", "pointwise", _pointwise_rows(cfg, split), "positive_relevance_score"))
    day1g = _day1g_rows(cfg, split)
    if day1g:
        rows.append(_metric_row("Day1g single listwise", "original_positive_position_1", day1g, "listwise_rank_score"))
    random_metrics = random_ranking_metrics(candidate_by_split[split])
    rows.append({"method": "random", "candidate_order": "baseline", **random_metrics, "AUROC": "NA", "top1_confidence_AUROC": "", "entropy_error_AUROC": "", "rank_variance_AUROC": "", "drop_vs_day1h": "", "interpretation": "random_baseline"})
    rows.append({"method": "oracle", "candidate_order": "oracle", "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "AUROC": 1.0, "top1_confidence_AUROC": "", "entropy_error_AUROC": "", "rank_variance_AUROC": "", "drop_vs_day1h": "", "interpretation": "oracle_upper_bound"})
    for row in rows:
        if row["method"].startswith("Day1i"):
            row["top1_confidence_AUROC"] = shuffled_diag.get("majority_top1_confidence_AUROC", "")
            row["entropy_error_AUROC"] = shuffled_diag.get("top1_vote_entropy_AUROC_for_error_risk", "")
            row["rank_variance_AUROC"] = shuffled_diag.get("rank_variance_inverse_AUROC_for_correctness", "")
    return rows


def _write_report(cfg: dict[str, Any], diag: list[dict[str, Any]], comp: list[dict[str, Any]]) -> str:
    test = next((row for row in diag if row.get("split") == "test"), {})
    shuffled = next((row for row in comp if row.get("method") == "Day1i shuffled-order behavioral rank_score"), {})
    stability = next((row for row in comp if row.get("method") == "Day1i shuffled-order stability_weighted_rank_score"), {})
    interpretation = stability.get("interpretation") or shuffled.get("interpretation") or "pending"
    text = f"""# Framework-Observation-Day1i Order-Shuffled Behavioral Uncertainty Report

## Why Day1i

Day1h was strong but confounded by positive-at-position-1 candidate order bias. Day1h cannot be used as clean evidence until this shuffle control is complete.

## Concept Boundary

Behavioral confidence / uncertainty is still confidence observation, but it is implicit rather than verbalized. It must be disentangled from candidate position bias before making stronger claims.

## Shuffle Setup

Day1i uses the same Beauty 100-user valid/test subset and the same six-candidate pools as Day1h. Candidate order is shuffled once per user with seed `{cfg.get('seed', 42)}`. Evaluation uses `{cfg.get('tie_break_policy', 'order_neutral_expected_tie_metric')}` and does not use input order for tie-breaking.

## Shuffled Ranking Result

- shuffled rank_score test MRR/HR@1/NDCG@3/AUROC: `{shuffled.get('MRR', 'NA')}` / `{shuffled.get('HR@1', 'NA')}` / `{shuffled.get('NDCG@3', 'NA')}` / `{shuffled.get('AUROC', 'NA')}`
- shuffled stability-weighted test MRR/HR@1/NDCG@3/AUROC: `{stability.get('MRR', 'NA')}` / `{stability.get('HR@1', 'NA')}` / `{stability.get('NDCG@3', 'NA')}` / `{stability.get('AUROC', 'NA')}`

## Shuffled Behavioral Uncertainty

- majority_top1_confidence AUROC for top1 correctness: `{test.get('majority_top1_confidence_AUROC', 'NA')}`
- top1_vote_entropy AUROC for error risk: `{test.get('top1_vote_entropy_AUROC_for_error_risk', 'NA')}`
- rank_variance inverse AUROC for correctness: `{test.get('rank_variance_inverse_AUROC_for_correctness', 'NA')}`

## Conclusion

`{interpretation}`

If performance remains high, behavioral uncertainty is promising. If it drops near random, Day1h was mostly order bias. If ranking remains useful but uncertainty is weak, use listwise ranking score, not confidence.
"""
    REPORT_MD.write_text(text, encoding="utf-8")
    GO_NO_GO_MD.write_text(f"# Framework-Observation-Day1i Go/No-Go Decision\n\n`{interpretation}`\n\nThis remains observation only. Do not enter CEP/evidence from Day1i alone.\n", encoding="utf-8")
    return interpretation


def analyze(cfg: dict[str, Any]) -> None:
    create_shuffle_subset(cfg)
    rows_by_split = {split: _read_pred(Path(str(cfg["output_dir"])) / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    if not all(rows_by_split.values()):
        raise FileNotFoundError("Day1i shuffled valid/test predictions are required for analysis.")
    diag, behavior_by_split, candidate_by_split = _diag_and_candidates(rows_by_split)
    cal = _calibration_rows(behavior_by_split, candidate_by_split)
    rank = _ranking_rows(candidate_by_split)
    comp = _comparison_rows(cfg, diag, candidate_by_split)
    _write_csv(DIAG_CSV, diag)
    _write_csv(CAL_CSV, cal)
    _write_csv(RANK_CSV, rank)
    _write_csv(COMPARISON_CSV, comp)
    interpretation = _write_report(cfg, diag, comp)
    print(json.dumps({"order_diagnostics": str(ORDER_DIAG_CSV), "diagnostics": str(DIAG_CSV), "calibration": str(CAL_CSV), "ranking": str(RANK_CSV), "comparison": str(COMPARISON_CSV), "report": str(REPORT_MD), "go_no_go": str(GO_NO_GO_MD), "interpretation": interpretation}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1i candidate-order shuffled behavioral uncertainty control.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--create_shuffle_subset", action="store_true")
    parser.add_argument("--run_inference", choices=["valid", "test"], default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _read_config(args.config)
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    if args.create_shuffle_subset:
        entries = create_shuffle_subset(cfg)
        print(json.dumps({"shuffle_subset_path": str(cfg.get("shuffle_subset_path", SHUFFLE_JSON)), "num_entries": len(entries), "order_diagnostics": str(ORDER_DIAG_CSV)}, ensure_ascii=False, indent=2))
    if args.run_inference:
        path = run_inference(cfg, args.run_inference, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_inference, "mode": "shuffled_listwise_behavioral_uncertainty"}, ensure_ascii=False, indent=2))
    if args.analyze_only:
        analyze(cfg)


if __name__ == "__main__":
    main()
