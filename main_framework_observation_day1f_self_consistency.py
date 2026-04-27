from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from main_framework_day4_train_qwen_lora_baseline import _read_config
from main_framework_observation_day1_local_confidence_infer import (
    _compact_json,
    _read_jsonl,
    _truncate_text,
)
from main_framework_observation_day1d_logit_confidence import (
    _load_model as _load_transformers_model,
    _true_false_scores_batch,
)


DEFAULT_CONFIG = "configs/framework_observation/beauty_qwen_lora_self_consistency.yaml"
SUBSET_JSON = Path("data_done/framework_observation_day1f_self_consistency_subset.json")
DIAG_CSV = Path("data_done/framework_observation_day1f_self_consistency_diagnostics.csv")
CAL_CSV = Path("data_done/framework_observation_day1f_self_consistency_calibration.csv")
RANKING_CSV = Path("data_done/framework_observation_day1f_self_consistency_ranking_eval.csv")
REPORT_MD = Path("data_done/framework_observation_day1f_self_consistency_report.md")
COMPARISON_CSV = Path("data_done/framework_observation_day1f_logit_vs_self_consistency_comparison.csv")
GO_NO_GO_MD = Path("data_done/framework_observation_day1f_go_no_go_decision.md")
DAY1G_PLAN_MD = Path("data_done/framework_observation_day1g_pair_list_context_plan.md")


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


def _read_pred(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl(path)


def _sample_id(split: str, row: dict[str, Any]) -> str:
    return f"{split}_{row.get('user_id', '')}_{row.get('candidate_item_id', '')}"


def _group_by_user(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("user_id", ""))].append(row)
    return dict(grouped)


def _selected_rows(split_file: Path, user_ids: list[str]) -> list[dict[str, Any]]:
    selected = set(user_ids)
    return [row for row in _read_jsonl(split_file) if str(row.get("user_id", "")) in selected]


def _subset_entries(subset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["split"]: entry for entry in subset.get("subsets", [])}


def create_subset(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    num_users = int(cfg.get("num_users", 100))
    rng = random.Random(seed)
    entries = []
    for split in ["valid", "test"]:
        rows = _read_jsonl(Path(str(cfg[f"{split}_file"])))
        grouped = _group_by_user(rows)
        complete_user_ids = sorted([user_id for user_id, items in grouped.items() if len(items) == 6])
        if len(complete_user_ids) < num_users:
            raise ValueError(f"{split} has only {len(complete_user_ids)} complete users; need {num_users}.")
        sampled = sorted(rng.sample(complete_user_ids, num_users))
        pool_sizes = [len(grouped[user_id]) for user_id in sampled]
        entries.append(
            {
                "split": split,
                "num_users": len(sampled),
                "num_rows": sum(pool_sizes),
                "candidate_pool_size_mean": mean(pool_sizes),
                "sampled_user_ids": sampled,
                "seed": seed,
            }
        )
    subset = {"seed": seed, "subsets": entries}
    path = Path(str(cfg.get("subset_path", SUBSET_JSON)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return subset


def load_subset(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(cfg.get("subset_path", SUBSET_JSON)))
    if not path.exists():
        return create_subset(cfg)
    return json.loads(path.read_text(encoding="utf-8"))


def format_decision_prompt(
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


def _parse_recommend(raw_text: str) -> bool | None:
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


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in _read_jsonl(path)}


def _load_vllm(cfg: dict[str, Any], model_variant: str):
    from vllm import LLM  # type: ignore

    model_path = str(cfg["model_name_or_path"])
    tokenizer_path = str(cfg.get("tokenizer_name_or_path") or model_path)
    enable_lora = model_variant == "lora" and bool(cfg.get("vllm_enable_lora", True))
    kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "dtype": "bfloat16" if str(cfg.get("bf16", "true")).lower() == "true" else "float16",
        "tensor_parallel_size": int(cfg.get("vllm_tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(cfg.get("vllm_gpu_memory_utilization", 0.85)),
        "max_model_len": int(cfg.get("vllm_max_model_len", cfg.get("max_seq_len", 4096))),
        "enable_lora": enable_lora,
    }
    if enable_lora:
        kwargs["max_loras"] = int(cfg.get("vllm_max_loras", 1))
        kwargs["max_lora_rank"] = int(cfg.get("vllm_max_lora_rank", 8))
    return LLM(**kwargs)


def _lora_request(cfg: dict[str, Any], model_variant: str):
    if model_variant != "lora":
        return None
    from vllm.lora.request import LoRARequest  # type: ignore

    adapter_path = str(cfg.get("adapter_path") or "")
    if not adapter_path or not Path(adapter_path).exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")
    return LoRARequest("qwen_lora_self_consistency", 1, adapter_path)


def run_self_consistency(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    from vllm import SamplingParams  # type: ignore

    subset = load_subset(cfg)
    entries = _subset_entries(subset)
    user_ids = entries[split]["sampled_user_ids"]
    rows = _selected_rows(Path(str(cfg[f"{split}_file"])), user_ids)
    output_dir = Path(str(cfg["output_dir"]))
    output_path = output_dir / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    llm = _load_vllm(cfg, model_variant)
    lora_request = _lora_request(cfg, model_variant)
    n = int(cfg.get("self_consistency_samples", 5))
    batch_size = int(cfg.get("vllm_batch_size", 16))
    seed = int(cfg.get("seed", 42))
    max_history_items = int(cfg.get("max_history_items", 8))
    max_title_chars = int(cfg.get("max_title_chars", 160))
    max_history_text_chars = int(cfg.get("max_history_text_chars", 260))
    max_candidate_text_chars = int(cfg.get("max_candidate_text_chars", 360))
    prompt_rows = []
    for row in rows:
        sid = _sample_id(split, row)
        if sid in finished:
            continue
        prompt_rows.append(
            (
                row,
                sid,
                format_decision_prompt(
                    row,
                    cfg["prompt_template"],
                    max_history_items,
                    max_title_chars,
                    max_history_text_chars,
                    max_candidate_text_chars,
                ),
            )
        )

    for start in range(0, len(prompt_rows), batch_size):
        batch = prompt_rows[start : start + batch_size]
        prompts = [item[2] for item in batch]
        raw_generations: list[list[str]] = [[] for _ in batch]
        parsed_votes: list[list[bool | None]] = [[] for _ in batch]
        for sample_idx in range(n):
            sampling_params = SamplingParams(
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.9)),
                max_tokens=int(cfg.get("max_new_tokens", 64)),
                seed=seed + sample_idx,
            )
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            for i, output in enumerate(outputs):
                text = output.outputs[0].text if output.outputs else ""
                raw_generations[i].append(text)
                parsed_votes[i].append(_parse_recommend(text))
        out_rows = []
        for (row, sid, _), generations, votes in zip(batch, raw_generations, parsed_votes):
            valid_votes = [vote for vote in votes if vote is not None]
            true_count = sum(1 for vote in valid_votes if vote is True)
            false_count = sum(1 for vote in valid_votes if vote is False)
            denom = len(valid_votes)
            true_freq = true_count / denom if denom else 0.0
            majority = true_count > false_count if denom else False
            majority_rate = max(true_freq, 1.0 - true_freq) if denom else 0.0
            out_rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "user_id": row.get("user_id", ""),
                    "candidate_item_id": row.get("candidate_item_id", ""),
                    "label": int(row.get("label", 0)),
                    "num_samples": n,
                    "recommend_true_count": true_count,
                    "recommend_false_count": false_count,
                    "recommend_true_frequency": true_freq,
                    "majority_recommend": majority,
                    "majority_vote_rate": majority_rate,
                    "self_consistency_confidence": majority_rate,
                    "self_consistency_uncertainty": 1.0 - majority_rate,
                    "raw_generations": generations,
                    "parse_success_count": denom,
                    "parse_success_rate": denom / n if n else 0.0,
                    "model_variant": model_variant,
                    "adapter_path": str(cfg.get("adapter_path", "")) if model_variant == "lora" else "",
                    "inference_backend": "vllm",
                    "confidence_source": "self_consistency",
                }
            )
        _write_jsonl(output_path, out_rows)
    return output_path


def run_logit_subset(cfg: dict[str, Any], split: str, model_variant: str, resume: bool) -> Path:
    subset = load_subset(cfg)
    entries = _subset_entries(subset)
    rows = _selected_rows(Path(str(cfg[f"{split}_file"])), entries[split]["sampled_user_ids"])
    output_dir = Path(str(cfg["logit_subset_output_dir"]))
    output_path = output_dir / f"{split}_raw.jsonl"
    finished = _existing_ids(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    model, tokenizer = _load_transformers_model(cfg, model_variant)
    batch_size = int(cfg.get("logit_batch_size", 4))
    max_seq_len = int(cfg.get("max_seq_len", 4096))
    max_history_items = int(cfg.get("max_history_items", 8))
    max_title_chars = int(cfg.get("max_title_chars", 160))
    max_history_text_chars = int(cfg.get("max_history_text_chars", 260))
    max_candidate_text_chars = int(cfg.get("max_candidate_text_chars", 360))
    pending: list[tuple[dict[str, Any], str, str]] = []

    def flush(items: list[tuple[dict[str, Any], str, str]]) -> None:
        if not items:
            return
        prompts = [item[2] for item in items]
        scores = _true_false_scores_batch(model, tokenizer, prompts, max_seq_len)
        out_rows = []
        for (row, sid, _), score in zip(items, scores):
            recommend = bool(score["p_true"] >= score["p_false"])
            out_rows.append(
                {
                    "sample_id": sid,
                    "split": split,
                    "user_id": row.get("user_id", ""),
                    "candidate_item_id": row.get("candidate_item_id", ""),
                    "label": int(row.get("label", 0)),
                    "recommend": recommend,
                    "p_true": score["p_true"],
                    "p_false": score["p_false"],
                    "positive_relevance_score": score["positive_relevance_score"],
                    "decision_confidence": score["decision_confidence"],
                    "correctness": int(recommend == (int(row.get("label", 0)) == 1)),
                    "confidence_source": "token_probability",
                    "inference_backend": "transformers",
                }
            )
        _write_jsonl(output_path, out_rows)

    for row in rows:
        sid = _sample_id(split, row)
        if sid in finished:
            continue
        prompt = format_decision_prompt(
            row,
            cfg.get("logit_prompt_template", cfg["prompt_template"]),
            max_history_items,
            max_title_chars,
            max_history_text_chars,
            max_candidate_text_chars,
        )
        pending.append((row, sid, prompt))
        if len(pending) >= batch_size:
            flush(pending)
            pending = []
    flush(pending)
    return output_path


def _label(row: dict[str, Any]) -> int:
    return int(row.get("label", 0))


def _score(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0))


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, score in enumerate(scores) if score >= lo and (score < hi or b == bins - 1)]
        if idx:
            out += len(idx) / total * abs(mean([scores[i] for i in idx]) - mean([labels[i] for i in idx]))
    return out


def brier(scores: list[float], labels: list[int]) -> float:
    return mean([(s - y) ** 2 for s, y in zip(scores, labels)]) if scores else 0.0


def auroc(scores: list[float], labels: list[int]) -> float | str:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return "NA"
    wins = 0.0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _binary_metrics(scores: list[float], labels: list[int], threshold: float = 0.5) -> dict[str, Any]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "f1": f1,
        "balanced_accuracy": (recall + tnr) / 2,
        "recommend_true_rate": mean(preds) if preds else 0.0,
    }


def _fit_logistic(valid_scores: list[float], valid_labels: list[int]):
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(solver="lbfgs")
        model.fit([[s] for s in valid_scores], valid_labels)
        return model
    except Exception:
        return None


def _calibrate(valid_scores: list[float], valid_labels: list[int], target_scores: list[float]) -> list[float]:
    model = _fit_logistic(valid_scores, valid_labels)
    if model is None:
        return target_scores
    return [float(x[1]) for x in model.predict_proba([[s] for s in target_scores])]


def _dcg(labels: list[int], k: int) -> float:
    return sum((2**label - 1) / math.log2(rank + 2) for rank, label in enumerate(labels[:k]))


def _ndcg(labels: list[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    return _dcg(labels, k) / ideal_dcg if ideal_dcg > 0 else 0.0


def _mrr(labels: list[int]) -> float:
    for rank, label in enumerate(labels, 1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def _hr(labels: list[int], k: int) -> float:
    return 1.0 if any(label > 0 for label in labels[:k]) else 0.0


def ranking_metrics(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    grouped = _group_by_user(rows)
    vals = {key: [] for key in ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5", "HR@10"]}
    pool_sizes = []
    for items in grouped.values():
        ranked = sorted(items, key=lambda row: _score(row, score_key), reverse=True)
        labels = [_label(row) for row in ranked]
        pool_sizes.append(len(items))
        vals["NDCG@10"].append(_ndcg(labels, 10))
        vals["MRR"].append(_mrr(labels))
        vals["HR@1"].append(_hr(labels, 1))
        vals["HR@3"].append(_hr(labels, 3))
        vals["NDCG@3"].append(_ndcg(labels, 3))
        vals["NDCG@5"].append(_ndcg(labels, 5))
        vals["HR@10"].append(_hr(labels, 10))
    out = {k: mean(v) if v else 0.0 for k, v in vals.items()}
    out["candidate_pool_size_mean"] = mean(pool_sizes) if pool_sizes else 0.0
    out["hr10_trivial_flag"] = out["candidate_pool_size_mean"] <= 10
    return out


def random_ranking_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_user(rows)
    metrics = {"NDCG@10": [], "MRR": [], "HR@1": [], "HR@3": [], "NDCG@3": [], "NDCG@5": [], "HR@10": []}
    for items in grouped.values():
        n = len(items)
        positives = sum(_label(row) for row in items)
        if positives == 1:
            expected_values = {key: [] for key in metrics}
            for rank in range(1, n + 1):
                labels = [0] * n
                labels[rank - 1] = 1
                expected_values["NDCG@10"].append(_ndcg(labels, 10))
                expected_values["MRR"].append(1.0 / rank)
                expected_values["HR@1"].append(1.0 if rank <= 1 else 0.0)
                expected_values["HR@3"].append(1.0 if rank <= 3 else 0.0)
                expected_values["NDCG@3"].append(_ndcg(labels, 3))
                expected_values["NDCG@5"].append(_ndcg(labels, 5))
                expected_values["HR@10"].append(1.0 if rank <= 10 else 0.0)
            for key in metrics:
                metrics[key].append(mean(expected_values[key]))
        else:
            labels = sorted([_label(row) for row in items], reverse=True)
            for key in metrics:
                if key == "MRR":
                    metrics[key].append(_mrr(labels))
                elif key.startswith("HR"):
                    metrics[key].append(_hr(labels, int(key.split("@")[1])))
                else:
                    metrics[key].append(_ndcg(labels, int(key.split("@")[1])))
    return {k: mean(v) if v else 0.0 for k, v in metrics.items()}


def diagnostics(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        scores = [_score(row, "recommend_true_frequency") for row in rows]
        labels = [_label(row) for row in rows]
        decisions = [1 if bool(row.get("majority_recommend", False)) else 0 for row in rows]
        correctness = [1 if p == y else 0 for p, y in zip(decisions, labels)]
        conf = [_score(row, "self_consistency_confidence") for row in rows]
        rel_binary = _binary_metrics(scores, labels)
        high_conf_wrong = [1 for s, y in zip(conf, correctness) if s >= 0.8 and y == 0]
        out.append(
            {
                "split": split,
                "num_rows": len(rows),
                "num_users": len(_group_by_user(rows)),
                "parse_success_rate": mean([_score(row, "parse_success_rate") for row in rows]) if rows else 0.0,
                "positive_label_rate": mean(labels) if labels else 0.0,
                "recommend_true_rate": mean(decisions) if decisions else 0.0,
                "AUROC": auroc(scores, labels),
                "Brier": brier(scores, labels),
                "ECE": ece(scores, labels),
                "accuracy_at_threshold_0.5": rel_binary["accuracy"],
                "f1_at_threshold_0.5": rel_binary["f1"],
                "balanced_accuracy_at_threshold_0.5": rel_binary["balanced_accuracy"],
                "AUROC_for_correctness": auroc(conf, correctness),
                "ECE_for_correctness": ece(conf, correctness),
                "Brier_for_correctness": brier(conf, correctness),
                "high_conf_error_rate": len(high_conf_wrong) / len(rows) if rows else 0.0,
                "score_mean": mean(scores) if scores else 0.0,
                "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
                "confidence_mean": mean(conf) if conf else 0.0,
                "confidence_std": pstdev(conf) if len(conf) > 1 else 0.0,
                "unique_count": len(set(scores)),
            }
        )
    return out


def calibration_rows(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    targets = [
        ("recommend_true_frequency_to_label", "recommend_true_frequency", lambda row: _label(row)),
        (
            "self_consistency_confidence_to_correctness",
            "self_consistency_confidence",
            lambda row: int(bool(row.get("majority_recommend", False)) == (_label(row) == 1)),
        ),
    ]
    for target, key, label_fn in targets:
        valid_scores = [_score(row, key) for row in valid_rows]
        valid_labels = [label_fn(row) for row in valid_rows]
        test_scores = [_score(row, key) for row in test_rows]
        test_labels = [label_fn(row) for row in test_rows]
        cal_scores = _calibrate(valid_scores, valid_labels, test_scores)
        raw = {"ECE": ece(test_scores, test_labels), "Brier": brier(test_scores, test_labels), "AUROC": auroc(test_scores, test_labels)}
        cal = {"ECE": ece(cal_scores, test_labels), "Brier": brier(cal_scores, test_labels), "AUROC": auroc(cal_scores, test_labels)}
        rows.append({"target": target, "score_type": "raw", **raw, "calibration_method": "none", "status": "ok"})
        rows.append(
            {
                "target": target,
                "score_type": "logistic_calibrated",
                **cal,
                "calibration_method": "logistic",
                "status": "ok",
                "delta_ECE_vs_raw": cal["ECE"] - raw["ECE"],
                "delta_Brier_vs_raw": cal["Brier"] - raw["Brier"],
            }
        )
    return rows


def ranking_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in rows_by_split.items():
        for method, key in [("self_consistency_true_frequency", "recommend_true_frequency")]:
            out.append({"split": split, "ranking_method": method, **ranking_metrics(rows, key), "random_baseline": json.dumps(random_ranking_metrics(rows), sort_keys=True), "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "random", **random_ranking_metrics(rows), "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "oracle_upper_bound": 1.0})
        out.append({"split": split, "ranking_method": "oracle", "NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "oracle_upper_bound": 1.0})
    return out


def _metric_bundle(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    scores = [_score(row, score_key) for row in rows]
    labels = [_label(row) for row in rows]
    rank = ranking_metrics(rows, score_key)
    return {
        "num_users": len(_group_by_user(rows)),
        "num_rows": len(rows),
        "AUROC": auroc(scores, labels),
        "ECE": ece(scores, labels),
        "Brier": brier(scores, labels),
        **rank,
        "recommend_true_rate": mean([1 if score >= 0.5 else 0 for score in scores]) if scores else 0.0,
        "score_mean": mean(scores) if scores else 0.0,
        "score_std": pstdev(scores) if len(scores) > 1 else 0.0,
        "unique_count": len(set(scores)),
    }


def comparison_rows(cfg: dict[str, Any], self_rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    logit_dir = Path(str(cfg["logit_subset_output_dir"]))
    logit_by_split = {split: _read_pred(logit_dir / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    out = []
    for split in ["valid", "test"]:
        self_rows = self_rows_by_split[split]
        logit_rows = logit_by_split[split]
        random_metrics = random_ranking_metrics(self_rows)
        oracle = {"NDCG@10": 1.0, "MRR": 1.0, "HR@1": 1.0, "HR@3": 1.0, "NDCG@3": 1.0, "NDCG@5": 1.0}
        methods = []
        if logit_rows:
            valid_logit = logit_by_split["valid"]
            valid_scores = [_score(row, "positive_relevance_score") for row in valid_logit]
            valid_labels = [_label(row) for row in valid_logit]
            cal_scores = _calibrate(valid_scores, valid_labels, [_score(row, "positive_relevance_score") for row in logit_rows])
            for row, score in zip(logit_rows, cal_scores):
                row["calibrated_logit_ptrue"] = score
            methods.extend([("logit_ptrue", logit_rows, "positive_relevance_score"), ("calibrated_logit_ptrue", logit_rows, "calibrated_logit_ptrue")])
        valid_self = self_rows_by_split["valid"]
        valid_self_scores = [_score(row, "recommend_true_frequency") for row in valid_self]
        valid_self_labels = [_label(row) for row in valid_self]
        cal_self_scores = _calibrate(valid_self_scores, valid_self_labels, [_score(row, "recommend_true_frequency") for row in self_rows])
        for row, score in zip(self_rows, cal_self_scores):
            row["calibrated_self_consistency"] = score
        methods.extend([("self_consistency_true_frequency", self_rows, "recommend_true_frequency"), ("calibrated_self_consistency", self_rows, "calibrated_self_consistency")])
        for method, rows, key in methods:
            metric = _metric_bundle(rows, key)
            out.append({"method": method, "split": split, **metric, "notes": "same Day1f user subset"})
        out.append({"method": "random", "split": split, "num_users": len(_group_by_user(self_rows)), "num_rows": len(self_rows), "AUROC": "NA", "ECE": "NA", "Brier": "NA", **random_metrics, "recommend_true_rate": "NA", "score_mean": "NA", "score_std": "NA", "unique_count": "NA", "notes": "expected random ranking baseline"})
        out.append({"method": "oracle", "split": split, "num_users": len(_group_by_user(self_rows)), "num_rows": len(self_rows), "AUROC": 1.0, "ECE": 0.0, "Brier": 0.0, **oracle, "HR@10": 1.0, "candidate_pool_size_mean": 6.0, "hr10_trivial_flag": True, "recommend_true_rate": "NA", "score_mean": "NA", "score_std": "NA", "unique_count": "NA", "notes": "oracle ranking upper bound"})
    return out


def write_go_no_go(comp_rows: list[dict[str, Any]]) -> str:
    test = [row for row in comp_rows if row.get("split") == "test"]
    logit = next((row for row in test if row.get("method") == "logit_ptrue"), {})
    selfc = next((row for row in test if row.get("method") == "self_consistency_true_frequency"), {})
    recommendation = "move_to_pair_or_list_context_audit"
    if logit and selfc:
        self_better = _score(selfc, "MRR") > _score(logit, "MRR") and _score(selfc, "AUROC") > _score(logit, "AUROC")
        similar = abs(_score(selfc, "MRR") - _score(logit, "MRR")) < 0.02 and abs(_score(selfc, "AUROC") - _score(logit, "AUROC")) < 0.02
        if self_better:
            recommendation = "use_self_consistency_for_confidence_line"
        elif similar:
            recommendation = "keep_logit_confidence_as_primary"
        elif _score(selfc, "MRR") < _score(logit, "MRR"):
            recommendation = "do_not_use_self_consistency"
    text = f"""# Framework-Observation-Day1f Go/No-Go Decision

## Recommendation

`{recommendation}`

## Interpretation

Self-consistency is compared only against logit P(true) on the exact same Day1f 100-user valid/test subsets. If both methods remain weak, the next route is pair/list context rather than more scalar confidence wording.

## Test Snapshot

- logit P(true) MRR/AUROC: `{logit.get('MRR', 'NA')}` / `{logit.get('AUROC', 'NA')}`
- self-consistency MRR/AUROC: `{selfc.get('MRR', 'NA')}` / `{selfc.get('AUROC', 'NA')}`
- logit P(true) HR@1/NDCG@3: `{logit.get('HR@1', 'NA')}` / `{logit.get('NDCG@3', 'NA')}`
- self-consistency HR@1/NDCG@3: `{selfc.get('HR@1', 'NA')}` / `{selfc.get('NDCG@3', 'NA')}`
"""
    GO_NO_GO_MD.write_text(text, encoding="utf-8")
    return recommendation


def write_report(diag: list[dict[str, Any]], cal: list[dict[str, Any]], rank: list[dict[str, Any]], comp: list[dict[str, Any]], recommendation: str) -> None:
    test_diag = next((row for row in diag if row.get("split") == "test"), {})
    test_rank = next((row for row in rank if row.get("split") == "test" and row.get("ranking_method") == "self_consistency_true_frequency"), {})
    test_random = next((row for row in rank if row.get("split") == "test" and row.get("ranking_method") == "random"), {})
    report = f"""# Framework-Observation-Day1f Self-Consistency Confidence Report

## Scope

Day1f is a local Beauty 100-user valid/test smoke with complete six-candidate pools. It does not train, use evidence, use CEP, call external APIs, or run four domains.

## Self-Consistency Signal

- test AUROC: `{test_diag.get('AUROC', 'NA')}`
- test ECE/Brier: `{test_diag.get('ECE', 'NA')}` / `{test_diag.get('Brier', 'NA')}`
- test correctness AUROC: `{test_diag.get('AUROC_for_correctness', 'NA')}`
- test parse success rate: `{test_diag.get('parse_success_rate', 'NA')}`

## Ranking

- self-consistency test MRR / random MRR: `{test_rank.get('MRR', 'NA')}` / `{test_random.get('MRR', 'NA')}`
- self-consistency test HR@1 / random HR@1: `{test_rank.get('HR@1', 'NA')}` / `{test_random.get('HR@1', 'NA')}`
- self-consistency test NDCG@3 / random NDCG@3: `{test_rank.get('NDCG@3', 'NA')}` / `{test_random.get('NDCG@3', 'NA')}`

## Logit Comparison

See `data_done/framework_observation_day1f_logit_vs_self_consistency_comparison.csv` for same-subset comparison.

## Recommendation

`{recommendation}`
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def write_day1g_plan() -> None:
    text = """# Framework-Observation-Day1g Pair/List Context Plan

If pointwise logit and self-consistency signals remain weak, the next step is not more scalar confidence wording. The input context should change.

1. Pairwise context: present a positive-like and negative-like candidate pair for the same user and ask which is more suitable.
2. Listwise context: present the full six-candidate pool and ask for a relative ranking or relative score for every candidate.
3. Confidence/uncertainty should then be extracted from relative decisions, rank margins, or calibrated pair/list probabilities.

Day1g should remain a Beauty smoke first. Do not run full Beauty or four domains until pair/list context shows a stronger signal than pointwise logit/self-consistency.
"""
    DAY1G_PLAN_MD.write_text(text, encoding="utf-8")


def analyze(cfg: dict[str, Any]) -> None:
    pred_dir = Path(str(cfg["output_dir"]))
    rows_by_split = {split: _read_pred(pred_dir / f"{split}_raw.jsonl") for split in ["valid", "test"]}
    diag = diagnostics(rows_by_split)
    cal = calibration_rows(rows_by_split["valid"], rows_by_split["test"])
    rank = ranking_rows(rows_by_split)
    comp = comparison_rows(cfg, rows_by_split)
    _write_csv(DIAG_CSV, diag)
    _write_csv(CAL_CSV, cal)
    _write_csv(RANKING_CSV, rank)
    _write_csv(COMPARISON_CSV, comp)
    recommendation = write_go_no_go(comp)
    write_report(diag, cal, rank, comp, recommendation)
    write_day1g_plan()
    print(json.dumps({"diagnostics": str(DIAG_CSV), "calibration": str(CAL_CSV), "ranking": str(RANKING_CSV), "comparison": str(COMPARISON_CSV), "report": str(REPORT_MD), "go_no_go": str(GO_NO_GO_MD), "day1g_plan": str(DAY1G_PLAN_MD)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Observation-Day1f self-consistency confidence audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--create_subset", action="store_true")
    parser.add_argument("--run_self_consistency", choices=["valid", "test"], default=None)
    parser.add_argument("--run_logit_subset", choices=["valid", "test"], default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--model_variant", choices=["base", "lora"], default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = _read_config(args.config)
    variant = args.model_variant or str(cfg.get("model_variant", "lora"))
    if args.create_subset:
        subset = create_subset(cfg)
        print(json.dumps({"subset_path": str(cfg.get("subset_path", SUBSET_JSON)), "subsets": subset["subsets"]}, ensure_ascii=False, indent=2))
    if args.run_self_consistency:
        path = run_self_consistency(cfg, args.run_self_consistency, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_self_consistency, "mode": "self_consistency"}, ensure_ascii=False, indent=2))
    if args.run_logit_subset:
        path = run_logit_subset(cfg, args.run_logit_subset, variant, args.resume)
        print(json.dumps({"output_path": str(path), "split": args.run_logit_subset, "mode": "logit_subset"}, ensure_ascii=False, indent=2))
    if args.analyze_only:
        analyze(cfg)


if __name__ == "__main__":
    main()
