from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5", "HR@10"]


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_ids_from_listwise(sample: dict[str, Any]) -> list[str]:
    return [
        str(x.get("candidate_item_id", "")).strip()
        for x in sample.get("input", {}).get("candidate_pool", [])
        if str(x.get("candidate_item_id", "")).strip()
    ]


def target_id_from_listwise(sample: dict[str, Any]) -> str:
    output = sample.get("output", {})
    target = str(output.get("target_item_id", "")).strip()
    if target:
        return target
    ranked = output.get("ranked_item_ids", [])
    return str(ranked[0]).strip() if ranked else ""


def rank_metrics(rank: int, pool_size: int) -> dict[str, float]:
    if rank < 1 or rank > pool_size:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        "MRR": 1.0 / rank,
        "HR@1": 1.0 if rank <= 1 else 0.0,
        "HR@3": 1.0 if rank <= 3 else 0.0,
        "HR@10": 1.0 if rank <= 10 else 0.0,
        "NDCG@3": 1.0 / math.log2(rank + 1) if rank <= 3 else 0.0,
        "NDCG@5": 1.0 / math.log2(rank + 1) if rank <= 5 else 0.0,
        "NDCG@10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
    }


def complete_ranking(ranked: list[str], candidate_pool: list[str], tie_break_policy: str = "lexical") -> list[str]:
    seen = set()
    cleaned = []
    pool_set = set(candidate_pool)
    for item_id in ranked:
        iid = str(item_id).strip()
        if iid in pool_set and iid not in seen:
            cleaned.append(iid)
            seen.add(iid)
    missing = [iid for iid in candidate_pool if iid not in seen]
    if tie_break_policy == "lexical":
        missing = sorted(missing)
    elif tie_break_policy.startswith("seeded_random"):
        seed = int(tie_break_policy.split(":")[1]) if ":" in tie_break_policy else 42
        rng = random.Random(seed)
        rng.shuffle(missing)
    elif tie_break_policy == "candidate_order":
        pass
    else:
        raise ValueError(f"Unsupported tie_break_policy={tie_break_policy}")
    return cleaned + missing


def rankings_from_listwise_predictions(
    samples: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    tie_break_policy: str = "lexical",
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    rankings = []
    parse_rows = []
    for sample, pred in zip(samples, pred_rows):
        pool = candidate_ids_from_listwise(sample)
        ranked = [str(x).strip() for x in pred.get("ranked_item_ids", [])]
        complete = complete_ranking(ranked, pool, tie_break_policy=tie_break_policy)
        rankings.append(complete)
        parse_rows.append(
            {
                "parse_success": bool(pred.get("parse_success", False)),
                "schema_valid": bool(pred.get("schema_valid", False)),
                "invalid_item_count": int(pred.get("invalid_item_count", 0) or 0),
                "duplicate_item_count": int(pred.get("duplicate_item_count", 0) or 0),
                "raw_output_items": len(pred.get("raw_ranked_item_ids", pred.get("ranked_item_ids", []))),
            }
        )
    return rankings, parse_rows


def evaluate_rankings(
    method: str,
    samples: list[dict[str, Any]],
    rankings: list[list[str]],
    parse_rows: list[dict[str, Any]] | None = None,
    tie_break_policy: str = "lexical",
) -> dict[str, Any]:
    pool_sizes = []
    metric_rows = []
    for sample, ranking in zip(samples, rankings):
        pool = candidate_ids_from_listwise(sample)
        target = target_id_from_listwise(sample)
        pool_sizes.append(len(pool))
        rank = ranking.index(target) + 1 if target in ranking else len(pool) + 1
        metric_rows.append(rank_metrics(rank, len(pool)))
    row: dict[str, Any] = {
        "method": method,
        "num_samples": len(samples),
        "candidate_pool_size_mean": mean(pool_sizes) if pool_sizes else 0,
        "hr10_trivial_flag": (max(pool_sizes) <= 10 if pool_sizes else True),
        "tie_break_policy": tie_break_policy,
    }
    for key in METRIC_KEYS:
        row[key] = mean([m[key] for m in metric_rows]) if metric_rows else 0.0
    if parse_rows is None:
        row.update(
            {
                "parse_success_rate": 1.0,
                "schema_valid_rate": 1.0,
                "invalid_item_rate": 0.0,
                "duplicate_item_rate": 0.0,
            }
        )
    else:
        total = len(parse_rows)
        total_output_items = sum(int(r.get("raw_output_items", 0) or 0) for r in parse_rows)
        row.update(
            {
                "parse_success_rate": sum(1 for r in parse_rows if r.get("parse_success")) / total if total else 0.0,
                "schema_valid_rate": sum(1 for r in parse_rows if r.get("schema_valid")) / total if total else 0.0,
                "invalid_item_rate": sum(int(r.get("invalid_item_count", 0) or 0) for r in parse_rows) / total_output_items if total_output_items else 0.0,
                "duplicate_item_rate": sum(int(r.get("duplicate_item_count", 0) or 0) for r in parse_rows) / total_output_items if total_output_items else 0.0,
            }
        )
    return row


def random_rankings(samples: list[dict[str, Any]], seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    out = []
    for sample in samples:
        pool = candidate_ids_from_listwise(sample)
        pool = pool[:]
        rng.shuffle(pool)
        out.append(pool)
    return out


def oracle_rankings(samples: list[dict[str, Any]]) -> list[list[str]]:
    out = []
    for sample in samples:
        pool = candidate_ids_from_listwise(sample)
        target = target_id_from_listwise(sample)
        out.append([target] + sorted([iid for iid in pool if iid != target]))
    return out


def positive_position_rows(dataset_version: str, split: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[int] = Counter()
    pool_sizes = []
    for sample in samples:
        pool = candidate_ids_from_listwise(sample)
        target = target_id_from_listwise(sample)
        pool_sizes.append(len(pool))
        if target in pool:
            counts[pool.index(target) + 1] += 1
    total = sum(counts.values())
    max_pos = max(pool_sizes) if pool_sizes else 0
    return [
        {
            "dataset_version": dataset_version,
            "split": split,
            "position": pos,
            "positive_count": counts[pos],
            "positive_rate": counts[pos] / total if total else 0.0,
            "num_users": total,
            "candidate_pool_size_mean": mean(pool_sizes) if pool_sizes else 0,
        }
        for pos in range(1, max_pos + 1)
    ]
