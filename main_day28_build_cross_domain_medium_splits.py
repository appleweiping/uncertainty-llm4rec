"""Build regular-domain medium pointwise splits for cross-domain validation.

Day28 is data readiness only: no DeepSeek API calls, no LoRA, no backbone
training. Medium splits are built from regular processed interactions/items,
not from the old small directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SUMMARY_DIR = Path("output-repaired/summary")
BEAUTY_DAY9_ROWS = 5838 + 5838
DOMAINS = {
    "movies": Path("data/processed/amazon_movies"),
    "books": Path("data/processed/amazon_books"),
    "electronics": Path("data/processed/amazon_electronics"),
}


def _fast_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 16), b""):
            count += chunk.count(b"\n")
    return max(count - 1, 0)


def _csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return next(csv.reader(f), [])


def _read_item_ids(items_path: Path) -> list[str]:
    item_ids: list[str] = []
    with items_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if item_id:
                item_ids.append(item_id)
    return item_ids


def _finish_user(
    user_id: str | None,
    events: list[tuple[str, int]],
    item_set: set[str],
) -> tuple[str, list[tuple[str, int]]] | None:
    if not user_id:
        return None
    filtered = [(item, ts) for item, ts in events if item in item_set]
    if not filtered:
        return None
    # Match Beauty sample_builder: chronological order, then keep first item occurrence.
    filtered.sort(key=lambda x: x[1])
    seen = set()
    deduped: list[tuple[str, int]] = []
    for item, ts in filtered:
        if item in seen:
            continue
        seen.add(item)
        deduped.append((item, ts))
    if len(deduped) < 3:
        return None
    return user_id, deduped


def _sample_eligible_users(
    interactions_path: Path,
    item_set: set[str],
    medium_users: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    eligible_users = 0
    raw_users = 0
    raw_interactions = 0
    current_user: str | None = None
    current_events: list[tuple[str, int]] = []

    def consider(user_id: str | None, events: list[tuple[str, int]]) -> None:
        nonlocal eligible_users, sampled, raw_users
        if user_id is None:
            return
        raw_users += 1
        finished = _finish_user(user_id, events, item_set)
        if finished is None:
            return
        uid, seq = finished
        eligible_users += 1
        record = {"user_id": uid, "sequence": seq}
        if len(sampled) < medium_users:
            sampled.append(record)
        else:
            j = rng.randint(1, eligible_users)
            if j <= medium_users:
                sampled[j - 1] = record

    with interactions_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = str(row.get("user_id", "")).strip()
            item_id = str(row.get("item_id", "")).strip()
            if not user_id or not item_id:
                continue
            try:
                ts = int(float(row.get("timestamp", 0)))
            except Exception:
                ts = 0
            raw_interactions += 1
            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                consider(current_user, current_events)
                current_user = user_id
                current_events = []
            current_events.append((item_id, ts))
    consider(current_user, current_events)
    sampled.sort(key=lambda x: x["user_id"])
    stats = {
        "raw_users_seen": raw_users,
        "raw_interactions_seen": raw_interactions,
        "eligible_users": eligible_users,
        "medium_users": len(sampled),
        "sampling_seed": seed,
        "sampling_method": "reservoir_sample_eligible_users_then_sort_user_id",
    }
    return sampled, stats


def _sample_negatives(
    all_item_ids: list[str],
    user_seen: set[str],
    num_negatives: int,
    rng: random.Random,
) -> list[str]:
    negatives: list[str] = []
    blocked = set(user_seen)
    attempts = 0
    while len(negatives) < num_negatives and attempts < num_negatives * 200:
        attempts += 1
        item_id = rng.choice(all_item_ids)
        if item_id in blocked or item_id in negatives:
            continue
        negatives.append(item_id)
    return negatives


def _record_skeletons(
    sampled_users: list[dict[str, Any]],
    all_item_ids: list[str],
    seed: int,
    num_negatives: int,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "test": []}
    needed_items: set[str] = set()
    for user in sampled_users:
        user_id = user["user_id"]
        seq = user["sequence"]
        item_ids = [item for item, _ in seq]
        timestamps = [ts for _, ts in seq]
        user_seen = set(item_ids)
        train_items = item_ids[:-2]
        train_timestamps = timestamps[:-2]
        valid_item, valid_ts = item_ids[-2], timestamps[-2]
        test_item, test_ts = item_ids[-1], timestamps[-1]
        needed_items.update(item_ids)

        # Train: each non-initial train item becomes a positive target.
        for idx in range(1, len(train_items)):
            history = train_items[:idx]
            target = train_items[idx]
            ts = train_timestamps[idx]
            splits["train"].append(
                {"user_id": user_id, "history_item_ids": history, "candidate_item_id": target, "label": 1, "timestamp": ts}
            )
            negatives = _sample_negatives(all_item_ids, user_seen, num_negatives, rng)
            needed_items.update(negatives)
            for neg in negatives:
                splits["train"].append(
                    {"user_id": user_id, "history_item_ids": history, "candidate_item_id": neg, "label": 0, "timestamp": ts}
                )

        for split, target, ts in [("valid", valid_item, valid_ts), ("test", test_item, test_ts)]:
            history = train_items
            splits[split].append(
                {"user_id": user_id, "history_item_ids": history, "candidate_item_id": target, "label": 1, "timestamp": ts}
            )
            negatives = _sample_negatives(all_item_ids, user_seen, num_negatives, rng)
            needed_items.update(negatives)
            for neg in negatives:
                splits[split].append(
                    {"user_id": user_id, "history_item_ids": history, "candidate_item_id": neg, "label": 0, "timestamp": ts}
                )
    return splits, needed_items


def _load_item_lookup(items_path: Path, needed_items: set[str]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with items_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row.get("item_id", "")).strip()
            if item_id not in needed_items:
                continue
            title = str(row.get("title", "") or "").strip()
            candidate_text = str(row.get("candidate_text", "") or "").strip()
            if not candidate_text:
                candidate_text = f"Title: {title}" if title else f"Item ID: {item_id}"
            lookup[item_id] = {
                "candidate_title": title,
                "candidate_text": candidate_text,
                "popularity_group": str(row.get("popularity_group", "") or "mid").strip() or "mid",
            }
    return lookup


def _format_history(history_item_ids: list[str], lookup: dict[str, dict[str, str]], max_history_len: int) -> list[str]:
    history = []
    for item_id in history_item_ids[-max_history_len:]:
        info = lookup.get(item_id, {})
        title = info.get("candidate_title", "")
        candidate_text = info.get("candidate_text", "")
        history.append(title or candidate_text[:200] or item_id)
    return history


def _materialize_records(
    skeletons: dict[str, list[dict[str, Any]]],
    lookup: dict[str, dict[str, str]],
    max_history_len: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split, rows in skeletons.items():
        materialized = []
        for row in rows:
            item_id = row["candidate_item_id"]
            info = lookup.get(item_id, {})
            materialized.append(
                {
                    "user_id": row["user_id"],
                    "history": _format_history(row["history_item_ids"], lookup, max_history_len),
                    "candidate_item_id": item_id,
                    "candidate_title": info.get("candidate_title", ""),
                    "candidate_text": info.get("candidate_text", f"Item ID: {item_id}"),
                    "label": int(row["label"]),
                    "target_popularity_group": info.get("popularity_group", "mid"),
                    "timestamp": row["timestamp"],
                }
            )
        out[split] = materialized
    return out


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_split(domain: str, processed_path: Path, split: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(records)
    required = ["user_id", "history", "candidate_item_id", "candidate_text", "label"]
    missing = [c for c in required if c not in df.columns]
    counts = df.groupby("user_id").size() if "user_id" in df.columns and len(df) else pd.Series(dtype=float)
    labels = df["label"].astype(int) if "label" in df.columns and len(df) else pd.Series(dtype=int)
    return {
        "domain": domain,
        "processed_path": str(processed_path),
        "split": split,
        "num_rows": int(len(df)),
        "num_users": int(df["user_id"].astype(str).nunique()) if "user_id" in df.columns and len(df) else 0,
        "num_items": int(df["candidate_item_id"].astype(str).nunique()) if "candidate_item_id" in df.columns and len(df) else 0,
        "positive_rows": int((labels == 1).sum()) if len(labels) else 0,
        "negative_rows": int((labels == 0).sum()) if len(labels) else 0,
        "avg_candidates_per_user": float(counts.mean()) if len(counts) else math.nan,
        "min_candidates_per_user": float(counts.min()) if len(counts) else math.nan,
        "max_candidates_per_user": float(counts.max()) if len(counts) else math.nan,
        "has_user_id": "user_id" in df.columns,
        "has_history": "history" in df.columns,
        "has_candidate_item_id": "candidate_item_id" in df.columns,
        "has_candidate_text": "candidate_text" in df.columns,
        "has_candidate_title": "candidate_title" in df.columns,
        "has_label": "label" in df.columns,
        "schema_compatible_with_beauty": not missing and "candidate_title" in df.columns,
        "missing_fields": ",".join(missing),
        "notes": "medium split from regular processed domain",
    }


def _medium_size(eligible_users: int, requested_users: int) -> int:
    if eligible_users >= 2000:
        return min(requested_users, eligible_users)
    if eligible_users >= 500:
        return min(1000, eligible_users)
    return 0


def build_domain(domain: str, source_dir: Path, requested_users: int, seed: int, num_negatives: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    interactions_path = source_dir / "interactions.csv"
    items_path = source_dir / "items.csv"
    users_path = source_dir / "users.csv"
    output_dir = Path(f"data/processed/amazon_{domain}_medium")
    inventory = {
        "domain": domain,
        "source_dir": str(source_dir),
        "interactions_exists": interactions_path.exists(),
        "interactions_rows": _fast_line_count(interactions_path),
        "interactions_fields": _csv_header(interactions_path),
        "items_exists": items_path.exists(),
        "items_rows": _fast_line_count(items_path),
        "items_fields": _csv_header(items_path),
        "users_exists": users_path.exists(),
        "users_rows": _fast_line_count(users_path),
    }
    if not interactions_path.exists() or not items_path.exists():
        stats = {**inventory, "status": "blocked_schema", "eligible_users": 0, "medium_users": 0}
        return stats, []

    item_ids = _read_item_ids(items_path)
    item_set = set(item_ids)
    sampled_raw, sample_stats = _sample_eligible_users(interactions_path, item_set, requested_users, seed)
    medium_users = _medium_size(sample_stats["eligible_users"], requested_users)
    if medium_users == 0:
        stats = {**inventory, **sample_stats, "status": "domain_too_small_for_medium", "medium_users": 0}
        return stats, []
    sampled_users = sampled_raw[:medium_users]
    skeletons, needed_items = _record_skeletons(sampled_users, item_ids, seed, num_negatives)
    lookup = _load_item_lookup(items_path, needed_items)
    records = _materialize_records(skeletons, lookup, max_history_len=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_records in records.items():
        _write_jsonl(split_records, output_dir / f"{split}.jsonl")
    sampled_user_payload = [
        {"user_id": row["user_id"], "num_unique_interactions": len(row["sequence"])} for row in sampled_users
    ]
    (output_dir / "sampled_users.json").write_text(json.dumps(sampled_user_payload, indent=2), encoding="utf-8")
    validation_rows = [_validate_split(domain, output_dir, split, split_records) for split, split_records in records.items()]
    split_stats = {
        **inventory,
        **sample_stats,
        "status": "ready",
        "medium_users": medium_users,
        "output_dir": str(output_dir),
        "seed": seed,
        "num_negatives": num_negatives,
        "train_rows": len(records["train"]),
        "valid_rows": len(records["valid"]),
        "test_rows": len(records["test"]),
        "valid_test_total_rows": len(records["valid"]) + len(records["test"]),
    }
    (output_dir / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    (output_dir / "schema_validation.json").write_text(json.dumps(validation_rows, indent=2), encoding="utf-8")
    return split_stats, validation_rows


def _cost_rows(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stats in stats_rows:
        valid_rows = int(stats.get("valid_rows", 0))
        test_rows = int(stats.get("test_rows", 0))
        total = valid_rows + test_rows
        relative = total / BEAUTY_DAY9_ROWS if BEAUTY_DAY9_ROWS else math.nan
        if stats.get("status") != "ready":
            mode = "blocked_schema"
            reason = "Domain did not produce a ready medium split."
        elif stats["domain"] == "movies":
            mode = "movies_medium_first"
            reason = "Movies is the recommended first cross-domain medium run."
        elif stats["domain"] == "books":
            mode = "books_medium_first"
            reason = "Books medium is ready after Movies."
        else:
            mode = "electronics_medium_first"
            reason = "Electronics medium is ready but recommended after Movies/Books as a harder/noisier domain."
        rows.append(
            {
                "domain": stats["domain"],
                "medium_users": int(stats.get("medium_users", 0)),
                "valid_rows": valid_rows,
                "test_rows": test_rows,
                "total_api_rows": total,
                "relative_to_beauty_day9": relative,
                "recommended_day29_mode": mode,
                "reason": reason,
            }
        )
    return rows


def _write_configs(stats_rows: list[dict[str, Any]]) -> None:
    Path("configs/exp").mkdir(parents=True, exist_ok=True)
    Path("configs/external_backbone").mkdir(parents=True, exist_ok=True)
    for stats in stats_rows:
        if stats.get("status") != "ready":
            continue
        domain = stats["domain"]
        processed = f"data/processed/amazon_{domain}_medium"
        exp_config = {
            "exp_name": f"{domain}_deepseek_relevance_evidence_medium",
            "domain": domain,
            "train_input_path": f"{processed}/train.jsonl",
            "split_input_paths": {
                "valid": f"{processed}/valid.jsonl",
                "test": f"{processed}/test.jsonl",
            },
            "prompt_path": "prompts/candidate_relevance_evidence.txt",
            "output_root": "output-repaired",
            "output_dir": f"output-repaired/{domain}_deepseek_relevance_evidence_medium",
            "model_config": "configs/model/deepseek.yaml",
            "output_schema": "relevance_evidence",
            "method_variant": "candidate_relevance_evidence_posterior_medium",
            "resume": True,
            "concurrent": True,
            "max_workers": 4,
            "requests_per_minute": 120,
            "max_retries": 3,
            "retry_backoff_seconds": 2.0,
            "checkpoint_every": 1,
            "max_samples": None,
            "notes": "Day28 medium config only. Do not launch automatically.",
        }
        with Path(f"configs/exp/{domain}_deepseek_relevance_evidence_medium.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(exp_config, f, sort_keys=False, allow_unicode=True)
        backbone_config = {
            "backbone_name": "sasrec",
            "domain": domain,
            "stage": "cross_domain_medium",
            "train_input_path": f"{processed}/train.jsonl",
            "valid_input_path": f"{processed}/valid.jsonl",
            "test_input_path": f"{processed}/test.jsonl",
            "score_output_path": f"output-repaired/backbone/sasrec_{domain}_medium/candidate_scores.csv",
            "evidence_table": {
                "path": f"output-repaired/{domain}_deepseek_relevance_evidence_medium/calibrated/relevance_evidence_posterior_test.jsonl",
                "join_keys": ["user_id", "candidate_item_id"],
                "fields": [
                    "relevance_probability",
                    "calibrated_relevance_probability",
                    "evidence_risk",
                    "ambiguity",
                    "missing_information",
                    "abs_evidence_margin",
                    "positive_evidence",
                    "negative_evidence",
                ],
            },
            "rerank": {
                "top_k": 10,
                "normalizations": ["minmax", "zscore"],
                "lambdas": [0.0, 0.05, 0.1, 0.2, 0.5],
                "alphas": [0.5, 0.75, 0.9],
                "settings": [
                    "Backbone only",
                    "Backbone + calibrated relevance",
                    "Backbone + evidence risk",
                    "Backbone + calibrated relevance + evidence risk",
                ],
            },
        }
        with Path(f"configs/external_backbone/{domain}_sasrec_plugin_medium.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(backbone_config, f, sort_keys=False, allow_unicode=True)


def _markdown_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if math.isnan(val) else f"{val:.4f}")
            else:
                vals.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(stats_df: pd.DataFrame, validation_df: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    text = f"""# Day28 Cross-domain Medium Benchmark Report

## 1. Why Not Direct Full

Regular Books/Electronics/Movies are large. Direct full evidence inference would mix data repair, API cost, and experimental validation all at once. Day28 therefore builds a standardized medium benchmark first.

## 2. Why Not Old Small

The old small domains are useful for quick debugging, but they are too small to carry the final cross-domain claim. The medium benchmark is sampled from regular processed domain interactions and items, so it is not a toy small split.

## 3. Medium Benchmark Definition

- Source: regular `data/processed/amazon_*` interactions/items/users.
- Sampling: eligible users have at least 3 unique chronological interactions.
- User sample: reservoir sampling with seed `42`, then deterministic user-id sorting.
- Split: user-level chronological leave-one-out; last interaction is test, second-to-last is valid, earlier interactions are train history.
- Negative sampling: seed `42`, 5 negatives per positive, excluding items already interacted with by the user.
- Schema: Beauty-compatible pointwise JSONL with `user_id`, `history`, `candidate_item_id`, `candidate_title`, `candidate_text`, `label`, `target_popularity_group`, and `timestamp`.

## 4. Eligible Users And Medium Users

{_markdown_table(stats_df[['domain', 'eligible_users', 'medium_users', 'train_rows', 'valid_rows', 'test_rows', 'status', 'output_dir']])}

## 5. Schema Validation

{_markdown_table(validation_df)}

## 6. API Cost Estimate

{_markdown_table(cost_df)}

## 7. Day29 Recommendation

Run **Movies medium relevance evidence** first. Movies is the preferred first cross-domain run because it already had partial pointwise traces and now has a repaired medium split from the regular domain. Books and Electronics medium splits are built and validated, but should follow after Movies.
"""
    (SUMMARY_DIR / "day28_cross_domain_medium_benchmark_report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["movies", "books", "electronics"])
    parser.add_argument("--medium_users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_negatives", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    validation_rows = []
    for domain in args.domains:
        source = DOMAINS[domain]
        stats, validation = build_domain(domain, source, args.medium_users, args.seed, args.num_negatives)
        stats_rows.append(stats)
        validation_rows.extend(validation)
    stats_df = pd.DataFrame(stats_rows)
    validation_df = pd.DataFrame(validation_rows)
    cost_df = pd.DataFrame(_cost_rows(stats_rows))
    validation_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_schema_validation.csv", index=False)
    cost_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_cost_estimate.csv", index=False)
    stats_df.to_csv(SUMMARY_DIR / "day28_cross_domain_medium_split_stats.csv", index=False)
    _write_configs(stats_rows)
    _write_report(stats_df, validation_df, cost_df)
    print("Day28 cross-domain medium benchmark construction complete.")


if __name__ == "__main__":
    main()
