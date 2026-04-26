from __future__ import annotations

import csv
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42
MAX_USERS = 10000
NEGATIVES_PER_POSITIVE = 5
ROOT = Path("data_done")
PROCESSED_ROOT = Path("data/processed")
RAW_ROOT = Path("data/raw")


DOMAINS = {
    "beauty": {
        "processed": PROCESSED_ROOT / "amazon_beauty",
        "raw_interactions": RAW_ROOT / "amazon_beauty" / "reviews_Beauty.jsonl",
        "raw_items": RAW_ROOT / "amazon_beauty" / "meta_Beauty.jsonl",
        "review_field": "parent_asin",
        "meta_field": "parent_asin",
    },
    "books": {
        "processed": PROCESSED_ROOT / "amazon_books",
        "raw_interactions": RAW_ROOT / "amazon_books" / "Books.jsonl.gz",
        "raw_items": RAW_ROOT / "amazon_books" / "meta_Books.jsonl.gz",
        "review_field": "parent_asin",
        "meta_field": "parent_asin",
    },
    "electronics": {
        "processed": PROCESSED_ROOT / "amazon_electronics",
        "raw_interactions": RAW_ROOT / "amazon_electronics" / "Electronics.jsonl.gz",
        "raw_items": RAW_ROOT / "amazon_electronics" / "meta_Electronics.jsonl.gz",
        "review_field": "parent_asin",
        "meta_field": "parent_asin",
    },
    "movies": {
        "processed": PROCESSED_ROOT / "amazon_movies",
        "raw_interactions": RAW_ROOT / "amazon_movies" / "Movies_and_TV.jsonl.gz",
        "raw_items": RAW_ROOT / "amazon_movies" / "meta_Movies_and_TV.jsonl.gz",
        "review_field": "parent_asin",
        "meta_field": "parent_asin",
    },
}


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _write_json(path: Path, data: Any) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _read_first_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _candidate_text(row: dict[str, Any]) -> str:
    text = _safe_str(row.get("candidate_text"))
    if text:
        return text
    parts = []
    title = _safe_str(row.get("title"))
    categories = _safe_str(row.get("categories"))
    description = _safe_str(row.get("description"))
    if title:
        parts.append(f"Title: {title}")
    if categories:
        parts.append(f"Categories: {categories}")
    if description:
        parts.append(f"Description: {description}")
    return " ".join(parts)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _quantiles(counter: Counter[str]) -> dict[str, float]:
    vals = pd.Series(list(counter.values()), dtype="float64")
    if vals.empty:
        return {k: math.nan for k in ["mean", "median", "p50", "p75", "p90", "p95", "p99"]}
    return {
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "p50": float(vals.quantile(0.50)),
        "p75": float(vals.quantile(0.75)),
        "p90": float(vals.quantile(0.90)),
        "p95": float(vals.quantile(0.95)),
        "p99": float(vals.quantile(0.99)),
    }


def _chunk_interactions(path: Path, chunksize: int = 750_000):
    yield from pd.read_csv(
        path,
        dtype={"user_id": "string", "item_id": "string", "rating": "float64", "timestamp": "Int64"},
        chunksize=chunksize,
    )


def _count_interactions(path: Path, allowed_users: set[str] | None = None, allowed_items: set[str] | None = None) -> tuple[Counter[str], Counter[str], int]:
    user_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    n_rows = 0
    for chunk in _chunk_interactions(path):
        chunk["user_id"] = chunk["user_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)
        if allowed_users is not None:
            chunk = chunk[chunk["user_id"].isin(allowed_users)]
        if allowed_items is not None:
            chunk = chunk[chunk["item_id"].isin(allowed_items)]
        n_rows += len(chunk)
        user_counts.update(chunk["user_id"].tolist())
        item_counts.update(chunk["item_id"].tolist())
    return user_counts, item_counts, n_rows


def _kcore_counts(path: Path, k: int = 5) -> tuple[set[str], set[str], int]:
    allowed_users: set[str] | None = None
    allowed_items: set[str] | None = None
    last = (-1, -1, -1)
    for _ in range(20):
        user_counts, item_counts, rows = _count_interactions(path, allowed_users, allowed_items)
        new_users = {u for u, c in user_counts.items() if c >= k}
        new_items = {i for i, c in item_counts.items() if c >= k}
        state = (len(new_users), len(new_items), rows)
        allowed_users, allowed_items = new_users, new_items
        if state == last:
            break
        last = state
    _, _, final_rows = _count_interactions(path, allowed_users, allowed_items)
    return allowed_users or set(), allowed_items or set(), final_rows


def _load_items(path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for r in df.to_dict("records"):
        item_id = _safe_str(r.get("item_id"))
        if not item_id:
            continue
        out[item_id] = {
            "item_id": item_id,
            "title": _safe_str(r.get("title")),
            "categories": _safe_str(r.get("categories")),
            "description": _safe_str(r.get("description")),
            "candidate_text": _candidate_text(r),
            "popularity_group": _safe_str(r.get("popularity_group")),
        }
    return out


def _item_obj(item_id: str, timestamp: Any, item_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = item_map.get(item_id, {})
    title = _safe_str(meta.get("title")) or f"Unknown item {item_id}"
    text = _safe_str(meta.get("candidate_text")) or f"Item ID: {item_id}"
    return {
        "item_id": item_id,
        "title": title,
        "text": text,
        "timestamp": None if pd.isna(timestamp) else int(timestamp),
    }


def _candidate_obj(user_id: str, history: list[dict[str, Any]], item_id: str, label: int, timestamp: Any, domain: str, split: str, item_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = item_map.get(item_id, {})
    title = _safe_str(meta.get("title")) or f"Unknown item {item_id}"
    text = _safe_str(meta.get("candidate_text")) or f"Item ID: {item_id}"
    return {
        "user_id": user_id,
        "history": history,
        "candidate_item_id": item_id,
        "candidate_title": title,
        "candidate_text": text,
        "label": int(label),
        "timestamp": None if pd.isna(timestamp) else int(timestamp),
        "domain": domain,
        "split": split,
    }


def _sample_negatives(
    rng: random.Random,
    user_seen: set[str],
    train_vocab: list[str],
    all_items: list[str],
    n: int,
) -> tuple[list[str], str]:
    pool = [i for i in train_vocab if i not in user_seen]
    mode = "warm_train_vocab"
    if len(pool) < n:
        pool = [i for i in all_items if i not in user_seen]
        mode = "all_filtered_items_fallback"
    if len(pool) >= n:
        return rng.sample(pool, n), mode
    if not pool:
        return [], "no_available_negative"
    return [rng.choice(pool) for _ in range(n)], "sample_with_replacement_fallback"


def _schema_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    per_user = Counter(r["user_id"] for r in rows)
    labels = sorted(set(r.get("label") for r in rows))
    return {
        "has_user_id": all("user_id" in r and r["user_id"] for r in rows),
        "has_history": all(isinstance(r.get("history"), list) for r in rows),
        "has_candidate_item_id": all(r.get("candidate_item_id") for r in rows),
        "has_candidate_title": all("candidate_title" in r for r in rows),
        "has_candidate_text": all("candidate_text" in r for r in rows),
        "has_label": all("label" in r for r in rows),
        "history_nonempty_rate": sum(1 for r in rows if r.get("history")) / len(rows),
        "candidate_text_nonempty_rate": sum(1 for r in rows if _safe_str(r.get("candidate_text"))) / len(rows),
        "label_values": labels,
        "candidate_pool_size_mean": float(pd.Series(list(per_user.values())).mean()),
        "candidate_pool_size_min": int(min(per_user.values())),
        "candidate_pool_size_max": int(max(per_user.values())),
        "hr10_trivial_flag": max(per_user.values()) <= 10 or float(pd.Series(list(per_user.values())).mean()) <= 10,
    }


def _schema_validation(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], neg_modes: Counter[str]) -> dict[str, Any]:
    valid_stats = _schema_stats(valid_rows)
    test_stats = _schema_stats(test_rows)
    combined = valid_rows + test_rows
    return {
        "has_user_id": all("user_id" in r and r["user_id"] for r in combined),
        "has_history": all(isinstance(r.get("history"), list) for r in combined),
        "has_candidate_item_id": all(r.get("candidate_item_id") for r in combined),
        "has_candidate_title": all("candidate_title" in r for r in combined),
        "has_candidate_text": all("candidate_text" in r for r in combined),
        "has_label": all("label" in r for r in combined),
        "history_nonempty_rate": sum(1 for r in combined if r.get("history")) / len(combined) if combined else math.nan,
        "candidate_text_nonempty_rate": sum(1 for r in combined if _safe_str(r.get("candidate_text"))) / len(combined) if combined else math.nan,
        "label_values": sorted(set(r.get("label") for r in combined)),
        "candidate_pool_size_mean": valid_stats.get("candidate_pool_size_mean", math.nan),
        "candidate_pool_size_min": min(valid_stats.get("candidate_pool_size_min", 0), test_stats.get("candidate_pool_size_min", 0)),
        "candidate_pool_size_max": max(valid_stats.get("candidate_pool_size_max", 0), test_stats.get("candidate_pool_size_max", 0)),
        "valid_candidate_pool_size_mean": valid_stats.get("candidate_pool_size_mean", math.nan),
        "test_candidate_pool_size_mean": test_stats.get("candidate_pool_size_mean", math.nan),
        "hr10_trivial_flag": bool(valid_stats.get("hr10_trivial_flag", True) and test_stats.get("hr10_trivial_flag", True)),
        "negative_sampling_mode": dict(neg_modes),
        "seed": SEED,
    }


def _cold_rows(
    split_name: str,
    rows: list[dict[str, Any]],
    vocab_name: str,
    vocab: set[str],
) -> dict[str, Any]:
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    cand = [r["candidate_item_id"] for r in rows]
    pos_ids = [r["candidate_item_id"] for r in pos]
    neg_ids = [r["candidate_item_id"] for r in neg]
    pos_cold = [i for i in pos_ids if i not in vocab]
    neg_cold = [i for i in neg_ids if i not in vocab]
    all_cold = [i for i in cand if i not in vocab]
    return {
        "split": split_name,
        "vocab_definition": vocab_name,
        "train_vocab_size": len(vocab),
        "num_rows": len(rows),
        "num_positive_rows": len(pos),
        "num_negative_rows": len(neg),
        "positive_cold_rows": len(pos_cold),
        "positive_cold_rate": len(pos_cold) / len(pos) if pos else math.nan,
        "negative_cold_rows": len(neg_cold),
        "negative_cold_rate": len(neg_cold) / len(neg) if neg else math.nan,
        "all_candidate_cold_rows": len(all_cold),
        "all_candidate_cold_rate": len(all_cold) / len(rows) if rows else math.nan,
        "unique_candidate_items": len(set(cand)),
        "unique_cold_candidate_items": len(set(all_cold)),
        "unique_positive_items": len(set(pos_ids)),
        "unique_positive_cold_items": len(set(pos_cold)),
        "unique_negative_items": len(set(neg_ids)),
        "unique_negative_cold_items": len(set(neg_cold)),
    }


def _write_protocol_audit() -> None:
    rows = [
        {
            "paper_name": "OpenP5",
            "domain/dataset": "Beauty, Movies, Electronics and other public rec datasets",
            "filtering rule": "Repo reports fixed preprocessed dataset statistics; README does not specify detailed filtering in the visible local notes.",
            "minimum user interactions": "not specified",
            "minimum item interactions": "not specified",
            "k-core 是否使用": "not specified",
            "split protocol": "sequential recommendation supported; local README points to generated dataset scripts.",
            "negative sampling protocol": "not specified in local README",
            "evaluation candidate size": "not specified in local README",
            "是否 leave-one-out": "not specified in local README",
            "是否 multi-instance / prefix-style": "supported by task generation, but not specified as our eval protocol",
            "notes": "External repo requires generated data/checkpoints for generative scoring, so we use it only as protocol context.",
        },
        {
            "paper_name": "LLM-ESR",
            "domain/dataset": "Yelp, fashion, beauty",
            "filtering rule": "README states preprocessing filters cold-start users and items.",
            "minimum user interactions": "not specified",
            "minimum item interactions": "not specified",
            "k-core 是否使用": "cold-start filtering mentioned; exact k not specified",
            "split protocol": "Uses handled inter_seq format for sequential recommendation.",
            "negative sampling protocol": "not specified in local README",
            "evaluation candidate size": "not specified in local README",
            "是否 leave-one-out": "not specified",
            "是否 multi-instance / prefix-style": "not specified",
            "notes": "Useful evidence that ID-based sequential backbones need cold-start filtering.",
        },
        {
            "paper_name": "Project Day20-Day41 CEP validation",
            "domain/dataset": "Beauty full and small domains",
            "filtering rule": "Existing CEP experiments used leave-one-out style user histories and pointwise candidate pools.",
            "minimum user interactions": "at least 3 for train history + valid + test; Framework-Day1 compares >=4/>=5.",
            "minimum item interactions": "not forced in main CEP split",
            "k-core 是否使用": "not used for CEP observation-stage main data",
            "split protocol": "chronological leave-one-out, last item test and second last item valid",
            "negative sampling protocol": "1 positive + 5 negatives in primary continuity setting; 20neg variants available for non-trivial HR@10",
            "evaluation candidate size": "6 in continuity setting; HR@10 trivial and not primary",
            "是否 leave-one-out": "yes",
            "是否 multi-instance / prefix-style": "not for evaluation; can be derived later for LoRA training",
            "notes": "This is the closest implemented protocol to preserve continuity while making data_done cleaner.",
        },
    ]
    text_lines = [
        "# Framework-Day1 Protocol Audit",
        "",
        "This audit uses local project notes and cloned external repositories. When a local README does not specify a detail, it is marked not specified rather than inferred.",
        "",
        "| paper_name | domain/dataset | filtering rule | minimum user interactions | minimum item interactions | k-core | split protocol | negative sampling protocol | evaluation candidate size | leave-one-out | prefix-style | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        text_lines.append(
            "| "
            + " | ".join(
                str(r[k]).replace("|", "/")
                for k in [
                    "paper_name",
                    "domain/dataset",
                    "filtering rule",
                    "minimum user interactions",
                    "minimum item interactions",
                    "k-core 是否使用",
                    "split protocol",
                    "negative sampling protocol",
                    "evaluation candidate size",
                    "是否 leave-one-out",
                    "是否 multi-instance / prefix-style",
                    "notes",
                ]
            )
            + " |"
        )
    text_lines += [
        "",
        "## Setting Comparison",
        "",
        "- Setting A, user-history leave-one-out: keep users with enough chronological interactions, use the last item as test, the second last item as valid, and previous items as train history. This is the cleanest match for our user-history plus candidate-item CEP schema.",
        "- Setting B, k-core plus leave-one-out: iteratively filters users/items so both sides have enough observations. It is common in recommendation papers but can substantially alter scale and remove long-tail/cold-start behavior.",
        "- Setting C, iterative prefix / multi-instance: useful for local generator or LoRA training data, but it should be derived from train sequences rather than used as the primary evaluation split.",
        "",
        "## Recommendation",
        "",
        "Framework-Day1 uses Setting A as the first recommended data foundation: user_min4 plus chronological leave-one-out, max 10,000 users per domain, and warm negative sampling for ID-backbone compatibility. Strategy B/C statistics are still reported. Prefix-style examples should be generated later for LoRA training while preserving leave-one-out evaluation.",
    ]
    (ROOT / "framework_day1_protocol_audit.md").write_text("\n".join(text_lines), encoding="utf-8")


def _audit_raw_domains() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Counter[str]]]:
    raw_rows = []
    dist_rows = []
    user_counts_by_domain = {}
    for domain, cfg in DOMAINS.items():
        proc = cfg["processed"]
        interactions_path = proc / "interactions.csv"
        items_path = proc / "items.csv"
        users_path = proc / "users.csv"
        first_review = _read_first_json(cfg["raw_interactions"])
        first_meta = _read_first_json(cfg["raw_items"])
        user_counts, item_counts, n_rows = _count_interactions(interactions_path)
        user_counts_by_domain[domain] = user_counts
        uq = _quantiles(user_counts)
        iq = _quantiles(item_counts)
        raw_rows.append(
            {
                "domain": domain,
                "raw_interactions_path": str(cfg["raw_interactions"]),
                "raw_items_path": str(cfg["raw_items"]),
                "raw_users_path": str(users_path) if users_path.exists() else "",
                "normalized_interactions_path": str(interactions_path),
                "normalized_items_path": str(items_path),
                "num_raw_interactions": n_rows,
                "num_raw_users": len(user_counts),
                "num_raw_items": len(item_counts),
                "has_timestamp": "timestamp" in first_review,
                "has_rating": "rating" in first_review,
                "has_item_title": "title" in first_meta,
                "has_item_text": any(k in first_meta for k in ["description", "features", "categories"]),
                "user_id_field": "user_id",
                "item_id_field": cfg["review_field"],
                "timestamp_field": "timestamp",
                "rating_field": "rating",
                "item_title_field": "title",
                "item_text_field": "description/features/categories",
                "notes": "Raw JSON is retained as source of truth; Framework-Day1 processing uses the existing raw-derived normalized CSV to avoid reparsing multi-GB JSON.",
            }
        )
        dist_rows.append(
            {
                "domain": domain,
                "num_users": len(user_counts),
                "num_items": len(item_counts),
                "num_interactions": n_rows,
                "user_interactions_mean": uq["mean"],
                "user_interactions_median": uq["median"],
                "user_interactions_p50": uq["p50"],
                "user_interactions_p75": uq["p75"],
                "user_interactions_p90": uq["p90"],
                "user_interactions_p95": uq["p95"],
                "user_interactions_p99": uq["p99"],
                "num_users_ge_3": sum(1 for c in user_counts.values() if c >= 3),
                "num_users_ge_4": sum(1 for c in user_counts.values() if c >= 4),
                "num_users_ge_5": sum(1 for c in user_counts.values() if c >= 5),
                "num_users_ge_10": sum(1 for c in user_counts.values() if c >= 10),
                "item_interactions_mean": iq["mean"],
                "item_interactions_median": iq["median"],
                "item_interactions_p90": iq["p90"],
                "item_interactions_p95": iq["p95"],
                "item_interactions_p99": iq["p99"],
            }
        )
    raw_df = pd.DataFrame(raw_rows)
    dist_df = pd.DataFrame(dist_rows)
    raw_df.to_csv(ROOT / "framework_day1_raw_domain_audit.csv", index=False)
    dist_df.to_csv(ROOT / "framework_day1_interaction_distribution.csv", index=False)
    return raw_df, dist_df, user_counts_by_domain


def _strategy_comparison(user_counts_by_domain: dict[str, Counter[str]]) -> dict[str, str]:
    rows = []
    recommendations = {}
    for domain, cfg in DOMAINS.items():
        interactions_path = cfg["processed"] / "interactions.csv"
        user_counts = user_counts_by_domain[domain]
        for strategy, min_u, min_i, is_kcore in [
            ("user_min4", 4, 0, False),
            ("user_min5", 5, 0, False),
            ("5-core", 5, 5, True),
        ]:
            if is_kcore:
                users, items, rows_after = _kcore_counts(interactions_path, 5)
            else:
                users = {u for u, c in user_counts.items() if c >= min_u}
                item_counter: Counter[str] = Counter()
                rows_after = 0
                for chunk in _chunk_interactions(interactions_path):
                    chunk["user_id"] = chunk["user_id"].astype(str)
                    chunk["item_id"] = chunk["item_id"].astype(str)
                    chunk = chunk[chunk["user_id"].isin(users)]
                    rows_after += len(chunk)
                    item_counter.update(chunk["item_id"].tolist())
                items = set(item_counter)
            user_vals = [user_counts[u] for u in users if u in user_counts]
            rows.append(
                {
                    "domain": domain,
                    "strategy": strategy,
                    "min_user_interactions": min_u,
                    "min_item_interactions": min_i,
                    "is_iterative_kcore": is_kcore,
                    "users_after_filter": len(users),
                    "items_after_filter": len(items),
                    "interactions_after_filter": rows_after,
                    "avg_user_interactions": float(pd.Series(user_vals).mean()) if user_vals else math.nan,
                    "median_user_interactions": float(pd.Series(user_vals).median()) if user_vals else math.nan,
                    "avg_item_interactions": rows_after / len(items) if items else math.nan,
                    "median_item_interactions": math.nan,
                    "would_sample_to_10000": len(users) > MAX_USERS,
                    "recommended": strategy == "user_min4",
                    "reason": "Recommended first foundation: closest to leave-one-out CEP schema, preserves enough users, and avoids over-filtering long-tail items.",
                }
            )
        recommendations[domain] = "user_min4"
    pd.DataFrame(rows).to_csv(ROOT / "framework_day1_filter_strategy_comparison.csv", index=False)
    return recommendations


def _load_selected_interactions(domain: str, selected_users: set[str]) -> pd.DataFrame:
    path = DOMAINS[domain]["processed"] / "interactions.csv"
    parts = []
    for chunk in _chunk_interactions(path):
        chunk["user_id"] = chunk["user_id"].astype(str)
        chunk["item_id"] = chunk["item_id"].astype(str)
        chunk = chunk[chunk["user_id"].isin(selected_users)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])
    return pd.concat(parts, ignore_index=True)


def _build_domain(domain: str, user_counts: Counter[str], strategy: str) -> dict[str, Any]:
    rng = random.Random(SEED)
    out_dir = ROOT / domain
    _mkdir(out_dir)
    min_interactions = 4 if strategy == "user_min4" else 5
    eligible = sorted([u for u, c in user_counts.items() if c >= min_interactions])
    if len(eligible) > MAX_USERS:
        selected_users = set(rng.sample(eligible, MAX_USERS))
        users_sampled = True
    else:
        selected_users = set(eligible)
        users_sampled = False
    (out_dir / "sampled_users.json").write_text(
        json.dumps({"seed": SEED, "strategy": strategy, "users": sorted(selected_users)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    df = _load_selected_interactions(domain, selected_users)
    df = df.dropna(subset=["user_id", "item_id"]).copy()
    df["timestamp_sort"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp_sort", "item_id"], kind="mergesort")
    item_map = _load_items(DOMAINS[domain]["processed"] / "items.csv")

    sequences: dict[str, list[dict[str, Any]]] = {}
    for user_id, group in df.groupby("user_id", sort=True):
        seq = [
            {
                "item_id": str(r["item_id"]),
                "rating": None if pd.isna(r["rating"]) else float(r["rating"]),
                "timestamp": None if pd.isna(r["timestamp"]) else int(r["timestamp"]),
            }
            for r in group.to_dict("records")
        ]
        if len(seq) >= min_interactions:
            sequences[user_id] = seq

    train_rows: list[dict[str, Any]] = []
    user_sequence_rows: list[dict[str, Any]] = []
    train_history_vocab: set[str] = set()
    train_candidate_vocab: set[str] = set()
    for user_id, seq in sequences.items():
        train_seq = seq[:-2]
        train_history = [_item_obj(x["item_id"], x["timestamp"], item_map) for x in train_seq]
        train_history_vocab.update(x["item_id"] for x in train_seq)
        train_candidate_vocab.update(x["item_id"] for x in train_seq)
        train_rows.append(
            {
                "user_id": user_id,
                "history": train_history,
                "target_items": train_history,
                "domain": domain,
                "split": "train",
            }
        )
        user_sequence_rows.append(
            {
                "user_id": user_id,
                "sequence": [_item_obj(x["item_id"], x["timestamp"], item_map) for x in seq],
                "valid_positive": seq[-2]["item_id"],
                "test_positive": seq[-1]["item_id"],
                "domain": domain,
            }
        )

    train_vocab_list = sorted(train_history_vocab)
    all_filtered_items = sorted(set(df["item_id"].astype(str).tolist()))
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    neg_modes = Counter()
    for user_id, seq in sequences.items():
        train_seq = seq[:-2]
        history = [_item_obj(x["item_id"], x["timestamp"], item_map) for x in train_seq]
        user_seen = {x["item_id"] for x in seq}
        for split, pos in [("valid", seq[-2]), ("test", seq[-1])]:
            rows = valid_rows if split == "valid" else test_rows
            rows.append(_candidate_obj(user_id, history, pos["item_id"], 1, pos["timestamp"], domain, split, item_map))
            negs, mode = _sample_negatives(rng, user_seen, train_vocab_list, all_filtered_items, NEGATIVES_PER_POSITIVE)
            neg_modes[mode] += 1
            for neg in negs:
                rows.append(_candidate_obj(user_id, history, neg, 0, pos["timestamp"], domain, split, item_map))

    _write_jsonl(out_dir / "user_sequences.jsonl", user_sequence_rows)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "valid.jsonl", valid_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)
    df.drop(columns=["timestamp_sort"]).to_csv(out_dir / "interactions.csv", index=False)
    selected_item_ids = sorted(set(df["item_id"].astype(str).tolist()) | {r["candidate_item_id"] for r in valid_rows + test_rows})
    item_rows = []
    for i in selected_item_ids:
        meta = item_map.get(i, {})
        item_rows.append(
            {
                "item_id": i,
                "title": _safe_str(meta.get("title")) or f"Unknown item {i}",
                "categories": _safe_str(meta.get("categories")),
                "description": _safe_str(meta.get("description")),
                "candidate_text": _safe_str(meta.get("candidate_text")) or f"Item ID: {i}",
                "popularity_group": _safe_str(meta.get("popularity_group")),
            }
        )
    pd.DataFrame(item_rows).to_csv(out_dir / "items.csv", index=False)
    pd.DataFrame({"user_id": sorted(sequences)}).to_csv(out_dir / "users.csv", index=False)

    train_backbone_vocab = train_candidate_vocab | train_history_vocab
    cold_rows = []
    for split_name, rows in [("valid", valid_rows), ("test", test_rows)]:
        cold_rows.append(_cold_rows(split_name, rows, "train_candidate_vocab", train_candidate_vocab))
        cold_rows.append(_cold_rows(split_name, rows, "train_history_vocab", train_history_vocab))
        cold_rows.append(_cold_rows(split_name, rows, "train_backbone_vocab", train_backbone_vocab))
    pd.DataFrame(cold_rows).to_csv(out_dir / "cold_rate_diagnostics.csv", index=False)

    candidate_lengths = [len(r["history"]) for r in train_rows]
    split_stats = {
        "num_train_users": len(train_rows),
        "num_valid_users": len(set(r["user_id"] for r in valid_rows)),
        "num_test_users": len(set(r["user_id"] for r in test_rows)),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "num_items": len(selected_item_ids),
        "num_interactions": len(df),
        "avg_history_len": float(pd.Series(candidate_lengths).mean()) if candidate_lengths else math.nan,
        "median_history_len": float(pd.Series(candidate_lengths).median()) if candidate_lengths else math.nan,
        "positive_rows_valid": sum(1 for r in valid_rows if r["label"] == 1),
        "negative_rows_valid": sum(1 for r in valid_rows if r["label"] == 0),
        "positive_rows_test": sum(1 for r in test_rows if r["label"] == 1),
        "negative_rows_test": sum(1 for r in test_rows if r["label"] == 0),
        "users_sampled": users_sampled,
        "sampling_seed": SEED,
        "filter_strategy": strategy,
        "negative_sampling_mode": dict(neg_modes),
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
    }
    schema_validation = _schema_validation(valid_rows, test_rows, neg_modes)
    _write_json(out_dir / "split_stats.json", split_stats)
    _write_json(out_dir / "schema_validation.json", schema_validation)

    (out_dir / "processing_report.md").write_text(
        f"""# {domain} data_done Processing Report

- Source interactions: `{DOMAINS[domain]['raw_interactions']}` via normalized `{DOMAINS[domain]['processed'] / 'interactions.csv'}`
- Source item metadata: `{DOMAINS[domain]['raw_items']}` via normalized `{DOMAINS[domain]['processed'] / 'items.csv'}`
- Strategy: `{strategy}`
- Seed: `{SEED}`
- Users: `{len(train_rows)}`
- Valid rows: `{len(valid_rows)}`
- Test rows: `{len(test_rows)}`
- Negative sampling: warm train-vocab first, with recorded fallback modes if needed.
- Candidate pool size is 6, so HR@10 is trivial and should not be a primary metric.
""",
        encoding="utf-8",
    )
    return split_stats


def _write_framework_configs() -> None:
    cfg_dir = Path("configs/framework")
    _mkdir(cfg_dir)
    for domain in DOMAINS:
        base = f"data_done/{domain}"
        (cfg_dir / f"{domain}_relevance_evidence.yaml").write_text(
            f"""domain: {domain}
train_input_path: {base}/train.jsonl
valid_input_path: {base}/valid.jsonl
test_input_path: {base}/test.jsonl
prompt_path: prompts/candidate_relevance_evidence.txt
output_dir: output-repaired/framework/{domain}_relevance_evidence
schema: relevance_evidence
resume: true
concurrency:
  max_workers: 4
  requests_per_minute: 120
notes: Template only. Do not launch API from Framework-Day1.
""",
            encoding="utf-8",
        )
        (cfg_dir / f"{domain}_sasrec_plugin.yaml").write_text(
            f"""domain: {domain}
train_input_path: {base}/train.jsonl
valid_input_path: {base}/valid.jsonl
test_input_path: {base}/test.jsonl
backbone_name: sasrec
candidate_score_output_dir: output-repaired/framework/backbone/sasrec_{domain}
negative_sampling_mode: warm_train_vocab
seed: 42
notes: Template only. Do not train from Framework-Day1.
""",
            encoding="utf-8",
        )
        (cfg_dir / f"{domain}_lora_evidence_train.yaml").write_text(
            f"""domain: {domain}
source_train_sequences: {base}/train.jsonl
source_valid_candidates: {base}/valid.jsonl
source_test_candidates: {base}/test.jsonl
output_dir: data_done/{domain}/lora_evidence_training
schema: relevance_evidence
training_data_mode: derive_prefix_style_from_train_sequences
seed: 42
notes: Template only. LoRA data generation/training starts after Framework-Day1.
""",
            encoding="utf-8",
        )


def _write_final_report(raw_df: pd.DataFrame, dist_df: pd.DataFrame, split_stats: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Framework-Day1 Data Processing Report",
        "",
        "## 1. Motivation",
        "",
        "Observation-stage artifacts mixed old processed splits, small/medium variants, and experiment outputs. Framework-stage work needs a clean, reproducible data foundation for Qwen-LoRA, local evidence generation, ID-based backbones, and external baselines. Framework-Day1 therefore writes new artifacts under `data_done/` without overwriting `data/processed/`.",
        "",
        "## 2. Literature / Protocol Audit",
        "",
        "Local OpenP5 and LLM-ESR notes confirm that Amazon-style sequential recommendation commonly relies on preprocessed sequential data and cold-start filtering, but the visible local READMEs do not specify every filtering/evaluation detail. Our implemented project protocol is chronological leave-one-out with one valid and one test positive per user. Prefix-style examples are deferred to LoRA training data generation.",
        "",
        "## 3. Chosen Protocol",
        "",
        "The recommended first foundation is `user_min4 + chronological leave-one-out + max 10,000 users/domain + warm negative sampling`. This preserves enough history for train/valid/test, keeps scale controlled, and makes ID-based sequential backbones healthier than all-item cold sampling. We still report user_min5 and 5-core strategy comparisons.",
        "",
        "## 4. Domain Statistics",
        "",
        "| domain | raw-derived interactions | raw-derived users | raw-derived items | users >=4 | users >=5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in dist_df.iterrows():
        lines.append(
            f"| {r['domain']} | {int(r['num_interactions'])} | {int(r['num_users'])} | {int(r['num_items'])} | {int(r['num_users_ge_4'])} | {int(r['num_users_ge_5'])} |"
        )
    lines += [
        "",
        "## 5. Split Statistics",
        "",
        "| domain | train users | valid rows | test rows | items | avg history len | sampled users |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for domain, s in split_stats.items():
        lines.append(
            f"| {domain} | {s['num_train_users']} | {s['valid_rows']} | {s['test_rows']} | {s['num_items']} | {s['avg_history_len']:.2f} | {s['users_sampled']} |"
        )
    lines += [
        "",
        "## 6. Cold/Warm Diagnostics",
        "",
        "Each domain includes `cold_rate_diagnostics.csv` with valid/test cold rates under train_candidate_vocab, train_history_vocab, and train_backbone_vocab. Warm negative sampling should make negative cold rate close to zero under train_backbone_vocab. Positive cold rate may remain non-zero because chronological held-out positives can be item cold-start cases.",
        "",
        "## 7. Limitations",
        "",
        "- Beauty may have fewer than 10,000 eligible users; we do not fabricate users.",
        "- If positive cold rate is high for a domain, ID-based backbone evaluation should be marked caution.",
        "- The first split uses 1 positive + 5 negatives, so HR@10 is trivial. Primary metrics should be NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.",
        "- A 20-negative evaluation split can be generated later from the same `data_done` foundation.",
        "- Some Movies metadata rows have missing title/description; Framework-Day1 fills deterministic `Unknown item <item_id>` / `Item ID: <item_id>` placeholders so the schema remains complete for later Qwen-LoRA data generation.",
        "- Raw Amazon JSON files are huge; Framework-Day1 uses existing normalized CSVs derived from those raw files and records the raw paths explicitly.",
        "",
        "## 8. Next Steps",
        "",
        "- Framework-Day2: run baseline pipeline sanity on `data_done` without API-heavy experiments.",
        "- Framework-Day3: derive LoRA evidence-generator training pairs from train sequences.",
        "- Framework-Day4: train or connect Qwen-LoRA evidence generator on the server.",
    ]
    (ROOT / "framework_day1_data_processing_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _mkdir(ROOT)
    _write_protocol_audit()
    raw_df, dist_df, user_counts_by_domain = _audit_raw_domains()
    recommendations = _strategy_comparison(user_counts_by_domain)
    split_stats = {}
    for domain in DOMAINS:
        split_stats[domain] = _build_domain(domain, user_counts_by_domain[domain], recommendations[domain])
    _write_framework_configs()
    _write_final_report(raw_df, dist_df, split_stats)
    print("Framework-Day1 data_done processing complete.")


if __name__ == "__main__":
    main()
