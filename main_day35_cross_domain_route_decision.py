from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42
NUM_NEGATIVES = 5
SUMMARY_DIR = Path("output-repaired/summary")
DOMAINS = {
    "movies": Path("data/processed/amazon_movies_medium_5neg"),
    "books": Path("data/processed/amazon_books_medium_5neg"),
    "electronics": Path("data/processed/amazon_electronics_medium_5neg"),
}
WARM_STRICT_DIRS = {
    "movies": Path("data/processed/amazon_movies_medium_5neg_warm_strict"),
    "books": Path("data/processed/amazon_books_medium_5neg_warm_strict"),
    "electronics": Path("data/processed/amazon_electronics_medium_5neg_warm_strict"),
}
ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract_item_id(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("Item ID:"):
        return text.removeprefix("Item ID:").strip()
    match = ASIN_RE.search(text)
    return match.group(0) if match else text


def history_ids(row: dict[str, Any]) -> list[str]:
    hist = row.get("history", [])
    if not isinstance(hist, list):
        return []
    return [extract_item_id(x) for x in hist if str(x).strip()]


def load_domain(domain: str) -> dict[str, list[dict[str, Any]]]:
    base = DOMAINS[domain]
    return {split: read_jsonl(base / f"{split}.jsonl") for split in ["train", "valid", "test"]}


def build_vocabs(train_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    candidate_vocab = {extract_item_id(row["candidate_item_id"]) for row in train_rows}
    history_vocab = set()
    for row in train_rows:
        history_vocab.update(history_ids(row))
    return {
        "train_candidate_vocab": candidate_vocab,
        "train_history_vocab": history_vocab,
        "train_backbone_vocab": candidate_vocab | history_vocab,
    }


def cold_stats(domain: str, split: str, rows: list[dict[str, Any]], vocab_name: str, vocab: set[str]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    df["candidate_item_id"] = df["candidate_item_id"].map(extract_item_id)
    df["label"] = df["label"].astype(int)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    pos_cold = ~pos["candidate_item_id"].isin(vocab)
    neg_cold = ~neg["candidate_item_id"].isin(vocab)
    all_cold = ~df["candidate_item_id"].isin(vocab)
    return {
        "domain": domain,
        "split": split,
        "vocab_definition": vocab_name,
        "train_vocab_size": len(vocab),
        "num_rows": len(df),
        "num_positive_rows": len(pos),
        "num_negative_rows": len(neg),
        "positive_cold_rows": int(pos_cold.sum()),
        "positive_cold_rate": float(pos_cold.mean()) if len(pos) else 0.0,
        "negative_cold_rows": int(neg_cold.sum()),
        "negative_cold_rate": float(neg_cold.mean()) if len(neg) else 0.0,
        "all_candidate_cold_rows": int(all_cold.sum()),
        "all_candidate_cold_rate": float(all_cold.mean()) if len(df) else 0.0,
        "unique_candidate_items": df["candidate_item_id"].nunique(),
        "unique_cold_candidate_items": len(set(df["candidate_item_id"]) - vocab),
        "unique_positive_items": pos["candidate_item_id"].nunique(),
        "unique_positive_cold_items": len(set(pos["candidate_item_id"]) - vocab),
        "unique_negative_items": neg["candidate_item_id"].nunique(),
        "unique_negative_cold_items": len(set(neg["candidate_item_id"]) - vocab),
    }


def cold_rate_diagnostics(domain: str, rows_by_split: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    vocabs = build_vocabs(rows_by_split["train"])
    rows = []
    for split in ["valid", "test"]:
        for name, vocab in vocabs.items():
            rows.append(cold_stats(domain, split, rows_by_split[split], name, vocab))
    return pd.DataFrame(rows)


def positive_by_user(rows: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for row in rows:
        if int(row.get("label", 0)) == 1:
            out[str(row["user_id"])] = extract_item_id(row["candidate_item_id"])
    return out


def user_seen_positive_items(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    seen = {}
    for split_rows in rows_by_split.values():
        for row in split_rows:
            user = str(row["user_id"])
            seen.setdefault(user, set())
            if int(row.get("label", 0)) == 1:
                seen[user].add(extract_item_id(row["candidate_item_id"]))
            for hist_id in history_ids(row):
                seen[user].add(hist_id)
    return seen


def item_lookup(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for split_rows in rows_by_split.values():
        for row in split_rows:
            item = extract_item_id(row["candidate_item_id"])
            if item not in lookup or int(row.get("label", 0)) == 1:
                lookup[item] = {
                    "candidate_title": row.get("candidate_title", ""),
                    "candidate_text": row.get("candidate_text", f"Item ID: {item}"),
                    "target_popularity_group": row.get("target_popularity_group", "mid"),
                }
    return lookup


def feasibility(domain: str, rows_by_split: dict[str, list[dict[str, Any]]], build_if_possible: bool = True) -> dict[str, Any]:
    vocabs = build_vocabs(rows_by_split["train"])
    backbone_vocab = vocabs["train_backbone_vocab"]
    valid_pos = positive_by_user(rows_by_split["valid"])
    test_pos = positive_by_user(rows_by_split["test"])
    users = sorted(set(valid_pos) & set(test_pos))
    valid_warm = {u for u in users if valid_pos[u] in backbone_vocab}
    test_warm = {u for u in users if test_pos[u] in backbone_vocab}
    both_warm = sorted(valid_warm & test_warm)
    recommended = min(2000, len(both_warm)) if len(both_warm) >= 1000 else 0
    row = {
        "domain": domain,
        "eligible_users_original": len(users),
        "eligible_users_positive_warm_valid": len(valid_warm),
        "eligible_users_positive_warm_test": len(test_warm),
        "eligible_users_positive_warm_both": len(both_warm),
        "max_warm_strict_users": len(both_warm),
        "recommended_user_count": recommended,
        "valid_positive_cold_rate_after_filter": 0.0 if recommended else math.nan,
        "test_positive_cold_rate_after_filter": 0.0 if recommended else math.nan,
        "expected_negative_cold_rate": 0.0 if recommended else math.nan,
        "notes": "warm-strict feasible" if recommended else "too few users with both valid/test positives in train_backbone_vocab",
    }
    if build_if_possible and recommended >= 1000:
        build_warm_strict(domain, rows_by_split, both_warm[:recommended], vocabs["train_candidate_vocab"])
    return row


def build_warm_strict(domain: str, rows_by_split: dict[str, list[dict[str, Any]]], selected_users: list[str], train_candidate_vocab: set[str]) -> None:
    selected = set(selected_users)
    out_dir = WARM_STRICT_DIRS[domain]
    lookup = item_lookup(rows_by_split)
    seen_by_user = user_seen_positive_items(rows_by_split)
    warm_pool = sorted(train_candidate_vocab)
    rng = random.Random(SEED)
    out = {"train": [], "valid": [], "test": []}
    for row in rows_by_split["train"]:
        if str(row["user_id"]) in selected:
            out["train"].append(dict(row))
    for split in ["valid", "test"]:
        for pos_row in rows_by_split[split]:
            if str(pos_row["user_id"]) not in selected or int(pos_row.get("label", 0)) != 1:
                continue
            user = str(pos_row["user_id"])
            pos_item = extract_item_id(pos_row["candidate_item_id"])
            base = dict(pos_row)
            base["candidate_item_id"] = pos_item
            out[split].append(base)
            blocked = set(seen_by_user[user]) | {pos_item}
            candidates = [item for item in warm_pool if item not in blocked]
            sampled = rng.sample(candidates, k=min(NUM_NEGATIVES, len(candidates)))
            for item in sampled:
                info = lookup.get(item, {})
                out[split].append(
                    {
                        "user_id": user,
                        "history": pos_row.get("history", []),
                        "candidate_item_id": item,
                        "candidate_title": info.get("candidate_title", ""),
                        "candidate_text": info.get("candidate_text", f"Item ID: {item}"),
                        "label": 0,
                        "target_popularity_group": info.get("target_popularity_group", "mid"),
                        "timestamp": pos_row.get("timestamp"),
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in out.items():
        write_jsonl(split_rows, out_dir / f"{split}.jsonl")
    (out_dir / "sampled_users.json").write_text(json.dumps([{"user_id": u} for u in selected_users], indent=2), encoding="utf-8")
    stats = {
        "domain": domain,
        "source_dir": str(DOMAINS[domain]),
        "output_dir": str(out_dir),
        "seed": SEED,
        "num_negatives": NUM_NEGATIVES,
        "warm_strict_users": len(selected_users),
        "negative_sampling_pool": "train_candidate_vocab_minus_user_seen_items",
        "train_rows": len(out["train"]),
        "valid_rows": len(out["valid"]),
        "test_rows": len(out["test"]),
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    schema = validate_schema(domain, out_dir, out)
    (out_dir / "schema_validation.json").write_text(schema.to_json(orient="records", indent=2), encoding="utf-8")
    write_csv(schema, SUMMARY_DIR / f"{domain if domain != 'movies' else 'movies'}_medium_5neg_warm_strict_schema_validation.csv")
    cold = cold_rate_diagnostics(domain, out)
    write_csv(cold, SUMMARY_DIR / f"{domain if domain != 'movies' else 'movies'}_medium_5neg_warm_strict_cold_rate_diagnostics.csv")
    write_warm_strict_report(domain, out_dir, stats, cold)


def validate_schema(domain: str, out_dir: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    required = ["user_id", "history", "candidate_item_id", "candidate_text", "label"]
    for split, split_rows in rows_by_split.items():
        df = pd.DataFrame(split_rows)
        labels = df["label"].astype(int) if len(df) else pd.Series(dtype=int)
        counts = df.groupby("user_id").size() if len(df) else pd.Series(dtype=float)
        missing = [c for c in required if c not in df.columns]
        rows.append(
            {
                "domain": domain,
                "processed_path": str(out_dir),
                "split": split,
                "num_rows": len(df),
                "num_users": int(df["user_id"].nunique()) if len(df) else 0,
                "num_items": int(df["candidate_item_id"].nunique()) if len(df) else 0,
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
                "notes": "warm-strict positive and warm negative candidates",
            }
        )
    return pd.DataFrame(rows)


def write_warm_strict_report(domain: str, out_dir: Path, stats: dict[str, Any], cold: pd.DataFrame) -> None:
    main = cold[cold["vocab_definition"] == "train_backbone_vocab"]
    valid = main[main["split"] == "valid"].iloc[0]
    test = main[main["split"] == "test"].iloc[0]
    text = f"""# {domain.title()} medium_5neg_warm_strict Build Report

Warm-strict split was constructed because at least 1000 users have both valid and test positives inside train_backbone_vocab.

Output: `{out_dir}`.

Users: `{stats['warm_strict_users']}`.

Rows: train `{stats['train_rows']}`, valid `{stats['valid_rows']}`, test `{stats['test_rows']}`.

Using train_backbone_vocab:

- Valid positive cold rate `{valid['positive_cold_rate']:.4f}`, negative cold rate `{valid['negative_cold_rate']:.4f}`, all candidate cold rate `{valid['all_candidate_cold_rate']:.4f}`.
- Test positive cold rate `{test['positive_cold_rate']:.4f}`, negative cold rate `{test['negative_cold_rate']:.4f}`, all candidate cold rate `{test['all_candidate_cold_rate']:.4f}`.

This split is suitable for ID-based backbone feasibility evaluation, pending evidence inference.
"""
    (SUMMARY_DIR / f"{domain}_medium_5neg_warm_strict_build_report.md").write_text(text, encoding="utf-8")


def content_carrier_attribution() -> pd.DataFrame:
    grid = pd.read_csv(SUMMARY_DIR / "day34_movies_cold_content_carrier_plugin_rerank_grid.csv")
    rows = []
    for backbone, group in grid.groupby("backbone_name"):
        def best(method_prefix: str) -> pd.Series:
            return group[group["method"].str.startswith(method_prefix)].sort_values(["NDCG@10", "MRR"], ascending=False).iloc[0]

        a = best("A_")
        b = best("B_")
        c = best("C_")
        d = best("D_")
        gains = {
            "calibrated_relevance_posterior": b["NDCG@10"] - a["NDCG@10"],
            "evidence_risk": c["NDCG@10"] - a["NDCG@10"],
            "combined": d["NDCG@10"] - a["NDCG@10"],
        }
        if gains["combined"] > gains["calibrated_relevance_posterior"] and gains["calibrated_relevance_posterior"] > gains["evidence_risk"]:
            contributor = "calibrated_relevance_primary_evidence_risk_secondary"
        elif gains["calibrated_relevance_posterior"] >= max(gains["evidence_risk"], 0):
            contributor = "calibrated_relevance_posterior"
        else:
            contributor = "evidence_risk_or_backbone_interaction"
        rows.append(
            {
                "backbone_name": backbone,
                "A_backbone_only_NDCG": a["NDCG@10"],
                "B_plus_calibrated_relevance_NDCG": b["NDCG@10"],
                "C_plus_evidence_risk_NDCG": c["NDCG@10"],
                "D_plus_both_NDCG": d["NDCG@10"],
                "A_MRR": a["MRR"],
                "B_MRR": b["MRR"],
                "C_MRR": c["MRR"],
                "D_MRR": d["MRR"],
                "main_contributor": contributor,
                "notes": "Content carrier is a cold-aware diagnostic baseline; HR@10 is trivial because candidate pool size is 6.",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, SUMMARY_DIR / "day35_movies_content_carrier_attribution.csv")
    return out


def route_for_domain(feas: dict[str, Any], current_cold: pd.DataFrame) -> str:
    main = current_cold[current_cold["vocab_definition"] == "train_backbone_vocab"]
    valid = main[main["split"] == "valid"].iloc[0]
    test = main[main["split"] == "test"].iloc[0]
    if feas["recommended_user_count"] >= 1000:
        return "id_backbone_warm_strict"
    if max(valid["all_candidate_cold_rate"], test["all_candidate_cold_rate"]) > 0.5:
        return "content_carrier_cold"
    return "calibration_only"


def write_books_electronics_report(diag: pd.DataFrame) -> None:
    lines = []
    for domain in ["books", "electronics"]:
        main = diag[(diag["domain"] == domain) & (diag["vocab_definition"] == "train_backbone_vocab")]
        valid = main[main["split"] == "valid"].iloc[0]
        test = main[main["split"] == "test"].iloc[0]
        lines.append(
            f"- {domain}: valid pos/neg/all cold `{valid['positive_cold_rate']:.4f}` / `{valid['negative_cold_rate']:.4f}` / `{valid['all_candidate_cold_rate']:.4f}`; "
            f"test pos/neg/all cold `{test['positive_cold_rate']:.4f}` / `{test['negative_cold_rate']:.4f}` / `{test['all_candidate_cold_rate']:.4f}`."
        )
    text = f"""# Day35 Books/Electronics medium_5neg Cold-Rate Report

This report applies the same diagnostic used for Movies. It separates candidate count from cold-candidate composition and reports cold rate under train_candidate_vocab, train_history_vocab, and train_backbone_vocab.

Using train_backbone_vocab:

{chr(10).join(lines)}

Interpretation: if positive cold is low and warm-strict users are available, the domain is more suitable for ID-based backbone evaluation. If all-candidate cold is high, content-carrier cold diagnostics are safer.
"""
    (SUMMARY_DIR / "day35_books_electronics_cold_rate_report.md").write_text(text, encoding="utf-8")


def write_route_report(movies_feas: pd.DataFrame, cross_feas: pd.DataFrame, attribution: pd.DataFrame, books_elec_diag: pd.DataFrame) -> None:
    movies = movies_feas.iloc[0]
    routes = "\n".join(
        f"- {row['domain']}: route `{row['recommended_route']}`, max warm-strict users `{row['max_warm_strict_users']}`."
        for _, row in cross_feas.iterrows()
    )
    attr_lines = "\n".join(
        f"- {row['backbone_name']}: A NDCG `{row['A_backbone_only_NDCG']:.4f}`, B `{row['B_plus_calibrated_relevance_NDCG']:.4f}`, C `{row['C_plus_evidence_risk_NDCG']:.4f}`, D `{row['D_plus_both_NDCG']:.4f}`; contributor `{row['main_contributor']}`."
        for _, row in attribution.iterrows()
    )
    text = f"""# Day35 Cross-Domain Route Decision Report

## 1. Movies Cold-Rate Recap

Current Movies medium_5neg is cold-style sampling. Negative cold was driven by all-items negative sampling, while positive cold is also high due to chronological/domain long-tail effects.

## 2. Movies Content Carrier Result

Day34 showed TF-IDF/BM25 can score cold candidates without item-id embeddings. This is a cold-aware diagnostic carrier, not a SOTA backbone claim.

{attr_lines}

## 3. Movies Warm Split Limitation

Warm negative sampling reduced negative cold to zero, but positive cold remained high. Warm-strict feasibility checks whether enough users have both valid/test positives inside train_backbone_vocab.

Movies warm-strict max users: `{int(movies['max_warm_strict_users'])}`. Recommended users: `{int(movies['recommended_user_count'])}`.

## 4. Books/Electronics Cold-Rate Audit

See `day35_books_electronics_medium_5neg_cold_rate_diagnostics.csv` for full details. The route summary is:

{routes}

## 5. Recommended Day36 Route

If a domain has at least 1000 warm-strict users, use that warm-strict split for ID-based backbone + CEP. If not, use the cold-style content carrier route and keep ID-backbone claims limited to Beauty or to domains where warm-strict is feasible.

## 6. Boundary

Do not mix the settings: warm setting is for ID-based backbones; cold setting is for content carrier / cold-start diagnostics. No DeepSeek API should be launched until the chosen route is clear.
"""
    (SUMMARY_DIR / "day35_cross_domain_route_decision_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    all_diag = []
    feasibility_rows = []
    for domain in ["movies", "books", "electronics"]:
        rows = load_domain(domain)
        diag = cold_rate_diagnostics(domain, rows)
        all_diag.append(diag)
        feas = feasibility(domain, rows, build_if_possible=True)
        current_main = diag[diag["vocab_definition"] == "train_backbone_vocab"]
        valid = current_main[current_main["split"] == "valid"].iloc[0]
        test = current_main[current_main["split"] == "test"].iloc[0]
        feas.update(
            {
                "positive_cold_rate_valid": valid["positive_cold_rate"],
                "positive_cold_rate_test": test["positive_cold_rate"],
                "negative_cold_rate_current": max(valid["negative_cold_rate"], test["negative_cold_rate"]),
                "expected_negative_cold_rate_warm": 0.0 if feas["recommended_user_count"] else math.nan,
                "recommended_route": route_for_domain(feas, diag),
            }
        )
        feasibility_rows.append(feas)

    all_diag_df = pd.concat(all_diag, ignore_index=True)
    books_elec = all_diag_df[all_diag_df["domain"].isin(["books", "electronics"])]
    write_csv(books_elec, SUMMARY_DIR / "day35_books_electronics_medium_5neg_cold_rate_diagnostics.csv")
    write_books_electronics_report(books_elec)

    feas_df = pd.DataFrame(feasibility_rows)
    movies_feas = feas_df[feas_df["domain"] == "movies"]
    write_csv(
        movies_feas[
            [
                "domain",
                "eligible_users_original",
                "eligible_users_positive_warm_valid",
                "eligible_users_positive_warm_test",
                "eligible_users_positive_warm_both",
                "max_warm_strict_users",
                "recommended_user_count",
                "valid_positive_cold_rate_after_filter",
                "test_positive_cold_rate_after_filter",
                "expected_negative_cold_rate",
                "notes",
            ]
        ],
        SUMMARY_DIR / "day35_movies_warm_strict_feasibility.csv",
    )
    write_csv(
        feas_df[
            [
                "domain",
                "eligible_users_original",
                "max_warm_strict_users",
                "recommended_user_count",
                "positive_cold_rate_valid",
                "positive_cold_rate_test",
                "negative_cold_rate_current",
                "expected_negative_cold_rate_warm",
                "recommended_route",
            ]
        ],
        SUMMARY_DIR / "day35_cross_domain_warm_strict_feasibility.csv",
    )
    attribution = content_carrier_attribution()
    write_route_report(movies_feas, feas_df, attribution, books_elec)
    print("Wrote Day35 feasibility, cold-rate audit, attribution, and route decision report.")


if __name__ == "__main__":
    main()
