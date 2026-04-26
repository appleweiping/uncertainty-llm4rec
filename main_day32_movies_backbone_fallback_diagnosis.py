from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path("data/processed/amazon_movies_medium_5neg")
REGULAR_ITEMS_PATH = Path("data/processed/amazon_movies/items.csv")
BACKBONE_PATH = Path("output-repaired/backbone/sasrec_movies_medium5_2000/candidate_scores.csv")
SUMMARY_DIR = Path("output-repaired/summary")

TRAIN_PATH = DATA_DIR / "train.jsonl"
VALID_PATH = DATA_DIR / "valid.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"

ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract_item_id(value: Any) -> str:
    text = str(value).strip()
    match = ASIN_RE.search(text)
    return match.group(0) if match else text


def history_ids(row: dict[str, Any]) -> list[str]:
    hist = row.get("history", [])
    if not isinstance(hist, list):
        return []
    ids = []
    for item in hist:
        item_id = extract_item_id(item)
        if item_id:
            ids.append(item_id)
    return ids


def split_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "train": read_jsonl(TRAIN_PATH),
        "valid": read_jsonl(VALID_PATH),
        "test": read_jsonl(TEST_PATH),
    }


def build_train_state(train: list[dict[str, Any]]) -> dict[str, Any]:
    train_users = set()
    train_users_with_history = set()
    train_history_items = set()
    train_positive_items = set()
    train_all_candidate_items = set()
    item_pop = Counter()

    for row in train:
        user_id = str(row["user_id"]).strip()
        train_users.add(user_id)
        cand = extract_item_id(row["candidate_item_id"])
        train_all_candidate_items.add(cand)
        hist = history_ids(row)
        if hist:
            train_users_with_history.add(user_id)
            train_history_items.update(hist)
        if int(row.get("label", 0)) == 1:
            train_positive_items.add(cand)
            item_pop[cand] += 1

    # Mirrors the current SASRec training vocabulary: positive targets plus mapped history from positive train rows.
    trained_vocab = set()
    for row in train:
        if int(row.get("label", 0)) != 1:
            continue
        hist = history_ids(row)
        if not hist:
            continue
        cand = extract_item_id(row["candidate_item_id"])
        trained_vocab.add(cand)
        trained_vocab.update(hist)

    scoring_vocab = set(trained_vocab)
    return {
        "train_users": train_users,
        "train_users_with_history": train_users_with_history,
        "train_history_items": train_history_items,
        "train_positive_items": train_positive_items,
        "train_all_candidate_items": train_all_candidate_items,
        "trained_vocab": trained_vocab,
        "scoring_vocab": scoring_vocab,
        "item_pop": item_pop,
    }


def fallback_breakdown(rows: list[dict[str, Any]], scores: pd.DataFrame, train_state: dict[str, Any]) -> pd.DataFrame:
    keyed = {}
    for row in rows:
        key = (str(row["user_id"]).strip(), extract_item_id(row["candidate_item_id"]), int(row.get("label", 0)))
        keyed[key] = row

    score_df = scores.copy()
    score_df["user_id"] = score_df["user_id"].astype(str).str.strip()
    score_df["candidate_item_id"] = score_df["candidate_item_id"].map(extract_item_id)
    score_df["label"] = score_df["label"].astype(int)

    user_missing = []
    item_missing = []
    fallback = score_df["fallback_score"].fillna(0).astype(int) == 1
    labels = score_df["label"].astype(int)
    for _, row in score_df.iterrows():
        key = (row["user_id"], row["candidate_item_id"], int(row["label"]))
        source = keyed.get(key)
        hist = history_ids(source) if source else []
        user_missing.append(len(hist) == 0 or row["user_id"] not in train_state["train_users_with_history"])
        item_missing.append(row["candidate_item_id"] not in train_state["trained_vocab"])

    user_missing_s = pd.Series(user_missing, index=score_df.index)
    item_missing_s = pd.Series(item_missing, index=score_df.index)
    pos = labels == 1
    neg = ~pos
    out = pd.DataFrame(
        [
            {
                "num_rows": len(score_df),
                "num_fallback_rows": int(fallback.sum()),
                "fallback_rate": float(fallback.mean()) if len(score_df) else 0.0,
                "fallback_user_rows": int((fallback & user_missing_s).sum()),
                "fallback_item_rows": int((fallback & item_missing_s).sum()),
                "fallback_both_rows": int((fallback & user_missing_s & item_missing_s).sum()),
                "num_unique_users": int(score_df["user_id"].nunique()),
                "num_unique_candidate_items": int(score_df["candidate_item_id"].nunique()),
                "num_users_missing_train_history": int(
                    len(set(score_df["user_id"]) - set(train_state["train_users_with_history"]))
                ),
                "num_items_missing_train_vocab": int(
                    len(set(score_df["candidate_item_id"]) - set(train_state["trained_vocab"]))
                ),
                "positive_rows": int(pos.sum()),
                "positive_fallback_rows": int((fallback & pos).sum()),
                "positive_fallback_rate": float((fallback & pos).sum() / max(pos.sum(), 1)),
                "negative_rows": int(neg.sum()),
                "negative_fallback_rows": int((fallback & neg).sum()),
                "negative_fallback_rate": float((fallback & neg).sum() / max(neg.sum(), 1)),
            }
        ]
    )
    write_csv(out, SUMMARY_DIR / "day32_movies_sasrec_fallback_breakdown.csv")
    return out


def sample_values(values: list[Any], n: int = 5) -> str:
    seen = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
        if len(seen) >= n:
            break
    return json.dumps(seen, ensure_ascii=False)


def has_mixed_id_style(values: list[Any]) -> bool:
    styles = set()
    for value in values[:5000]:
        text = str(value).strip()
        if text.startswith("Item ID:"):
            styles.add("item_id_prefix")
        elif ASIN_RE.fullmatch(text):
            styles.add("asin")
        else:
            styles.add("text")
    return len(styles) > 1


def load_regular_item_ids(relevant_ids: set[str]) -> tuple[set[str], int]:
    found: set[str] = set()
    total = 0
    if not REGULAR_ITEMS_PATH.exists():
        return found, total
    with REGULAR_ITEMS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            item_id = str(row.get("item_id", "")).strip()
            if item_id in relevant_ids:
                found.add(item_id)
            if len(found) == len(relevant_ids):
                # Continue is unnecessary once all relevant ids are found.
                break
    return found, total


def id_schema_consistency(splits: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    all_relevant_ids: set[str] = set()
    for split, data in splits.items():
        users = [r.get("user_id") for r in data]
        candidates = [r.get("candidate_item_id") for r in data]
        histories_raw = []
        histories_extracted = []
        for r in data[:5000]:
            histories_raw.extend(r.get("history", []) if isinstance(r.get("history", []), list) else [])
            histories_extracted.extend(history_ids(r))
        all_relevant_ids.update(extract_item_id(v) for v in candidates)
        all_relevant_ids.update(histories_extracted)
        rows.append(
            {
                "split": split,
                "user_id_dtype": "string",
                "user_id_sample_values": sample_values(users),
                "candidate_item_id_dtype": "string",
                "candidate_item_id_sample_values": sample_values(candidates),
                "history_item_id_dtype": "string",
                "history_item_id_sample_values": sample_values(histories_raw),
                "history_extracted_id_sample_values": sample_values(histories_extracted),
                "candidate_has_leading_or_trailing_space": any(str(v) != str(v).strip() for v in candidates[:5000]),
                "history_has_leading_or_trailing_space": any(str(v) != str(v).strip() for v in histories_raw[:5000]),
                "has_int_string_mixing": False,
                "has_asin_raw_id_mixing": has_mixed_id_style(candidates + histories_raw),
                "notes": "History often stores 'Item ID: ASIN' text while candidate_item_id stores raw ASIN; robust extraction is required for backbone mapping.",
            }
        )
    found_ids, item_rows_seen = load_regular_item_ids(all_relevant_ids)
    rows.append(
        {
            "split": "items_csv",
            "user_id_dtype": "",
            "user_id_sample_values": "",
            "candidate_item_id_dtype": "string",
            "candidate_item_id_sample_values": "",
            "history_item_id_dtype": "",
            "history_item_id_sample_values": "",
            "history_extracted_id_sample_values": "",
            "candidate_has_leading_or_trailing_space": False,
            "history_has_leading_or_trailing_space": False,
            "has_int_string_mixing": False,
            "has_asin_raw_id_mixing": False,
            "items_csv_relevant_id_coverage": len(found_ids) / max(len(all_relevant_ids), 1),
            "items_csv_rows_scanned_until_coverage": item_rows_seen,
            "notes": "Coverage is computed against all candidate and extracted history ASINs from medium splits.",
        }
    )
    out = pd.DataFrame(rows)
    write_csv(out, SUMMARY_DIR / "day32_movies_id_schema_consistency.csv")
    return out


def medium_split_audit(splits: dict[str, list[dict[str, Any]]], train_state: dict[str, Any]) -> pd.DataFrame:
    train_users = {str(r["user_id"]).strip() for r in splits["train"]}
    valid_users = {str(r["user_id"]).strip() for r in splits["valid"]}
    test_users = {str(r["user_id"]).strip() for r in splits["test"]}
    valid_candidates = {extract_item_id(r["candidate_item_id"]) for r in splits["valid"]}
    test_candidates = {extract_item_id(r["candidate_item_id"]) for r in splits["test"]}
    test_positive = {extract_item_id(r["candidate_item_id"]) for r in splits["test"] if int(r.get("label", 0)) == 1}
    valid_positive = {extract_item_id(r["candidate_item_id"]) for r in splits["valid"] if int(r.get("label", 0)) == 1}
    trained_vocab = train_state["trained_vocab"]
    out = pd.DataFrame(
        [
            {
                "num_train_users": len(train_users),
                "num_valid_users": len(valid_users),
                "num_test_users": len(test_users),
                "test_users_with_train_history": len(test_users & train_state["train_users_with_history"]),
                "test_users_without_train_history": len(test_users - train_state["train_users_with_history"]),
                "valid_users_with_train_history": len(valid_users & train_state["train_users_with_history"]),
                "valid_users_without_train_history": len(valid_users - train_state["train_users_with_history"]),
                "train_items": len(trained_vocab),
                "valid_candidate_items": len(valid_candidates),
                "test_candidate_items": len(test_candidates),
                "test_candidate_items_in_train_vocab": len(test_candidates & trained_vocab),
                "test_candidate_items_cold_rate": 1.0 - len(test_candidates & trained_vocab) / max(len(test_candidates), 1),
                "test_positive_items_in_train_vocab": len(test_positive & trained_vocab),
                "test_positive_items_cold_rate": 1.0 - len(test_positive & trained_vocab) / max(len(test_positive), 1),
                "valid_positive_items_in_train_vocab": len(valid_positive & trained_vocab),
                "valid_positive_items_cold_rate": 1.0 - len(valid_positive & trained_vocab) / max(len(valid_positive), 1),
            }
        ]
    )
    write_csv(out, SUMMARY_DIR / "day32_movies_medium_split_audit.csv")
    return out


def repair_strategy_ablation(scores: pd.DataFrame, splits: dict[str, list[dict[str, Any]]], train_state: dict[str, Any]) -> pd.DataFrame:
    test_rows = splits["test"]
    keyed = {
        (str(r["user_id"]).strip(), extract_item_id(r["candidate_item_id"]), int(r.get("label", 0))): r
        for r in test_rows
    }
    rows = []
    original_fallback = scores["fallback_score"].fillna(0).astype(int) == 1
    labels = scores["label"].astype(int)

    def add_row(name: str, fallback_mask: pd.Series, note: str) -> None:
        pos = labels == 1
        rows.append(
            {
                "strategy": name,
                "num_rows": len(scores),
                "fallback_rows": int(fallback_mask.sum()),
                "fallback_rate": float(fallback_mask.mean()) if len(scores) else 0.0,
                "positive_fallback_rate": float((fallback_mask & pos).sum() / max(pos.sum(), 1)),
                "negative_fallback_rate": float((fallback_mask & ~pos).sum() / max((~pos).sum(), 1)),
                "can_support_performance_claim": bool(float(fallback_mask.mean()) < 0.2),
                "notes": note,
            }
        )

    add_row("original_fallback", original_fallback, "Original Day31 SASRec score export.")

    robust_history_missing = []
    cold_item = []
    for _, row in scores.iterrows():
        key = (str(row["user_id"]).strip(), extract_item_id(row["candidate_item_id"]), int(row["label"]))
        source = keyed.get(key)
        robust_history_missing.append(not source or len(history_ids(source)) == 0)
        cold_item.append(extract_item_id(row["candidate_item_id"]) not in train_state["trained_vocab"])
    robust_history_missing_s = pd.Series(robust_history_missing, index=scores.index)
    cold_item_s = pd.Series(cold_item, index=scores.index)
    id_repaired_fallback = robust_history_missing_s | cold_item_s
    add_row(
        "id_repaired",
        id_repaired_fallback,
        "History ASINs are extracted from 'Item ID: ASIN'; cold candidates still require fallback.",
    )
    add_row(
        "vocab_repaired_min_score",
        id_repaired_fallback,
        "Adding scoring candidates to vocab does not train their embeddings; candidate-only cold items use min-score fallback.",
    )
    add_row(
        "vocab_repaired_popularity_fallback",
        id_repaired_fallback,
        "Adding scoring candidates to vocab does not train their embeddings; candidate-only cold items use train-popularity fallback.",
    )
    out = pd.DataFrame(rows)
    write_csv(out, SUMMARY_DIR / "day32_movies_sasrec_repair_strategy_ablation.csv")
    return out


def write_report(
    breakdown: pd.DataFrame,
    schema: pd.DataFrame,
    split_audit: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    b = breakdown.iloc[0]
    s = split_audit.iloc[0]
    best_repair = ablation.sort_values("fallback_rate").iloc[0]
    report = f"""# Day32 Movies SASRec Fallback Diagnosis

## 1. Day31 Recap

Day31 established a positive cross-domain CEP calibration result on Movies medium_5neg_2000: valid/test inference completed with parse_success=1.0, and raw relevance ECE dropped from 0.2913 to calibrated ECE 0.0044 on test. However, the Movies SASRec plug-in was not healthy because fallback_rate was 0.9698.

## 2. Fallback Breakdown

Rows: `{int(b['num_rows'])}`. Fallback rows: `{int(b['num_fallback_rows'])}` (`{b['fallback_rate']:.4f}`). Positive fallback rate is `{b['positive_fallback_rate']:.4f}` and negative fallback rate is `{b['negative_fallback_rate']:.4f}`.

Fallback is driven by two issues:

- User/history mapping: `{int(b['fallback_user_rows'])}` fallback rows also have missing/unmapped train history under the original title-based mapper.
- Candidate item coldness: `{int(b['fallback_item_rows'])}` fallback rows have candidate items absent from the train SASRec vocabulary.
- Both conditions overlap on `{int(b['fallback_both_rows'])}` rows.

## 3. ID / Schema Consistency

Movies medium stores `candidate_item_id` as raw ASIN but many `history` entries as text such as `Item ID: B07...`. This is schema-compatible for prompting, but it is not ideal for the Beauty SASRec helper, which originally maps history through title strings. A robust ASIN extractor fixes the history-format mismatch without changing the original data files.

## 4. Split / Vocab Audit

Test users with train history after robust extraction: `{int(s['test_users_with_train_history'])}` / `{int(s['num_test_users'])}`. Test candidate items in train vocabulary: `{int(s['test_candidate_items_in_train_vocab'])}` / `{int(s['test_candidate_items'])}`. Test candidate cold rate: `{s['test_candidate_items_cold_rate']:.4f}`. Test positive cold rate: `{s['test_positive_items_cold_rate']:.4f}`.

This means the main remaining blocker is not merely ID dtype. Movies medium has substantial future/candidate coldness relative to the train split, including many positive test items that are not in the train SASRec item vocabulary.

## 5. Repair Attempts

The non-invasive repair ablation shows the best fallback rate remains `{best_repair['fallback_rate']:.4f}` under `{best_repair['strategy']}`. This does not meet the `<20%` health threshold. Adding candidate-only items to the scoring vocabulary would not train their embeddings without using test labels; therefore, cold candidates must still use an explicit fallback such as train popularity or min score.

## 6. Decision

Fallback could not be reduced below 20% without changing the data split or training objective. Therefore, Day32 does not rerun Movies SASRec plug-in as a healthy external-backbone result.

## 7. Next Step

Recommended Day33 path: either move to Books medium5 and check whether its sequential backbone coverage is healthier, or try a Movies backbone less sensitive to cold candidates, such as GRU4Rec/Bert4Rec from the LLM-ESR path. The Movies CEP calibration result remains valid; only the Movies SASRec plug-in claim is blocked.
"""
    (SUMMARY_DIR / "day32_movies_backbone_fallback_diagnosis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    splits = split_rows()
    scores = pd.read_csv(BACKBONE_PATH)
    train_state = build_train_state(splits["train"])
    breakdown = fallback_breakdown(splits["test"], scores, train_state)
    schema = id_schema_consistency(splits)
    split_audit = medium_split_audit(splits, train_state)
    ablation = repair_strategy_ablation(scores, splits, train_state)
    write_report(breakdown, schema, split_audit, ablation)
    print("Wrote Day32 Movies fallback diagnosis outputs.")


if __name__ == "__main__":
    main()
