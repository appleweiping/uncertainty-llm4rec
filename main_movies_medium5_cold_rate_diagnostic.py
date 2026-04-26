from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path("data/processed/amazon_movies_medium_5neg")
SUMMARY_DIR = Path("output-repaired/summary")
ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_item_id(value: Any) -> str:
    text = str(value).strip()
    match = ASIN_RE.search(text)
    return match.group(0) if match else text


def history_ids(row: dict[str, Any]) -> list[str]:
    hist = row.get("history", [])
    if not isinstance(hist, list):
        return []
    return [extract_item_id(x) for x in hist if str(x).strip()]


def build_vocabs(train_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    candidate_vocab = {extract_item_id(row["candidate_item_id"]) for row in train_rows}
    history_vocab: set[str] = set()
    for row in train_rows:
        history_vocab.update(history_ids(row))
    return {
        "train_candidate_vocab": candidate_vocab,
        "train_history_vocab": history_vocab,
        "train_backbone_vocab": candidate_vocab | history_vocab,
    }


def cold_stats(split: str, rows: list[dict[str, Any]], vocab_name: str, vocab: set[str]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    df["candidate_item_id"] = df["candidate_item_id"].map(extract_item_id)
    df["label"] = df["label"].astype(int)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    pos_cold = ~pos["candidate_item_id"].isin(vocab)
    neg_cold = ~neg["candidate_item_id"].isin(vocab)
    all_cold = ~df["candidate_item_id"].isin(vocab)

    unique_candidates = set(df["candidate_item_id"])
    unique_pos = set(pos["candidate_item_id"])
    unique_neg = set(neg["candidate_item_id"])
    unique_cold = unique_candidates - vocab
    unique_pos_cold = unique_pos - vocab
    unique_neg_cold = unique_neg - vocab

    return {
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
        "unique_candidate_items": len(unique_candidates),
        "unique_cold_candidate_items": len(unique_cold),
        "unique_positive_items": len(unique_pos),
        "unique_positive_cold_items": len(unique_pos_cold),
        "unique_negative_items": len(unique_neg),
        "unique_negative_cold_items": len(unique_neg_cold),
    }


def write_report(diag: pd.DataFrame) -> None:
    main = diag[diag["vocab_definition"] == "train_backbone_vocab"].copy()
    valid = main[main["split"] == "valid"].iloc[0]
    test = main[main["split"] == "test"].iloc[0]

    def cause(row: pd.Series) -> str:
        if row["negative_cold_rate"] > 0.8 and row["positive_cold_rate"] < 0.3:
            return "mostly negative sampling from train-unseen items"
        if row["negative_cold_rate"] > 0.8 and row["positive_cold_rate"] > 0.5:
            return "both all-items negative sampling and cold future positives"
        if row["positive_cold_rate"] > 0.5:
            return "future positives are frequently train-unseen"
        return "moderate coldness"

    report = f"""# Movies medium_5neg Cold-Rate Diagnostic

## 1. Goal

This diagnostic separates candidate count from cold-candidate composition. The `5neg` setting controls how many negatives are sampled per positive. It does not by itself cause high cold rate. Cold rate is driven by train-unseen candidate coverage under the chosen sampling pool; 5neg controls candidate count, while the sampling pool determines warm/cold composition.

## 2. Train Vocabulary Definitions

The diagnostic reports three vocabularies:

- `train_candidate_vocab`: every `candidate_item_id` appearing in `train.jsonl`.
- `train_history_vocab`: item ids extracted from train `history` entries.
- `train_backbone_vocab`: union of the two. This is the main reference because ID-based backbones need item embeddings for both history and scored candidates.

## 3. Main Cold-Rate Result

Using `train_backbone_vocab`:

- Valid positive cold rate: `{valid['positive_cold_rate']:.4f}`; valid negative cold rate: `{valid['negative_cold_rate']:.4f}`; valid all-candidate cold rate: `{valid['all_candidate_cold_rate']:.4f}`.
- Test positive cold rate: `{test['positive_cold_rate']:.4f}`; test negative cold rate: `{test['negative_cold_rate']:.4f}`; test all-candidate cold rate: `{test['all_candidate_cold_rate']:.4f}`.

Interpretation:

- Valid cause: {cause(valid)}.
- Test cause: {cause(test)}.

Both negative cold rate and positive cold rate are high. This means the current Movies medium_5neg is not merely drawing cold negatives; the chronological split also places many valid/test positive items outside the train backbone vocabulary.

## 4. Why ID-Based Backbones Break

SASRec, GRU4Rec, and Bert4Rec use item-id embeddings. They can only score items that have train-time embeddings in the backbone vocabulary. When valid/test candidates are mostly train-unseen, these models must fallback or produce unreliable scores for many rows. This explains the Day32 Movies SASRec fallback problem without blaming CEP.

## 5. Setting Interpretation

The current `data/processed/amazon_movies_medium_5neg/` should be treated as a cold-style sampling setting because negatives were sampled from the regular domain all-item pool, and many future positives are also train-unseen. Do not overwrite this directory.

## 6. Recommended Split Strategy

Keep two separate Movies settings:

- `movies_medium_5neg_warm`: sample negatives from `train_seen_items - user_seen_items`. Use this for ID-based backbone plug-in evaluation with SASRec/GRU4Rec/Bert4Rec.
- `movies_medium_5neg_cold`: sample negatives from all items and allow cold candidates. Use this for TF-IDF/BM25/content carrier + CEP cold-start diagnostics.

TF-IDF/BM25 should be described as a cold-aware content carrier or diagnostic backbone, not as a SOTA recommender. It is useful here because it can score candidate_title/candidate_text without requiring train item-id embeddings.
"""
    (SUMMARY_DIR / "movies_medium_5neg_cold_rate_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    train = read_jsonl(DATA_DIR / "train.jsonl")
    valid = read_jsonl(DATA_DIR / "valid.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")
    vocabs = build_vocabs(train)

    rows = []
    for split, split_rows in [("valid", valid), ("test", test)]:
        for name, vocab in vocabs.items():
            rows.append(cold_stats(split, split_rows, name, vocab))
    diag = pd.DataFrame(rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    diag.to_csv(SUMMARY_DIR / "movies_medium_5neg_cold_rate_diagnostics.csv", index=False)
    write_report(diag)
    print("Wrote Movies medium_5neg cold-rate diagnostics and report.")


if __name__ == "__main__":
    main()
