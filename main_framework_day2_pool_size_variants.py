from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 42
NEG_5 = 5
NEG_20 = 20
BEAUTY_DAY9_ROWS = 5838 + 5838
ROOT = Path("data_done")
CONFIG_DIR = Path("configs/framework")
DOMAINS = ["beauty", "books", "electronics", "movies"]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _write_json(path: Path, data: Any) -> None:
    _mkdir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows) -> None:
    _mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fallback_flags(title: str, text: str) -> dict[str, Any]:
    title = _safe_str(title)
    text = _safe_str(text)
    fallback = (not text) or text.startswith("Item ID:") or title.startswith("Unknown item")
    return {
        "text_missing": bool(fallback),
        "text_fallback_used": bool(fallback),
        "text_source": "missing_metadata_fallback" if fallback else "metadata",
    }


def _candidate_flags(title: str, text: str) -> dict[str, Any]:
    flags = _fallback_flags(title, text)
    return {
        "candidate_text_missing": flags["text_missing"],
        "candidate_text_fallback_used": flags["text_fallback_used"],
        "candidate_text_source": flags["text_source"],
    }


def _enrich_history_item(item: dict[str, Any]) -> dict[str, Any]:
    item = dict(item)
    item_id = _safe_str(item.get("item_id"))
    if not _safe_str(item.get("title")):
        item["title"] = f"Unknown item {item_id}"
    if not _safe_str(item.get("text")):
        item["text"] = f"Item ID: {item_id}"
    item.update(_fallback_flags(item.get("title", ""), item.get("text", "")))
    return item


def _enrich_candidate(row: dict[str, Any], negatives: int, setting: str) -> dict[str, Any]:
    row = dict(row)
    item_id = _safe_str(row.get("candidate_item_id"))
    if not _safe_str(row.get("candidate_title")):
        row["candidate_title"] = f"Unknown item {item_id}"
    if not _safe_str(row.get("candidate_text")):
        row["candidate_text"] = f"Item ID: {item_id}"
    row["history"] = [_enrich_history_item(x) for x in row.get("history", [])]
    row.update(_candidate_flags(row.get("candidate_title", ""), row.get("candidate_text", "")))
    row["candidate_pool_setting"] = setting
    row["negatives_per_positive"] = negatives
    row.setdefault("fallback_negative_sampling_used", False)
    row.setdefault("negative_sampling_mode", "warm_train_vocab")
    return row


def _copy_replace(tmp: Path, target: Path) -> None:
    shutil.copyfile(tmp, target)
    try:
        tmp.unlink()
    except PermissionError:
        # Windows can briefly keep the just-written file handle busy. The
        # pipeline is still correct after copyfile; leftover tmp files are
        # cleaned in validation/finalization.
        pass


def _update_jsonl_in_place(path: Path, transform) -> None:
    tmp = path.with_name(path.name + ".tmp_day2")
    with path.open("r", encoding="utf-8") as inp, tmp.open("w", encoding="utf-8", newline="\n") as out:
        for line in inp:
            if not line.strip():
                continue
            row = transform(json.loads(line))
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    _copy_replace(tmp, path)


def _update_items(domain: str) -> dict[str, dict[str, Any]]:
    path = ROOT / domain / "items.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for r in df.to_dict("records"):
        item_id = _safe_str(r.get("item_id"))
        title = _safe_str(r.get("title")) or f"Unknown item {item_id}"
        text = _safe_str(r.get("candidate_text")) or f"Item ID: {item_id}"
        flags = _fallback_flags(title, text)
        r["title"] = title
        r["candidate_text"] = text
        r.update(flags)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return {
        _safe_str(r["item_id"]): {
            "title": _safe_str(r.get("title")),
            "candidate_text": _safe_str(r.get("candidate_text")),
            "text_missing": bool(r.get("text_missing")),
            "text_fallback_used": bool(r.get("text_fallback_used")),
            "text_source": _safe_str(r.get("text_source")),
        }
        for r in rows
    }


def _update_main_5neg(domain: str) -> None:
    base = ROOT / domain
    for split in ["valid", "test"]:
        _update_jsonl_in_place(base / f"{split}.jsonl", lambda row: _enrich_candidate(row, NEG_5, "5neg"))
    for name in ["train.jsonl", "user_sequences.jsonl"]:
        def transform(row: dict[str, Any]) -> dict[str, Any]:
            row = dict(row)
            if "history" in row:
                row["history"] = [_enrich_history_item(x) for x in row.get("history", [])]
            if "target_items" in row:
                row["target_items"] = [_enrich_history_item(x) for x in row.get("target_items", [])]
            if "sequence" in row:
                row["sequence"] = [_enrich_history_item(x) for x in row.get("sequence", [])]
            return row
        _update_jsonl_in_place(base / name, transform)

    stats_path = base / "split_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.update(
        {
            "candidate_pool_setting": "5neg",
            "candidates_per_user_per_split": 6,
            "hr10_trivial_flag": True,
            "recommended_primary_metrics": ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5"],
        }
    )
    _write_json(stats_path, stats)

    schema_path = base / "schema_validation.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema.update(
        {
            "candidate_pool_setting": "5neg",
            "candidates_per_user_per_split": 6,
            "candidate_pool_size_mean": 6,
            "candidate_pool_size_min": 6,
            "candidate_pool_size_max": 6,
            "hr10_trivial_flag": True,
            "recommended_primary_metrics": ["NDCG@10", "MRR", "HR@1", "HR@3", "NDCG@3", "NDCG@5"],
        }
    )
    _write_json(schema_path, schema)

    report_path = base / "processing_report.md"
    text = report_path.read_text(encoding="utf-8")
    if "Framework-Day2 pool-size annotation" not in text:
        text += (
            "\n\n## Framework-Day2 pool-size annotation\n\n"
            "- `candidate_pool_setting = 5neg`\n"
            "- `candidates_per_user_per_split = 6`\n"
            "- `hr10_trivial_flag = true`\n"
            "- Recommended primary metrics: NDCG@10, MRR, HR@1, HR@3, NDCG@3, NDCG@5.\n"
        )
        report_path.write_text(text, encoding="utf-8")


def _load_user_sequences(domain: str) -> dict[str, set[str]]:
    seen = {}
    for row in _jsonl_iter(ROOT / domain / "user_sequences.jsonl"):
        seq = row.get("sequence", [])
        seen[row["user_id"]] = {_safe_str(x.get("item_id")) for x in seq}
    return seen


def _load_positive_rows(domain: str, split: str) -> list[dict[str, Any]]:
    return [r for r in _jsonl_iter(ROOT / domain / f"{split}.jsonl") if int(r.get("label", 0)) == 1]


def _train_vocab(domain: str) -> set[str]:
    vocab = set()
    for row in _jsonl_iter(ROOT / domain / "train.jsonl"):
        for x in row.get("history", []):
            iid = _safe_str(x.get("item_id"))
            if iid:
                vocab.add(iid)
    return vocab


def _candidate_row(
    pos_row: dict[str, Any],
    item_id: str,
    label: int,
    split: str,
    item_map: dict[str, dict[str, Any]],
    fallback_used: bool,
    sampling_mode: str,
) -> dict[str, Any]:
    meta = item_map.get(item_id, {})
    row = {
        "user_id": pos_row["user_id"],
        "history": pos_row.get("history", []),
        "candidate_item_id": item_id,
        "candidate_title": meta.get("title") or f"Unknown item {item_id}",
        "candidate_text": meta.get("candidate_text") or f"Item ID: {item_id}",
        "label": int(label),
        "timestamp": pos_row.get("timestamp"),
        "domain": pos_row.get("domain"),
        "split": split,
        "candidate_pool_setting": "20neg",
        "negatives_per_positive": NEG_20,
        "fallback_negative_sampling_used": fallback_used,
        "negative_sampling_mode": sampling_mode,
    }
    return _enrich_candidate(row, NEG_20, "20neg")


def _sample_negs(rng: random.Random, user_seen: set[str], train_vocab: list[str], all_items: list[str]) -> tuple[list[str], bool, str]:
    warm_pool = [i for i in train_vocab if i not in user_seen]
    selected: list[str] = []
    fallback_used = False
    if len(warm_pool) >= NEG_20:
        return rng.sample(warm_pool, NEG_20), False, "warm_train_vocab"
    selected.extend(warm_pool)
    fallback_used = True
    remaining = NEG_20 - len(selected)
    fallback_pool = [i for i in all_items if i not in user_seen and i not in set(selected)]
    if len(fallback_pool) >= remaining:
        selected.extend(rng.sample(fallback_pool, remaining))
        return selected, fallback_used, "warm_plus_all_filtered_items_fallback"
    selected.extend(fallback_pool)
    while len(selected) < NEG_20 and fallback_pool:
        selected.append(rng.choice(fallback_pool))
    return selected[:NEG_20], fallback_used, "fallback_with_replacement"


def _schema_validation(rows_by_split: dict[str, list[dict[str, Any]]], fallback_events: Counter[str]) -> dict[str, Any]:
    all_rows = rows_by_split["valid"] + rows_by_split["test"]
    per_user_valid = Counter(r["user_id"] for r in rows_by_split["valid"])
    per_user_test = Counter(r["user_id"] for r in rows_by_split["test"])
    history_items = [x for r in all_rows for x in r.get("history", [])]
    fallback_candidates = sum(1 for r in all_rows if r.get("candidate_text_fallback_used"))
    fallback_history = sum(1 for x in history_items if x.get("text_fallback_used"))
    return {
        "candidate_pool_size_mean": float(pd.Series(list(per_user_valid.values()) + list(per_user_test.values())).mean()),
        "candidate_pool_size_min": int(min(list(per_user_valid.values()) + list(per_user_test.values()))),
        "candidate_pool_size_max": int(max(list(per_user_valid.values()) + list(per_user_test.values()))),
        "hr10_trivial_flag": False,
        "num_users": len(per_user_test),
        "valid_rows": len(rows_by_split["valid"]),
        "test_rows": len(rows_by_split["test"]),
        "positive_rows": sum(1 for r in all_rows if r["label"] == 1),
        "negative_rows": sum(1 for r in all_rows if r["label"] == 0),
        "candidate_text_missing_rate": fallback_candidates / len(all_rows) if all_rows else 0,
        "candidate_text_fallback_rate": fallback_candidates / len(all_rows) if all_rows else 0,
        "history_text_missing_rate": fallback_history / len(history_items) if history_items else 0,
        "fallback_negative_sampling_rate": fallback_events["fallback_users"] / fallback_events["total_users"] if fallback_events["total_users"] else 0,
        "candidate_pool_setting": "20neg",
        "negatives_per_positive": NEG_20,
        "seed": SEED,
    }


def _cold_diagnostics(domain: str, rows_by_split: dict[str, list[dict[str, Any]]], train_vocab: set[str], fallback_events_by_split: dict[str, Counter[str]]) -> pd.DataFrame:
    rows = []
    for split, rows_split in rows_by_split.items():
        pos = [r for r in rows_split if r["label"] == 1]
        neg = [r for r in rows_split if r["label"] == 0]
        pos_cold = [r for r in pos if r["candidate_item_id"] not in train_vocab]
        neg_cold = [r for r in neg if r["candidate_item_id"] not in train_vocab]
        all_cold = [r for r in rows_split if r["candidate_item_id"] not in train_vocab]
        events = fallback_events_by_split[split]
        rows.append(
            {
                "split": split,
                "positive_cold_rate": len(pos_cold) / len(pos) if pos else 0,
                "negative_cold_rate": len(neg_cold) / len(neg) if neg else 0,
                "all_candidate_cold_rate": len(all_cold) / len(rows_split) if rows_split else 0,
                "train_vocab_size": len(train_vocab),
                "candidate_pool_setting": "20neg",
                "negative_sampling_mode": "warm_train_vocab_first",
                "fallback_negative_sampling_rate": events["fallback_users"] / events["total_users"] if events["total_users"] else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_eval_20neg(domain: str, item_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(SEED)
    out_dir = ROOT / domain / "eval_20neg"
    _mkdir(out_dir)
    train_vocab = _train_vocab(domain)
    train_vocab_list = sorted(train_vocab)
    all_items = sorted(item_map)
    user_seen = _load_user_sequences(domain)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"valid": [], "test": []}
    fallback_events_by_split = {"valid": Counter(), "test": Counter()}
    total_events = Counter()
    for split in ["valid", "test"]:
        for pos in _load_positive_rows(domain, split):
            uid = pos["user_id"]
            negs, fallback_used, mode = _sample_negs(rng, user_seen.get(uid, set()), train_vocab_list, all_items)
            rows_by_split[split].append(
                _candidate_row(pos, pos["candidate_item_id"], 1, split, item_map, fallback_used, mode)
            )
            for neg in negs:
                rows_by_split[split].append(_candidate_row(pos, neg, 0, split, item_map, fallback_used, mode))
            fallback_events_by_split[split]["total_users"] += 1
            total_events["total_users"] += 1
            if fallback_used:
                fallback_events_by_split[split]["fallback_users"] += 1
                total_events["fallback_users"] += 1
    _write_jsonl(out_dir / "valid.jsonl", rows_by_split["valid"])
    _write_jsonl(out_dir / "test.jsonl", rows_by_split["test"])
    schema = _schema_validation(rows_by_split, total_events)
    _write_json(out_dir / "schema_validation.json", schema)
    split_stats = {
        "domain": domain,
        "candidate_pool_setting": "20neg",
        "negatives_per_positive": NEG_20,
        "candidates_per_user_per_split": 21,
        "num_users": schema["num_users"],
        "valid_rows": len(rows_by_split["valid"]),
        "test_rows": len(rows_by_split["test"]),
        "positive_rows_valid": sum(1 for r in rows_by_split["valid"] if r["label"] == 1),
        "negative_rows_valid": sum(1 for r in rows_by_split["valid"] if r["label"] == 0),
        "positive_rows_test": sum(1 for r in rows_by_split["test"] if r["label"] == 1),
        "negative_rows_test": sum(1 for r in rows_by_split["test"] if r["label"] == 0),
        "hr10_trivial_flag": False,
        "seed": SEED,
        "fallback_negative_sampling_rate": schema["fallback_negative_sampling_rate"],
    }
    _write_json(out_dir / "split_stats.json", split_stats)
    cold = _cold_diagnostics(domain, rows_by_split, train_vocab, fallback_events_by_split)
    cold.to_csv(out_dir / "cold_rate_diagnostics.csv", index=False)
    return split_stats


def _text_validation_rows(domain: str, setting: str, valid_path: Path, test_path: Path) -> list[dict[str, Any]]:
    out = []
    for split, path in [("valid", valid_path), ("test", test_path)]:
        rows = list(_jsonl_iter(path))
        hist = [x for r in rows for x in r.get("history", [])]
        cand_fb = sum(1 for r in rows if r.get("candidate_text_fallback_used") or str(r.get("candidate_text", "")).startswith("Item ID:"))
        hist_fb = sum(1 for x in hist if x.get("text_fallback_used") or str(x.get("text", "")).startswith("Item ID:"))
        cand_missing = sum(1 for r in rows if not _safe_str(r.get("candidate_text")) or r.get("candidate_text_missing"))
        hist_missing = sum(1 for x in hist if not _safe_str(x.get("text")) or x.get("text_missing"))
        status = "ok"
        notes = "metadata text available"
        if cand_fb or hist_fb:
            status = "fallback_present"
            notes = "deterministic missing-metadata fallback is present; do not treat fallback text as semantic description"
        out.append(
            {
                "domain": domain,
                "candidate_pool_setting": setting,
                "split": split,
                "candidate_text_missing_rate": cand_missing / len(rows) if rows else 0,
                "candidate_text_fallback_rate": cand_fb / len(rows) if rows else 0,
                "history_text_missing_rate": hist_missing / len(hist) if hist else 0,
                "history_text_fallback_rate": hist_fb / len(hist) if hist else 0,
                "text_coverage_status": status,
                "notes": notes,
            }
        )
    return out


def _pool_comparison_rows(domain: str, stats20: dict[str, Any]) -> list[dict[str, Any]]:
    stats5 = json.loads((ROOT / domain / "split_stats.json").read_text(encoding="utf-8"))
    users = stats5["num_test_users"]
    return [
        {
            "domain": domain,
            "candidate_pool_setting": "5neg",
            "users": users,
            "valid_rows": stats5["valid_rows"],
            "test_rows": stats5["test_rows"],
            "candidates_per_user": 6,
            "negatives_per_positive": 5,
            "hr10_trivial_flag": True,
            "estimated_api_rows_valid_test": stats5["valid_rows"] + stats5["test_rows"],
            "relative_to_beauty_day9": (stats5["valid_rows"] + stats5["test_rows"]) / BEAUTY_DAY9_ROWS,
            "recommended_usage": "low_cost_CEP_continuity, lora_training_candidate, not_for_HR10_claim",
            "notes": "Continuity split; HR@10 is trivial.",
        },
        {
            "domain": domain,
            "candidate_pool_setting": "20neg",
            "users": stats20["num_users"],
            "valid_rows": stats20["valid_rows"],
            "test_rows": stats20["test_rows"],
            "candidates_per_user": 21,
            "negatives_per_positive": 20,
            "hr10_trivial_flag": False,
            "estimated_api_rows_valid_test": stats20["valid_rows"] + stats20["test_rows"],
            "relative_to_beauty_day9": (stats20["valid_rows"] + stats20["test_rows"]) / BEAUTY_DAY9_ROWS,
            "recommended_usage": "ranking_eval, HR10_valid, high_api_cost",
            "notes": "Do not launch DeepSeek without explicit budget confirmation.",
        },
    ]


def _write_configs() -> None:
    _mkdir(CONFIG_DIR)
    for domain in DOMAINS:
        for setting, subdir, large in [("5neg", "", False), ("20neg", "/eval_20neg", True)]:
            base = f"data_done/{domain}{subdir}"
            train = f"data_done/{domain}/train.jsonl"
            (CONFIG_DIR / f"{domain}_relevance_evidence_{setting}.yaml").write_text(
                f"""domain: {domain}
candidate_pool_setting: {setting}
train_input_path: {train}
valid_input_path: {base}/valid.jsonl
test_input_path: {base}/test.jsonl
prompt_path: prompts/candidate_relevance_evidence.txt
output_dir: output-repaired/framework/{domain}_relevance_evidence_{setting}
schema: relevance_evidence
resume: true
large_api_cost: {str(large).lower()}
requires_explicit_confirmation: {str(large).lower()}
notes: Template only. Framework-Day2 does not launch API.
""",
                encoding="utf-8",
            )
            (CONFIG_DIR / f"{domain}_sasrec_plugin_{setting}.yaml").write_text(
                f"""domain: {domain}
candidate_pool_setting: {setting}
train_input_path: {train}
valid_input_path: {base}/valid.jsonl
test_input_path: {base}/test.jsonl
backbone_name: sasrec
candidate_score_output_dir: output-repaired/framework/backbone/sasrec_{domain}_{setting}
negative_sampling_mode: warm_train_vocab
large_api_cost: {str(large).lower()}
requires_explicit_confirmation: {str(large).lower()}
seed: 42
notes: Template only. Framework-Day2 does not train.
""",
                encoding="utf-8",
            )
        (CONFIG_DIR / f"{domain}_lora_evidence_train_5neg.yaml").write_text(
            f"""domain: {domain}
candidate_pool_setting: 5neg
source_train_sequences: data_done/{domain}/train.jsonl
source_valid_candidates: data_done/{domain}/valid.jsonl
source_test_candidates: data_done/{domain}/test.jsonl
output_dir: data_done/{domain}/lora_evidence_training
schema: relevance_evidence
training_data_mode: derive_prefix_style_from_train_sequences
seed: 42
notes: Template only. LoRA data generation/training starts after Framework-Day2.
""",
            encoding="utf-8",
        )


def _write_report(pool_df: pd.DataFrame, text_df: pd.DataFrame) -> None:
    def as_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_empty_"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Framework-Day2 Pool-Size and Readiness Report",
        "",
        "## 1. Day1 Recap",
        "",
        "Framework-Day1 created a clean `data_done/` foundation for Beauty, Books, Electronics, and Movies using user_min4, chronological leave-one-out, max 10,000 users/domain, seed=42, and warm negative sampling.",
        "",
        "## 2. Why Keep 5neg",
        "",
        "The original `valid.jsonl` and `test.jsonl` remain the 5neg continuity split. It is low cost, aligned with the Beauty observation-stage candidate-pool setting, and useful for LoRA/evidence-generator data scaffolding. HR@10 is trivial because each user has only six candidates, so claims should use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.",
        "",
        "## 3. Why Add 20neg",
        "",
        "`eval_20neg/` adds 20 negatives per positive, giving each user 21 candidates per split. HR@10 is no longer trivial, making this split better for formal ranking/backbone evaluation. It is not an API launch plan: Books/Electronics/Movies each have 420,000 valid+test rows in 20neg, so DeepSeek inference requires explicit budget confirmation.",
        "",
        "## 4. Four-Domain 5neg vs 20neg Cost",
        "",
        as_markdown(pool_df),
        "",
        "## 5. Text Fallback Coverage",
        "",
        as_markdown(text_df),
        "",
        "## 6. Cold-Rate Diagnostics",
        "",
        "Each `eval_20neg/cold_rate_diagnostics.csv` reports positive, negative, and all-candidate cold rates against train vocab. Warm negative sampling should keep negative cold rate near zero. Positive cold can remain high due to chronological held-out positives, so ID-based backbone results should be marked caution when positive cold rate exceeds 0.2.",
        "",
        "## 7. Recommended Next Step",
        "",
        "- Do not directly run DeepSeek on all three large-domain 20neg splits.",
        "- Framework-Day3 can use 5neg for Qwen-LoRA training data scaffold and evidence-generator pair design.",
        "- Alternatively, run Beauty `eval_20neg` local backbone ranking sanity first.",
        "- Run 20neg DeepSeek evidence only one domain at a time after explicit budget approval.",
    ]
    (ROOT / "framework_day2_pool_size_and_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    text_rows = []
    pool_rows = []
    for domain in DOMAINS:
        item_map = _update_items(domain)
        _update_main_5neg(domain)
        stats20 = _build_eval_20neg(domain, item_map)
        text_rows.extend(
            _text_validation_rows(domain, "5neg", ROOT / domain / "valid.jsonl", ROOT / domain / "test.jsonl")
        )
        text_rows.extend(
            _text_validation_rows(
                domain,
                "20neg",
                ROOT / domain / "eval_20neg" / "valid.jsonl",
                ROOT / domain / "eval_20neg" / "test.jsonl",
            )
        )
        pool_rows.extend(_pool_comparison_rows(domain, stats20))
    text_df = pd.DataFrame(text_rows)
    pool_df = pd.DataFrame(pool_rows)
    text_df.to_csv(ROOT / "framework_day2_text_fallback_validation.csv", index=False)
    pool_df.to_csv(ROOT / "framework_day2_pool_size_comparison.csv", index=False)
    _write_configs()
    _write_report(pool_df, text_df)
    print("Framework-Day2 pool-size variants complete.")


if __name__ == "__main__":
    main()
