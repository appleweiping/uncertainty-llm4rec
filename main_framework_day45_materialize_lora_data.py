from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


SEED = 42
DEFAULT_NEGATIVES_PER_POSITIVE = 5
DEFAULT_SAMPLE_USERS = 1000
DOMAINS = ["beauty", "books", "electronics", "movies"]


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to regenerate it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_by_user(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(_safe_str(row.get("user_id")), []).append(row)
    return out


def _item_from_history(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": _safe_str(x.get("item_id")),
        "title": _safe_str(x.get("title")),
        "text": _safe_str(x.get("text")),
        "text_missing": bool(x.get("text_missing", False)),
        "text_fallback_used": bool(x.get("text_fallback_used", False)),
        "text_source": _safe_str(x.get("text_source", "metadata")),
    }


def _candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_item_id": _safe_str(row.get("candidate_item_id")),
        "title": _safe_str(row.get("candidate_title")),
        "text": _safe_str(row.get("candidate_text")),
        "candidate_text_missing": bool(row.get("candidate_text_missing", False)),
        "candidate_text_fallback_used": bool(row.get("candidate_text_fallback_used", False)),
        "candidate_text_source": _safe_str(row.get("candidate_text_source", "metadata")),
    }


def _instruction(task: str) -> str:
    if task == "candidate_ranking":
        return (
            "You are a recommendation model. Rank only the candidate items provided in the candidate pool. "
            "Do not invent item IDs outside the candidate pool. Return JSON with ranked_item_ids."
        )
    return (
        "You are a recommendation model. Decide whether the candidate item matches the user's history. "
        "Return JSON with relevance_label. This label is not a calibrated probability."
    )


def _load_item_map(domain_dir: Path) -> dict[str, dict[str, Any]]:
    path = domain_dir / "items.csv"
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = _safe_str(row.get("item_id"))
            if not item_id:
                continue
            title = _safe_str(row.get("title")) or f"Unknown item {item_id}"
            text = _safe_str(row.get("text") or row.get("candidate_text")) or f"Item ID: {item_id}"
            fallback = str(row.get("text_fallback_used", "")).lower() == "true"
            fallback = fallback or title.startswith("Unknown item") or text.startswith("Item ID:")
            out[item_id] = {
                "title": title,
                "text": text,
                "candidate_text_missing": fallback,
                "candidate_text_fallback_used": fallback,
                "candidate_text_source": "missing_metadata_fallback" if fallback else "metadata",
            }
    return out


def _train_vocab(train_rows: list[dict[str, Any]]) -> list[str]:
    items: set[str] = set()
    for row in train_rows:
        for x in row.get("history", []):
            iid = _safe_str(x.get("item_id"))
            if iid:
                items.add(iid)
    return sorted(items)


def _select_users(domain: str, valid_rows: list[dict[str, Any]], mode: str, sample_users: int) -> list[str]:
    users = sorted(_group_by_user(valid_rows))
    if mode == "full" or domain == "beauty" or len(users) <= sample_users:
        return users
    rng = random.Random(SEED)
    return sorted(rng.sample(users, sample_users))


def _build_train_candidate_rows(
    domain: str,
    train_rows: list[dict[str, Any]],
    selected_users: set[str],
    item_map: dict[str, dict[str, Any]],
    negatives_per_positive: int,
) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    vocab = _train_vocab(train_rows)
    out: list[dict[str, Any]] = []
    for row in train_rows:
        uid = _safe_str(row.get("user_id"))
        if uid not in selected_users:
            continue
        hist = row.get("history", [])
        if len(hist) < 2:
            continue
        prefix = hist[:-1]
        target = hist[-1]
        seen = {_safe_str(x.get("item_id")) for x in hist}
        neg_pool = [iid for iid in vocab if iid not in seen]
        negs = rng.sample(neg_pool, min(len(neg_pool), negatives_per_positive))
        pos_iid = _safe_str(target.get("item_id"))
        out.append(
            {
                "user_id": uid,
                "history": prefix,
                "candidate_item_id": pos_iid,
                "candidate_title": _safe_str(target.get("title")),
                "candidate_text": _safe_str(target.get("text")),
                "candidate_text_missing": bool(target.get("text_missing", False)),
                "candidate_text_fallback_used": bool(target.get("text_fallback_used", False)),
                "candidate_text_source": _safe_str(target.get("text_source", "metadata")),
                "label": 1,
                "domain": domain,
                "split": "train",
                "candidate_pool_setting": f"{negatives_per_positive}neg",
            }
        )
        for neg in negs:
            meta = item_map.get(neg, {})
            out.append(
                {
                    "user_id": uid,
                    "history": prefix,
                    "candidate_item_id": neg,
                    "candidate_title": meta.get("title") or f"Unknown item {neg}",
                    "candidate_text": meta.get("text") or f"Item ID: {neg}",
                    "candidate_text_missing": bool(meta.get("candidate_text_missing", True)),
                    "candidate_text_fallback_used": bool(meta.get("candidate_text_fallback_used", True)),
                    "candidate_text_source": meta.get("candidate_text_source", "missing_metadata_fallback"),
                    "label": 0,
                    "domain": domain,
                    "split": "train",
                    "candidate_pool_setting": f"{negatives_per_positive}neg",
                }
            )
    return out


def _listwise_from_group(domain: str, split: str, uid: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if int(r.get("label", 0)) == 1]
    target = _safe_str(positives[0].get("candidate_item_id")) if positives else ""
    shuffled = sorted(rows, key=lambda r: _safe_str(r.get("candidate_item_id")))
    random.Random(f"{SEED}_{domain}_{split}_{uid}").shuffle(shuffled)
    candidate_pool = [_candidate_from_row(r) for r in shuffled]
    missing = sum(1 for c in candidate_pool if c["candidate_text_missing"] or c["candidate_text_fallback_used"])
    return {
        "sample_id": f"{domain}_{split}_{uid}_listwise",
        "domain": domain,
        "task": "candidate_ranking",
        "instruction": _instruction("candidate_ranking"),
        "input": {
            "user_history": [_item_from_history(x) for x in shuffled[0].get("history", [])],
            "candidate_pool": candidate_pool,
        },
        "output": {
            "ranked_item_ids": [target] + [c["candidate_item_id"] for c in candidate_pool if c["candidate_item_id"] != target],
            "target_item_id": target,
        },
        "metadata": {
            "candidate_pool_setting": shuffled[0].get("candidate_pool_setting", "5neg"),
            "text_missing_rate": missing / len(candidate_pool) if candidate_pool else 0.0,
            "source_split": split,
            "closed_catalog": True,
        },
    }


def _pointwise_from_row(domain: str, split: str, idx: int, row: dict[str, Any]) -> dict[str, Any]:
    cand = _candidate_from_row(row)
    return {
        "sample_id": f"{domain}_{split}_{idx}_pointwise",
        "domain": domain,
        "task": "candidate_relevance",
        "instruction": _instruction("candidate_relevance"),
        "input": {
            "user_history": [_item_from_history(x) for x in row.get("history", [])],
            "candidate_item": cand,
        },
        "output": {"relevance_label": int(row.get("label", 0))},
        "metadata": {
            "candidate_pool_setting": row.get("candidate_pool_setting", "5neg"),
            "text_missing": bool(cand["candidate_text_missing"]),
            "text_fallback_used": bool(cand["candidate_text_fallback_used"]),
            "source_split": split,
            "not_calibrated_probability": True,
        },
    }


def _materialize_domain(
    domain: str,
    input_root: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
    sample_users: int,
    negatives_per_positive: int,
) -> dict[str, Any]:
    domain_dir = input_root / domain
    out_dir = output_root / domain
    required = [domain_dir / "train.jsonl", domain_dir / "valid.jsonl", domain_dir / "test.jsonl"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data_done source files for {domain}: {missing}")

    train_rows = _read_jsonl(domain_dir / "train.jsonl")
    valid_rows = _read_jsonl(domain_dir / "valid.jsonl")
    test_rows = _read_jsonl(domain_dir / "test.jsonl")
    selected_users = set(_select_users(domain, valid_rows, mode, sample_users))
    item_map = _load_item_map(domain_dir)
    train_candidate_rows = _build_train_candidate_rows(
        domain, train_rows, selected_users, item_map, negatives_per_positive
    )
    split_candidate_rows = {
        "train": train_candidate_rows,
        "valid": [r for r in valid_rows if _safe_str(r.get("user_id")) in selected_users],
        "test": [r for r in test_rows if _safe_str(r.get("user_id")) in selected_users],
    }
    stats: dict[str, Any] = {
        "domain": domain,
        "seed": SEED,
        "mode": mode,
        "selected_users": len(selected_users),
        "negatives_per_positive": negatives_per_positive,
        "source_root": str(input_root),
        "output_root": str(output_root),
    }
    examples: dict[str, Any] = {}
    for split, rows in split_candidate_rows.items():
        grouped = _group_by_user(rows)
        listwise = [_listwise_from_group(domain, split, uid, group) for uid, group in sorted(grouped.items())]
        pointwise = [_pointwise_from_row(domain, split, i, row) for i, row in enumerate(rows)]
        if not listwise:
            raise ValueError(f"{domain} {split}_listwise would be empty; refusing to write silent empty training data.")
        if not pointwise:
            raise ValueError(f"{domain} {split}_pointwise would be empty; refusing to write silent empty training data.")
        _write_jsonl(out_dir / f"{split}_listwise.jsonl", listwise, overwrite=overwrite)
        _write_jsonl(out_dir / f"{split}_pointwise.jsonl", pointwise, overwrite=overwrite)
        stats[f"{split}_listwise_rows"] = len(listwise)
        stats[f"{split}_pointwise_rows"] = len(pointwise)
        stats[f"{split}_users"] = len(grouped)
        stats[f"{split}_positive_rows"] = sum(1 for r in rows if int(r.get("label", 0)) == 1)
        stats[f"{split}_negative_rows"] = sum(1 for r in rows if int(r.get("label", 0)) == 0)
        if split == "valid":
            examples["listwise"] = listwise[0]
            examples["pointwise"] = pointwise[0]

    _write_json(out_dir / "data_stats.json", stats)
    _write_json(out_dir / "prompt_examples.json", examples)
    _write_json(
        out_dir / "schema_validation.json",
        {
            "has_sample_id": True,
            "has_instruction": True,
            "has_input": True,
            "has_output": True,
            "listwise_closed_catalog": True,
            "pointwise_label_values": [0, 1],
            "calibrated_probability_used_as_label": False,
            "confidence_or_evidence_fields_used_as_label": False,
            "notes": "LoRA baseline learns raw relevance/ranking labels only. CEP calibration remains a later framework-stage operation.",
        },
    )
    return stats


def _write_report(stats_by_domain: dict[str, dict[str, Any]], output_root: Path) -> None:
    beauty = stats_by_domain.get("beauty", {})
    rows = [
        "# Framework-Day4.5 LoRA Data Materialization Report",
        "",
        "## Why Server Day4 Dry-Run Failed",
        "",
        "The server had `data_done_lora/beauty` metadata files but not the large instruction JSONL files. Those JSONL files are intentionally git-ignored and were not committed, so the Day4 dataset loader could not find `train_listwise.jsonl`.",
        "",
        "## Missing Files Repaired by Materialization",
        "",
        "- `train_listwise.jsonl`",
        "- `valid_listwise.jsonl`",
        "- `test_listwise.jsonl`",
        "- `train_pointwise.jsonl`",
        "- `valid_pointwise.jsonl`",
        "- `test_pointwise.jsonl`",
        "",
        "## Script",
        "",
        "`main_framework_day45_materialize_lora_data.py` regenerates the JSONL files from `data_done/{domain}` with seed=42. It does not use calibrated probability, confidence, evidence, or CEP fields as labels.",
        "",
        "## Beauty Expected Rows",
        "",
        f"- train listwise: `{beauty.get('train_listwise_rows', 'NA')}`",
        f"- train pointwise: `{beauty.get('train_pointwise_rows', 'NA')}`",
        f"- valid listwise: `{beauty.get('valid_listwise_rows', 'NA')}`",
        f"- valid pointwise: `{beauty.get('valid_pointwise_rows', 'NA')}`",
        f"- test listwise: `{beauty.get('test_listwise_rows', 'NA')}`",
        f"- test pointwise: `{beauty.get('test_pointwise_rows', 'NA')}`",
        "",
        "## Not Committed to GitHub",
        "",
        "The generated `data_done_lora/**/*.jsonl` files are data artifacts and remain git-ignored. Commit scripts, configs, manifests, and reports only.",
        "",
        "## Server Regeneration",
        "",
        "Run the command in `data_done/framework_day45_server_materialize_lora_data_instructions.md` after pulling the branch.",
        "",
        "## Day5 Readiness",
        "",
        "After materialization, Day5 can continue with Qwen tokenizer/model forward smoke and then a tiny LoRA train if the server model path and GPU remain available.",
        "",
    ]
    (Path("data_done") / "framework_day45_lora_data_materialization_report.md").write_text(
        "\n".join(rows), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Qwen-LoRA instruction JSONL from data_done.")
    parser.add_argument("--domain", default="beauty", choices=DOMAINS + ["all"])
    parser.add_argument("--input_root", default="data_done")
    parser.add_argument("--output_root", default="data_done_lora")
    parser.add_argument("--mode", default="full", choices=["full", "sample"])
    parser.add_argument("--sample_users", type=int, default=DEFAULT_SAMPLE_USERS)
    parser.add_argument("--negatives_per_positive", type=int, default=DEFAULT_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]
    stats_by_domain: dict[str, dict[str, Any]] = {}
    for domain in domains:
        stats_by_domain[domain] = _materialize_domain(
            domain=domain,
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            mode=args.mode,
            overwrite=args.overwrite,
            sample_users=args.sample_users,
            negatives_per_positive=args.negatives_per_positive,
        )
    _write_report(stats_by_domain, Path(args.output_root))
    print(json.dumps(stats_by_domain, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
