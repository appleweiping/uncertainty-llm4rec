from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


STRICT_INSTRUCTION = (
    "You are a closed-catalog recommendation ranking model. Output JSON only. "
    "Rank all candidate_item_id values from most to least recommended. Do not explain, "
    "do not use markdown, do not output titles, do not invent IDs, and do not duplicate IDs."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_ids(row: dict[str, Any]) -> list[str]:
    return [str(x.get("candidate_item_id", "")).strip() for x in row.get("input", {}).get("candidate_pool", [])]


def _strict_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["instruction"] = STRICT_INSTRUCTION
    out["task"] = "candidate_ranking"
    out["sample_id"] = f"{row.get('sample_id', 'sample')}_json_strict"
    candidate_ids = _candidate_ids(row)
    ranked = [str(x).strip() for x in row.get("output", {}).get("ranked_item_ids", []) if str(x).strip()]
    if not ranked:
        target = str(row.get("output", {}).get("target_item_id", "")).strip()
        ranked = [target] if target else []
    seen = set()
    cleaned: list[str] = []
    for item_id in ranked + candidate_ids:
        if item_id and item_id in candidate_ids and item_id not in seen:
            cleaned.append(item_id)
            seen.add(item_id)
    out["output"] = {
        "ranked_item_ids": cleaned,
        "target_item_id": str(row.get("output", {}).get("target_item_id", cleaned[0] if cleaned else "")),
    }
    meta = dict(row.get("metadata", {}))
    meta.update(
        {
            "prompt_style": "json_strict",
            "strict_output_schema": True,
            "requires_full_candidate_ranking": True,
            "no_confidence_evidence_or_cep_targets": True,
        }
    )
    out["metadata"] = meta
    return out


def _write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def build(domain: str, input_root: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    domain_dir = output_root / domain
    stats: dict[str, Any] = {
        "domain": domain,
        "source_root": str(output_root / domain),
        "prompt_style": "json_strict",
        "splits": {},
    }
    for split in ["train", "valid", "test"]:
        src = domain_dir / f"{split}_listwise.jsonl"
        dst = domain_dir / f"{split}_listwise_json_strict.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing source listwise file: {src}")
        if dst.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to regenerate: {dst}")
        rows = [_strict_row(row) for row in _read_jsonl(src)]
        if not rows:
            raise RuntimeError(f"Refusing to write empty strict listwise split: {split}")
        _write_jsonl(dst, rows)
        pool_sizes = [len(_candidate_ids(row)) for row in rows]
        full_rank_rate = mean(
            [
                1.0
                if len(row.get("output", {}).get("ranked_item_ids", [])) == len(_candidate_ids(row))
                else 0.0
                for row in rows
            ]
        )
        stats["splits"][split] = {
            "source_file": str(src),
            "output_file": str(dst),
            "rows": len(rows),
            "candidate_pool_size_mean": mean(pool_sizes) if pool_sizes else 0,
            "candidate_pool_size_min": min(pool_sizes) if pool_sizes else 0,
            "candidate_pool_size_max": max(pool_sizes) if pool_sizes else 0,
            "full_ranking_output_rate": full_rank_rate,
        }
    flat = {
        "domain": domain,
        "train_rows": stats["splits"]["train"]["rows"],
        "valid_rows": stats["splits"]["valid"]["rows"],
        "test_rows": stats["splits"]["test"]["rows"],
        "candidate_pool_size_mean": stats["splits"]["train"]["candidate_pool_size_mean"],
        "prompt_style": "json_strict",
        "notes": "Strict listwise data aligns train and inference prompts; no CEP/confidence/evidence targets are added.",
    }
    stats["summary"] = flat
    _write_stats(input_root / "framework_day9_listwise_strict_data_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Day9 JSON-strict listwise data.")
    parser.add_argument("--domain", default="beauty")
    parser.add_argument("--input_root", default="data_done")
    parser.add_argument("--output_root", default="data_done_lora")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = build(args.domain, Path(args.input_root), Path(args.output_root), args.overwrite)
    print(json.dumps(stats["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
