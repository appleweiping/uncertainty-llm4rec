from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

from src.framework.safe_ranking_eval import (
    candidate_ids_from_listwise,
    positive_position_rows,
    read_jsonl,
    target_id_from_listwise,
    write_csv,
    write_jsonl,
)


OLD_ROOT = Path("data_done_lora/beauty")
NEW_ROOT = Path("data_done_lora_v2/beauty")
DIAG_CSV = Path("data_done/framework_day10_candidate_order_diagnostics.csv")
REPORT_MD = Path("data_done/framework_day10_candidate_order_diagnostics_report.md")


def _shuffle_indices(sample_id: str, n: int, seed: int) -> list[int]:
    rng = random.Random(f"{seed}:{sample_id}")
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


def _shuffle_listwise_sample(row: dict[str, Any], seed: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    sample_id = str(row.get("sample_id", "sample"))
    candidates = out.get("input", {}).get("candidate_pool", [])
    order = _shuffle_indices(sample_id, len(candidates), seed)
    shuffled = []
    position_map = []
    for new_pos, old_idx in enumerate(order, 1):
        cand = copy.deepcopy(candidates[old_idx])
        # Keep position metadata out of prompt-visible candidate objects to avoid
        # reintroducing label/order leakage in the model input.
        position_map.append(
            {
                "candidate_item_id": str(cand.get("candidate_item_id", "")).strip(),
                "original_candidate_position": old_idx + 1,
                "shuffled_candidate_position": new_pos,
                "candidate_order_seed": seed,
            }
        )
        shuffled.append(cand)
    out["input"]["candidate_pool"] = shuffled
    target = target_id_from_listwise(out)
    shuffled_ids = candidate_ids_from_listwise(out)
    ranked = [target] + [iid for iid in shuffled_ids if iid != target]
    out["output"] = {
        "ranked_item_ids": ranked,
        "target_item_id": target,
    }
    meta = dict(out.get("metadata", {}))
    meta.update({"candidate_order": "shuffled_seed42", "candidate_order_seed": seed, "candidate_position_map": position_map})
    out["metadata"] = meta
    return out


def _pointwise_from_listwise(sample: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    target = target_id_from_listwise(sample)
    history = sample.get("input", {}).get("user_history", [])
    sample_id = str(sample.get("sample_id", f"beauty_{split}_sample"))
    for idx, cand in enumerate(sample.get("input", {}).get("candidate_pool", []), 1):
        cand_copy = copy.deepcopy(cand)
        cid = str(cand_copy.get("candidate_item_id", "")).strip()
        pos_map = {
            str(x.get("candidate_item_id", "")).strip(): x
            for x in sample.get("metadata", {}).get("candidate_position_map", [])
        }
        cand_meta = pos_map.get(cid, {})
        label = 1 if cid == target else 0
        rows.append(
            {
                "sample_id": f"{sample_id}_pointwise_pos{idx}",
                "domain": sample.get("domain", "beauty"),
                "task": "candidate_relevance",
                "instruction": (
                    "You are a recommendation model. Decide whether the candidate item matches the user's history. "
                    "Return JSON with relevance_label. This label is not a calibrated probability."
                ),
                "input": {
                    "user_history": history,
                    "candidate_item": cand_copy,
                },
                "output": {"relevance_label": label},
                "metadata": {
                    "candidate_pool_setting": sample.get("metadata", {}).get("candidate_pool_setting", "5neg"),
                    "candidate_order": "shuffled_seed42",
                    "candidate_order_seed": sample.get("metadata", {}).get("candidate_order_seed", 42),
                    "original_candidate_position": cand_meta.get("original_candidate_position"),
                    "shuffled_candidate_position": cand_meta.get("shuffled_candidate_position", idx),
                    "parent_listwise_sample_id": sample_id,
                    "source_split": split,
                    "not_calibrated_probability": True,
                    "text_missing": bool(cand_copy.get("candidate_text_missing", False)),
                    "text_fallback_used": bool(cand_copy.get("candidate_text_fallback_used", False)),
                },
            }
        )
    return rows


def _pointwise_position_rows(dataset_version: str, split: str, rows: list[dict[str, Any]], pool_size: int = 6) -> list[dict[str, Any]]:
    counts = {pos: 0 for pos in range(1, pool_size + 1)}
    total = 0
    for start in range(0, len(rows), pool_size):
        group = rows[start : start + pool_size]
        if len(group) != pool_size:
            continue
        for idx, row in enumerate(group, 1):
            if int(row.get("output", {}).get("relevance_label", 0)) == 1:
                counts[idx] += 1
                total += 1
                break
    return [
        {
            "dataset_version": dataset_version,
            "split": split,
            "position": pos,
            "positive_count": counts[pos],
            "positive_rate": counts[pos] / total if total else 0.0,
            "num_users": total,
            "candidate_pool_size_mean": pool_size,
        }
        for pos in range(1, pool_size + 1)
    ]


def build(seed: int, overwrite: bool) -> dict[str, Any]:
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    stats: dict[str, Any] = {"seed": seed, "domain": "beauty", "splits": {}}
    for split in ["train", "valid", "test"]:
        src = OLD_ROOT / f"{split}_listwise_json_strict.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing strict listwise source: {src}")
        old_rows = read_jsonl(src)
        shuffled_rows = [_shuffle_listwise_sample(row, seed=seed) for row in old_rows]
        list_dst = NEW_ROOT / f"{split}_listwise_json_strict_shuffled.jsonl"
        point_dst = NEW_ROOT / f"{split}_pointwise_shuffled.jsonl"
        if (list_dst.exists() or point_dst.exists()) and not overwrite:
            raise FileExistsError(f"Outputs exist for split={split}; pass --overwrite to regenerate.")
        write_jsonl(list_dst, shuffled_rows)
        point_rows = [row for sample in shuffled_rows for row in _pointwise_from_listwise(sample, split)]
        write_jsonl(point_dst, point_rows)
        diagnostics.extend(positive_position_rows("old_listwise_json_strict", split, old_rows))
        diagnostics.extend(positive_position_rows("shuffled_listwise_seed42", split, shuffled_rows))
        old_pointwise = read_jsonl(OLD_ROOT / f"{split}_pointwise.jsonl")
        diagnostics.extend(_pointwise_position_rows("old_pointwise", split, old_pointwise))
        diagnostics.extend(_pointwise_position_rows("shuffled_pointwise_seed42", split, point_rows))
        pool_sizes = [len(candidate_ids_from_listwise(row)) for row in shuffled_rows]
        stats["splits"][split] = {
            "listwise_rows": len(shuffled_rows),
            "pointwise_rows": len(point_rows),
            "candidate_pool_size_mean": mean(pool_sizes) if pool_sizes else 0,
            "listwise_output": str(list_dst),
            "pointwise_output": str(point_dst),
        }
    write_csv(DIAG_CSV, diagnostics)
    write_report(diagnostics, stats)
    return stats


def write_report(diag_rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    def rate(version: str, split: str, pos: int) -> Any:
        row = next((r for r in diag_rows if r["dataset_version"] == version and r["split"] == split and r["position"] == pos), None)
        return row.get("positive_rate", "NA") if row else "NA"

    lines = [
        "# Framework-Day10 Candidate Order Diagnostics Report",
        "",
        "## Scope",
        "",
        "Day10 creates Beauty-only candidate-order-shuffled LoRA instruction data. This does not train, call APIs, or implement confidence/evidence/CEP framework.",
        "",
        "## Outputs",
        "",
    ]
    for split, s in stats["splits"].items():
        lines.append(f"- {split}: listwise `{s['listwise_rows']}`, pointwise `{s['pointwise_rows']}`, pool mean `{s['candidate_pool_size_mean']}`")
    lines.extend(
        [
            "",
            "## Positive Position Check",
            "",
            f"- old listwise test position-1 rate: `{rate('old_listwise_json_strict', 'test', 1)}`",
            f"- old pointwise test position-1 rate: `{rate('old_pointwise', 'test', 1)}`",
            f"- shuffled listwise test position-1 rate: `{rate('shuffled_listwise_seed42', 'test', 1)}`",
            f"- shuffled pointwise test position-1 rate: `{rate('shuffled_pointwise_seed42', 'test', 1)}`",
            "",
            "Day9.5 identified the pointwise route as order-biased because old pointwise positives are fixed at position 1. The shuffled versions should spread positives across positions 1-6 under seed 42.",
            "",
            "## Boundary",
            "",
            "The shuffled JSONL files are generated artifacts and should not be committed. Commit scripts/configs/reports only.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-Day10 shuffle Beauty LoRA candidate order.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = build(seed=args.seed, overwrite=args.overwrite)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
