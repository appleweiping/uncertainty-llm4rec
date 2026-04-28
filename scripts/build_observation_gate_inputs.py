"""Build multiple no-API observation input variants for an Amazon sample gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyflow.generation import (  # noqa: E402
    CATALOG_CONSTRAINED_JSON_TEMPLATE,
    FORCED_JSON_TEMPLATE,
    RETRIEVAL_CONTEXT_JSON_TEMPLATE,
)
from storyflow.observation import (  # noqa: E402
    build_observation_input_records,
    default_observation_input_path,
    processed_dataset_dir,
    write_observation_inputs,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_records = [record for record in records if record.get("candidate_policy")]
    if not candidate_records:
        return {
            "candidate_record_count": 0,
            "target_leak_count": 0,
            "history_item_candidate_count": 0,
            "min_candidate_count": None,
            "max_candidate_count": None,
        }
    candidate_counts = [
        int(record["candidate_policy"]["candidate_count_actual"])
        for record in candidate_records
    ]
    return {
        "candidate_record_count": len(candidate_records),
        "target_leak_count": sum(
            bool(record["candidate_policy"]["target_in_candidates"])
            for record in candidate_records
        ),
        "history_item_candidate_count": sum(
            int(record["candidate_policy"]["history_item_count_in_candidates"])
            for record in candidate_records
        ),
        "min_candidate_count": min(candidate_counts),
        "max_candidate_count": max(candidate_counts),
    }


def _bucket_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        bucket = str(record["target_popularity_bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def _variant_candidate_policy(template: str) -> str:
    if template == RETRIEVAL_CONTEXT_JSON_TEMPLATE.name:
        return "history_token_overlap"
    return "round_robin_popularity"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-suffix", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--stratify-by-popularity", action="store_true")
    parser.add_argument(
        "--templates",
        default="forced_json,catalog_constrained_json,retrieval_context_json",
        help="Comma-separated prompt templates to build.",
    )
    parser.add_argument(
        "--allow-target-in-candidates",
        action="store_true",
        help="Diagnostic only. Default excludes held-out targets from candidate prompts.",
    )
    parser.add_argument("--output-manifest")
    args = parser.parse_args(argv)

    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else processed_dataset_dir(
            dataset=args.dataset,
            processed_suffix=args.processed_suffix,
            root=ROOT,
        )
    )
    if not processed_dir.exists():
        raise SystemExit(f"Processed dataset not found: {processed_dir}")

    templates = [template.strip() for template in args.templates.split(",") if template.strip()]
    allowed_templates = {
        FORCED_JSON_TEMPLATE.name,
        CATALOG_CONSTRAINED_JSON_TEMPLATE.name,
        RETRIEVAL_CONTEXT_JSON_TEMPLATE.name,
    }
    unknown_templates = sorted(set(templates) - allowed_templates)
    if unknown_templates:
        raise SystemExit(f"Unknown gate templates: {unknown_templates}")

    variant_manifests: list[dict[str, Any]] = []
    for template in templates:
        candidate_count = (
            args.candidate_count
            if template
            in {
                CATALOG_CONSTRAINED_JSON_TEMPLATE.name,
                RETRIEVAL_CONTEXT_JSON_TEMPLATE.name,
            }
            else None
        )
        candidate_policy = _variant_candidate_policy(template)
        records = build_observation_input_records(
            dataset=args.dataset,
            processed_suffix=args.processed_suffix,
            split=args.split,
            processed_dir=processed_dir,
            max_examples=args.max_examples,
            stratify_by_popularity=args.stratify_by_popularity,
            prompt_template=template,
            candidate_count=candidate_count,
            allow_target_in_candidates=args.allow_target_in_candidates,
            candidate_policy=candidate_policy,
        )
        output_jsonl = default_observation_input_path(
            dataset=args.dataset,
            processed_suffix=args.processed_suffix,
            split=args.split,
            prompt_template=template,
            candidate_count=candidate_count,
            root=ROOT,
        )
        input_manifest = write_observation_inputs(
            records,
            output_jsonl=output_jsonl,
            dataset=args.dataset,
            processed_suffix=args.processed_suffix,
            split=args.split,
            prompt_template=template,
            stratify_by_popularity=args.stratify_by_popularity,
            candidate_count=candidate_count,
            allow_target_in_candidates=args.allow_target_in_candidates,
            candidate_policy=candidate_policy,
        )
        variant_manifests.append(
            {
                "prompt_template": template,
                "candidate_policy": candidate_policy if candidate_count else None,
                "input_jsonl": input_manifest["output_jsonl"],
                "manifest_json": str(Path(input_manifest["output_jsonl"]).with_suffix(".manifest.json")),
                "input_count": input_manifest["input_count"],
                "bucket_counts": _bucket_counts(records),
                "candidate_summary": _candidate_summary(records),
                "is_experiment_result": False,
            }
        )

    output_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else ROOT
        / "outputs"
        / "observation_inputs"
        / args.dataset
        / args.processed_suffix
        / f"{args.split}_observation_gate_manifest.json"
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    gate_manifest = {
        "created_at_utc": _utc_now_iso(),
        "dataset": args.dataset,
        "processed_suffix": args.processed_suffix,
        "split": args.split,
        "max_examples": args.max_examples,
        "stratify_by_popularity": args.stratify_by_popularity,
        "candidate_count": args.candidate_count,
        "allow_target_in_candidates": args.allow_target_in_candidates,
        "processed_dir": str(processed_dir),
        "variants": variant_manifests,
        "api_called": False,
        "model_called": False,
        "is_experiment_result": False,
        "note": "Observation gate inputs only. No API, model, or paper result.",
    }
    output_manifest.write_text(
        json.dumps(gate_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(gate_manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
