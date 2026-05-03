"""Offline policy sweeps for R3 OursMethod artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm4rec.analysis.ours_error_decomposition import (
    DEFAULT_SEEDS,
    FALLBACK_METHOD,
    OURS_METHOD,
    enrich_rows,
    hit_at_k,
    load_dataset_context,
    load_method_predictions,
    method_run_dir,
    mrr_at_k,
    ndcg_at_k,
    row_validity_flags,
    write_csv,
)


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    min_accept_confidence: float | None = None
    min_grounding_score: float | None = None
    enable_accept: bool = True
    enable_rerank: bool = True
    require_candidate_adherent: bool = False
    require_candidate_normalized: bool = False
    fallback_on_any_uncertainty: bool = False
    strict_default_fallback: bool = True


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _load_yaml_subset(text)


def _load_yaml_subset(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].rstrip()
        if ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def variants_from_config(config: dict[str, Any]) -> list[PolicyVariant]:
    sweep = config.get("sweep") or {}
    thresholds = sweep.get("min_accept_confidence", [0.7, 0.8, 0.85, 0.9, 0.95])
    grounding = sweep.get("min_grounding_score", [0.7, 0.8, 0.9, 1.0])
    enable_accept_values = sweep.get("enable_accept", [True, False])
    enable_rerank_values = sweep.get("enable_rerank", [True, False])
    variants: list[PolicyVariant] = [
        PolicyVariant(
            name=(
                f"conf_{confidence:g}_ground_{score:g}_"
                f"accept_{int(bool(accept))}_rerank_{int(bool(rerank))}"
            ),
            min_accept_confidence=float(confidence),
            min_grounding_score=float(score),
            enable_accept=bool(accept),
            enable_rerank=bool(rerank),
            require_candidate_adherent=False,
        )
        for confidence in thresholds
        for score in grounding
        for accept in enable_accept_values
        for rerank in enable_rerank_values
    ]
    variants.extend(
        [
            PolicyVariant(name="fallback_on_any_uncertainty", fallback_on_any_uncertainty=True, enable_accept=False, enable_rerank=False),
            PolicyVariant(name="fallback_unless_candidate_adherent", require_candidate_adherent=True),
            PolicyVariant(name="fallback_unless_candidate_adherent_conf_0.85", min_accept_confidence=0.85, require_candidate_adherent=True),
            PolicyVariant(name="fallback_unless_candidate_adherent_conf_0.9", min_accept_confidence=0.9, require_candidate_adherent=True),
            PolicyVariant(name="fallback_unless_candidate_adherent_conf_0.95", min_accept_confidence=0.95, require_candidate_adherent=True),
            PolicyVariant(
                name="fallback_unless_candidate_adherent_top_candidate_normalized",
                require_candidate_adherent=True,
                require_candidate_normalized=True,
            ),
            PolicyVariant(
                name="ours_conservative_uncertainty_gate",
                min_accept_confidence=float((config.get("conservative_policy") or {}).get("min_accept_confidence", 0.95)),
                min_grounding_score=float((config.get("conservative_policy") or {}).get("min_grounding_score", 1.0)),
                enable_accept=bool((config.get("conservative_policy") or {}).get("enable_accept", True)),
                enable_rerank=bool((config.get("conservative_policy") or {}).get("enable_rerank", False)),
                require_candidate_adherent=bool((config.get("conservative_policy") or {}).get("require_candidate_adherent", True)),
                require_candidate_normalized=bool((config.get("conservative_policy") or {}).get("require_candidate_normalized", True)),
            ),
        ]
    )
    unique: dict[str, PolicyVariant] = {}
    for variant in variants:
        unique[variant.name] = variant
    return list(unique.values())


def run_policy_sweep_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    root = config_path.resolve().parents[2] if config_path.parent.name == "experiments" else Path(".").resolve()
    runs_dir = _resolve_path(root, config.get("runs_dir", "outputs/runs"))
    output_dir = _resolve_path(root, config.get("output_dir", "outputs/tables"))
    processed_dir = _resolve_path(root, (config.get("dataset") or {}).get("processed_dir", "data/processed/movielens_1m/r2_full_single_dataset"))
    seeds = tuple(int(seed) for seed in config.get("seeds", DEFAULT_SEEDS))
    validation_seed = int(config.get("validation_seed", seeds[0]))
    confirmation_seeds = tuple(int(seed) for seed in config.get("confirmation_seeds", [seed for seed in seeds if seed != validation_seed]))
    return run_policy_sweep(
        runs_dir=runs_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        seeds=seeds,
        validation_seed=validation_seed,
        confirmation_seeds=confirmation_seeds,
        variants=variants_from_config(config),
    )


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_policy_sweep(
    *,
    runs_dir: str | Path,
    output_dir: str | Path,
    processed_dir: str | Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    validation_seed: int = 13,
    confirmation_seeds: Iterable[int] = (21, 42),
    variants: Iterable[PolicyVariant] | None = None,
) -> dict[str, Any]:
    context = load_dataset_context(processed_dir)
    variant_list = list(variants or variants_from_config({}))
    per_seed_stats: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        ours = load_method_predictions(runs_dir, OURS_METHOD, int(seed))
        fallback = load_method_predictions(runs_dir, FALLBACK_METHOD, int(seed))
        per_seed_stats[int(seed)] = summarize_seed_variants(ours, fallback, variant_list, context=context)
    rows: list[dict[str, Any]] = []
    rows.extend(_summarize_stats_group(per_seed_stats, [validation_seed], "select_seed13"))
    rows.extend(_summarize_stats_group(per_seed_stats, list(confirmation_seeds), "confirm_seeds21_42"))
    rows.extend(_summarize_stats_group(per_seed_stats, list(seeds), "all_seeds"))
    output = Path(output_dir)
    write_csv(output / "r3_ours_policy_sweep.csv", rows)
    write_markdown_table(output / "r3_ours_policy_sweep.md", rows)
    write_latex_table(output / "r3_ours_policy_sweep.tex", rows)
    best = best_conservative_policy(rows)
    return {"rows": rows, "best_conservative_policy": best, "outputs": [str(output / "r3_ours_policy_sweep.csv")]}


def summarize_seed_variants(
    ours_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    variants: list[PolicyVariant],
    *,
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    fallback_by_key = {_row_key(row): row for row in fallback_rows}
    stats = {variant.name: _empty_stats() for variant in variants}
    stats["fallback_only_reference"] = _empty_stats()
    stats["ours_full_reference"] = _empty_stats()
    for ours in ours_rows:
        fallback = fallback_by_key.get(_row_key(ours))
        if fallback is None:
            continue
        ours_summary = _source_summary(ours, context=context)
        fallback_summary = _source_summary(fallback, context=context)
        _update_stats(stats["ours_full_reference"], ours_summary, override=True)
        _update_stats(stats["fallback_only_reference"], fallback_summary, override=False)
        for variant in variants:
            use_ours = should_override(ours, variant)
            _update_stats(stats[variant.name], ours_summary if use_ours else fallback_summary, override=use_ours)
    return stats


def _source_summary(row: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    is_valid, is_hallucinated = row_validity_flags(row)
    confidence = metadata.get("confidence")
    has_confidence = confidence is not None
    confidence_value = _float_or_zero(confidence) if has_confidence else None
    if isinstance(metadata.get("is_grounded_hit"), bool):
        correct = bool(metadata["is_grounded_hit"])
    else:
        predicted = [str(item) for item in row.get("predicted_items", [])]
        correct = bool(predicted and predicted[0] == str(row.get("target_item")))
    item_buckets = context.get("item_buckets") or {}
    target_bucket = item_buckets.get(str(row.get("target_item")), "unknown")
    predicted_buckets: dict[str, int] = {}
    grounded = metadata.get("grounded_item_id")
    if grounded is not None:
        bucket = item_buckets.get(str(grounded), str(metadata.get("popularity_bucket", "unknown")))
        predicted_buckets[bucket] = predicted_buckets.get(bucket, 0) + 1
    else:
        seen: set[str] = set()
        for item in row.get("predicted_items", []):
            item = str(item)
            if item in seen:
                continue
            seen.add(item)
            bucket = item_buckets.get(item, "unknown")
            predicted_buckets[bucket] = predicted_buckets.get(bucket, 0) + 1
            if len(seen) >= 10:
                break
    return {
        "hit": hit_at_k(row),
        "mrr": mrr_at_k(row),
        "ndcg": ndcg_at_k(row),
        "valid": is_valid,
        "hallucinated": is_hallucinated,
        "has_confidence": has_confidence,
        "confidence": confidence_value,
        "correct": correct,
        "target_bucket": target_bucket,
        "predicted_buckets": predicted_buckets,
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "override_count": 0,
        "hit_sum": 0.0,
        "mrr_sum": 0.0,
        "ndcg_sum": 0.0,
        "valid_count": 0,
        "hallucinated_count": 0,
        "confidence_count": 0,
        "confidence_sum": 0.0,
        "brier_sum": 0.0,
        "high_confidence_wrong_count": 0,
        "low_confidence_correct_count": 0,
        "target_bucket_counts": {},
        "target_bucket_hit_sums": {},
        "predicted_bucket_counts": {},
        "bin_counts": [0 for _ in range(10)],
        "bin_conf_sums": [0.0 for _ in range(10)],
        "bin_correct_sums": [0.0 for _ in range(10)],
    }


def _update_stats(stats: dict[str, Any], summary: dict[str, Any], *, override: bool) -> None:
    stats["count"] += 1
    stats["override_count"] += int(override)
    stats["hit_sum"] += float(summary["hit"])
    stats["mrr_sum"] += float(summary["mrr"])
    stats["ndcg_sum"] += float(summary["ndcg"])
    stats["valid_count"] += int(summary["valid"])
    stats["hallucinated_count"] += int(summary["hallucinated"])
    bucket = str(summary["target_bucket"])
    stats["target_bucket_counts"][bucket] = stats["target_bucket_counts"].get(bucket, 0) + 1
    stats["target_bucket_hit_sums"][bucket] = stats["target_bucket_hit_sums"].get(bucket, 0.0) + float(summary["hit"])
    for pred_bucket, count in (summary["predicted_buckets"] or {}).items():
        stats["predicted_bucket_counts"][pred_bucket] = stats["predicted_bucket_counts"].get(pred_bucket, 0) + int(count)
    if summary.get("has_confidence"):
        confidence = float(summary["confidence"])
        correct = bool(summary["correct"])
        stats["confidence_count"] += 1
        stats["confidence_sum"] += confidence
        stats["brier_sum"] += (confidence - float(correct)) ** 2
        stats["high_confidence_wrong_count"] += int(confidence >= 0.85 and not correct)
        stats["low_confidence_correct_count"] += int(confidence < 0.5 and correct)
        index = min(9, int(confidence * 10))
        stats["bin_counts"][index] += 1
        stats["bin_conf_sums"][index] += confidence
        stats["bin_correct_sums"][index] += float(correct)


def _combine_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    combined = _empty_stats()
    for stats in stats_list:
        for key in [
            "count",
            "override_count",
            "hit_sum",
            "mrr_sum",
            "ndcg_sum",
            "valid_count",
            "hallucinated_count",
            "confidence_count",
            "confidence_sum",
            "brier_sum",
            "high_confidence_wrong_count",
            "low_confidence_correct_count",
        ]:
            combined[key] += stats[key]
        for key in ["target_bucket_counts", "target_bucket_hit_sums", "predicted_bucket_counts"]:
            for bucket, value in stats[key].items():
                combined[key][bucket] = combined[key].get(bucket, 0) + value
        for index in range(10):
            combined["bin_counts"][index] += stats["bin_counts"][index]
            combined["bin_conf_sums"][index] += stats["bin_conf_sums"][index]
            combined["bin_correct_sums"][index] += stats["bin_correct_sums"][index]
    return combined


def _summarize_stats_group(
    per_seed_stats: dict[int, dict[str, dict[str, Any]]],
    seeds: list[int],
    group_name: str,
) -> list[dict[str, Any]]:
    available_seeds = [seed for seed in seeds if seed in per_seed_stats]
    if not available_seeds:
        return []
    variant_names = sorted(next(iter(per_seed_stats.values())).keys())
    rows: list[dict[str, Any]] = []
    for variant_name in variant_names:
        stats = _combine_stats([per_seed_stats[seed][variant_name] for seed in available_seeds])
        count = stats["count"]
        confidence_count = stats["confidence_count"]
        rows.append(
            {
                "seed_group": group_name,
                "seeds": ",".join(str(seed) for seed in available_seeds),
                "policy": variant_name,
                "count": count,
                "override_count": stats["override_count"],
                "override_rate": stats["override_count"] / count if count else 0.0,
                "recall@10": stats["hit_sum"] / count if count else 0.0,
                "ndcg@10": stats["ndcg_sum"] / count if count else 0.0,
                "mrr@10": stats["mrr_sum"] / count if count else 0.0,
                "hit_rate@10": stats["hit_sum"] / count if count else 0.0,
                "validity_rate": stats["valid_count"] / count if count else 0.0,
                "hallucination_rate": stats["hallucinated_count"] / count if count else 0.0,
                "mean_confidence": stats["confidence_sum"] / confidence_count if confidence_count else None,
                "ece": _ece_from_bins(stats),
                "brier": stats["brier_sum"] / confidence_count if confidence_count else 0.0,
                "high_confidence_wrong_count": stats["high_confidence_wrong_count"],
                "low_confidence_correct_count": stats["low_confidence_correct_count"],
                "tail_recall@10": _bucket_recall_from_stats(stats, "tail"),
                "head_recall@10": _bucket_recall_from_stats(stats, "head"),
                "target_popularity_bucket_distribution": json.dumps(stats["target_bucket_counts"], sort_keys=True),
                "predicted_item_popularity_bucket_distribution": json.dumps(stats["predicted_bucket_counts"], sort_keys=True),
            }
        )
    return rows


def _ece_from_bins(stats: dict[str, Any]) -> float:
    total = stats["confidence_count"]
    if not total:
        return 0.0
    ece = 0.0
    for count, conf_sum, correct_sum in zip(stats["bin_counts"], stats["bin_conf_sums"], stats["bin_correct_sums"]):
        if count:
            ece += count / total * abs((correct_sum / count) - (conf_sum / count))
    return ece


def _bucket_recall_from_stats(stats: dict[str, Any], bucket: str) -> float | None:
    count = stats["target_bucket_counts"].get(bucket, 0)
    if not count:
        return None
    return stats["target_bucket_hit_sums"].get(bucket, 0.0) / count


def replay_all_variants(
    ours_rows: list[dict[str, Any]],
    fallback_rows: list[dict[str, Any]],
    variants: list[PolicyVariant],
    *,
    context: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    fallback_by_key = {_row_key(row): row for row in fallback_rows}
    outputs = {variant.name: [] for variant in variants}
    outputs["fallback_only_reference"] = []
    outputs["ours_full_reference"] = []
    for ours in ours_rows:
        fallback = fallback_by_key.get(_row_key(ours))
        if fallback is None:
            continue
        ours = _with_analysis_validity(ours)
        fallback = _with_analysis_validity(fallback)
        outputs["fallback_only_reference"].append(enrich_rows([fallback], context)[0])
        outputs["ours_full_reference"].append(enrich_rows([ours], context)[0])
        for variant in variants:
            outputs[variant.name].append(replay_policy_row(ours, fallback, variant, context=context))
    return outputs


def replay_policy_row(
    ours: dict[str, Any],
    fallback: dict[str, Any],
    variant: PolicyVariant,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    use_ours = should_override(ours, variant)
    source = ours if use_ours else fallback
    row = {
        **source,
        "method": variant.name,
        "metadata": {
            **(source.get("metadata") or {}),
            "offline_policy_name": variant.name,
            "offline_policy_source": "ours" if use_ours else "fallback",
            "offline_policy_original_decision": (ours.get("metadata") or {}).get("uncertainty_decision"),
        },
    }
    return enrich_rows([row], context)[0] if context else row


def should_override(ours: dict[str, Any], variant: PolicyVariant) -> bool:
    if variant.fallback_on_any_uncertainty:
        return False
    metadata = ours.get("metadata") or {}
    decision = str(metadata.get("uncertainty_decision") or "")
    if decision == "accept" and not variant.enable_accept:
        return False
    if decision == "rerank" and not variant.enable_rerank:
        return False
    if decision not in {"accept", "rerank"}:
        return False
    if variant.min_accept_confidence is not None and _float_or_zero(metadata.get("confidence")) < variant.min_accept_confidence:
        return False
    if variant.min_grounding_score is not None and _float_or_zero(metadata.get("grounding_score")) < variant.min_grounding_score:
        return False
    if variant.require_candidate_adherent and not _candidate_adherent(ours):
        return False
    if variant.require_candidate_normalized and not _candidate_normalized_ok(metadata):
        return False
    return True


def _candidate_adherent(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    grounded = metadata.get("grounded_item_id")
    candidates = {str(value) for value in row.get("candidate_items", [])}
    return bool(metadata.get("candidate_adherent")) and grounded is not None and str(grounded) in candidates


def _candidate_normalized_ok(metadata: dict[str, Any]) -> bool:
    confidence = metadata.get("candidate_normalized_confidence")
    if confidence is not None:
        return _float_or_zero(confidence) > 0.0
    options = metadata.get("candidate_normalized_options") or []
    return bool(options)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") or {}
    return (str(metadata.get("example_id") or ""), str(row.get("user_id")), str(row.get("target_item")))


def _with_analysis_validity(row: dict[str, Any]) -> dict[str, Any]:
    copy = dict(row)
    metadata = dict(copy.get("metadata") or {})
    is_valid, is_hallucinated = row_validity_flags(copy)
    metadata["_analysis_is_valid_prediction"] = is_valid
    metadata["_analysis_is_hallucinated_prediction"] = is_hallucinated
    copy["metadata"] = metadata
    return copy


def _summarize_group(
    per_seed_rows: dict[int, dict[str, list[dict[str, Any]]]],
    seeds: list[int],
    group_name: str,
    *,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_seeds = [seed for seed in seeds if seed in per_seed_rows]
    if not available_seeds:
        return rows
    variant_names = sorted(next(iter(per_seed_rows.values())).keys())
    for variant_name in variant_names:
        predictions = [row for seed in available_seeds for row in per_seed_rows[seed][variant_name]]
        metrics = compute_group_metrics(predictions, context=context, k=10)
        override_count = sum(1 for row in predictions if (row.get("metadata") or {}).get("offline_policy_source") == "ours")
        target_tail = _bucket_recall(predictions, bucket="tail")
        target_head = _bucket_recall(predictions, bucket="head")
        rows.append(
            {
                "seed_group": group_name,
                "seeds": ",".join(str(seed) for seed in available_seeds),
                "policy": variant_name,
                "count": metrics["count"],
                "override_count": override_count,
                "override_rate": override_count / metrics["count"] if metrics["count"] else 0.0,
                "recall@10": metrics["recall@10"],
                "ndcg@10": metrics["ndcg@10"],
                "mrr@10": metrics["mrr@10"],
                "hit_rate@10": metrics["hit_rate@10"],
                "validity_rate": metrics["validity_rate"],
                "hallucination_rate": metrics["hallucination_rate"],
                "mean_confidence": metrics["mean_confidence"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
                "high_confidence_wrong_count": metrics["high_confidence_wrong_count"],
                "low_confidence_correct_count": metrics["low_confidence_correct_count"],
                "tail_recall@10": target_tail,
                "head_recall@10": target_head,
                "target_popularity_bucket_distribution": json.dumps(
                    metrics["target_popularity_bucket_distribution"], sort_keys=True
                ),
                "predicted_item_popularity_bucket_distribution": json.dumps(
                    metrics["predicted_item_popularity_bucket_distribution"], sort_keys=True
                ),
            }
        )
    return rows


def _bucket_recall(rows: list[dict[str, Any]], *, bucket: str) -> float | None:
    subset = [row for row in rows if str((row.get("metadata") or {}).get("target_item_popularity_bucket")) == bucket]
    if not subset:
        return None
    hits = sum(1 for row in subset if str(row.get("target_item")) in [str(item) for item in row.get("predicted_items", [])[:10]])
    return hits / len(subset)


def best_conservative_policy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("seed_group") == "select_seed13"
        and (
            "conservative" in str(row.get("policy"))
            or str(row.get("policy")).startswith("fallback_unless")
            or str(row.get("policy")) == "fallback_on_any_uncertainty"
        )
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            float(row.get("recall@10") or 0.0),
            float(row.get("validity_rate") or 0.0),
            -float(row.get("hallucination_rate") or 0.0),
        ),
        reverse=True,
    )[0]


def write_markdown_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = _display_rows(rows)
    columns = ["seed_group", "policy", "override_rate", "recall@10", "ndcg@10", "mrr@10", "validity_rate", "hallucination_rate"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["seed_group", "policy", "override_rate", "recall@10", "ndcg@10", "mrr@10"]
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Seed group & Policy & Override & Recall@10 & NDCG@10 & MRR@10 \\\\",
        "\\midrule",
    ]
    for row in _display_rows(rows):
        values = [_latex_escape(_format_cell(row.get(column))) for column in columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "fallback_only_reference",
        "ours_full_reference",
        "fallback_on_any_uncertainty",
        "fallback_unless_candidate_adherent",
        "fallback_unless_candidate_adherent_conf_0.9",
        "ours_conservative_uncertainty_gate",
    }
    return [row for row in rows if row.get("policy") in keep]


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return str(value)


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")
