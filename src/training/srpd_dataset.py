from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.io import load_jsonl, save_jsonl


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        idx += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == ">":
            collected: list[str] = []
            while idx < len(lines):
                follow = lines[idx]
                follow_indent = len(follow) - len(follow.lstrip(" "))
                if follow.strip() and follow_indent <= indent:
                    break
                if follow.strip():
                    collected.append(follow.strip())
                idx += 1
            current[key] = " ".join(collected)
            continue

        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_score_summary(teacher_row: dict[str, Any]) -> dict[str, float]:
    scores = teacher_row.get("candidate_scores")
    if not isinstance(scores, dict) or not scores:
        return {
            "mean_uncertainty": 0.0,
            "mean_effective_uncertainty": 0.0,
            "mean_risk_weight": 0.0,
            "matched_uncertainty_rate": 0.0,
        }
    uncertainties: list[float] = []
    effective_uncertainties: list[float] = []
    risk_weights: list[float] = []
    matched = 0
    for score in scores.values():
        if not isinstance(score, dict):
            continue
        uncertainties.append(_as_float(score.get("uncertainty")))
        effective_uncertainties.append(_as_float(score.get("effective_uncertainty")))
        risk_weights.append(_as_float(score.get("risk_weight")))
        if bool(score.get("matched_uncertainty")):
            matched += 1
    count = max(len(uncertainties), 1)
    return {
        "mean_uncertainty": sum(uncertainties) / count,
        "mean_effective_uncertainty": sum(effective_uncertainties) / count,
        "mean_risk_weight": sum(risk_weights) / max(len(risk_weights), 1),
        "matched_uncertainty_rate": matched / count,
    }


def _load_pairwise_preferences(path: str | Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv_rows(path):
        event_id = str(row.get("source_event_id", "")).strip()
        if not event_id:
            continue
        grouped[event_id].append(
            {
                "item_a_id": row.get("item_a_id", ""),
                "item_b_id": row.get("item_b_id", ""),
                "preferred_item_pred": row.get("preferred_item_pred", ""),
                "non_preferred_item_pred": row.get("non_preferred_item_pred", ""),
                "pair_type": row.get("pair_type", ""),
                "pair_confidence": _as_float(row.get("pair_confidence")),
                "pair_uncertainty": _as_float(row.get("pair_uncertainty")),
                "pair_reliability_weight": _as_float(row.get("pair_reliability_weight"), default=1.0),
            }
        )
    return dict(grouped)


def _sample_weight(
    *,
    stage: str,
    disagreement: bool,
    uncertainty_summary: dict[str, float],
    pairwise_preferences: list[dict[str, Any]],
    weight_cfg: dict[str, Any],
) -> float:
    if stage == "v1":
        return 1.0

    base = float(weight_cfg.get("base", 1.0))
    disagreement_bonus = float(weight_cfg.get("disagreement_bonus", 0.25)) if disagreement else 0.0
    uncertainty_scale = float(weight_cfg.get("uncertainty_scale", 0.5))
    risk_scale = float(weight_cfg.get("risk_scale", 0.1))
    pairwise_scale = float(weight_cfg.get("pairwise_scale", 0.2)) if stage == "v3" else 0.0
    pairwise_strength = 0.0
    if pairwise_preferences:
        pairwise_strength = sum(float(p.get("pair_reliability_weight", 0.0)) for p in pairwise_preferences) / len(
            pairwise_preferences
        )

    if stage in {"v4", "v5", "v6"}:
        gate_cfg = weight_cfg.get("gate", {}) or {}
        gate_mode = str(gate_cfg.get("mode", "teacher_gap")).strip().lower()
        min_effective_uncertainty = float(gate_cfg.get("min_effective_uncertainty", 0.0))
        min_risk_weight = float(gate_cfg.get("min_risk_weight", 0.0))
        uncertainty_trigger = (
            uncertainty_summary["mean_effective_uncertainty"] >= min_effective_uncertainty
            and uncertainty_summary["mean_risk_weight"] >= min_risk_weight
        )
        if gate_mode == "teacher_gap_or_uncertainty":
            gate_active = disagreement or uncertainty_trigger
        elif gate_mode == "teacher_gap_and_uncertainty":
            gate_active = disagreement and uncertainty_trigger
        else:
            gate_active = disagreement

        if not gate_active:
            fallback_weight = float(gate_cfg.get("fallback_weight", base))
            min_weight = float(weight_cfg.get("min_weight", 0.5))
            max_weight = float(weight_cfg.get("max_weight", 2.0))
            return round(max(min_weight, min(max_weight, fallback_weight)), 6)

        gate_boost = float(gate_cfg.get("gate_boost", 0.0))
        weight = (
            base
            + gate_boost
            + disagreement_bonus
            + uncertainty_scale * uncertainty_summary["mean_effective_uncertainty"]
            + risk_scale * uncertainty_summary["mean_risk_weight"]
        )
        min_weight = float(weight_cfg.get("min_weight", 0.5))
        max_weight = float(weight_cfg.get("max_weight", 2.0))
        return round(max(min_weight, min(max_weight, weight)), 6)

    weight = (
        base
        + disagreement_bonus
        + uncertainty_scale * uncertainty_summary["mean_effective_uncertainty"]
        + risk_scale * uncertainty_summary["mean_risk_weight"]
        + pairwise_scale * pairwise_strength
    )
    min_weight = float(weight_cfg.get("min_weight", 0.5))
    max_weight = float(weight_cfg.get("max_weight", 2.0))
    return round(max(min_weight, min(max_weight, weight)), 6)


def _split_records(records: list[dict[str, Any]], train_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    split_idx = int(len(records) * train_ratio)
    split_idx = max(1, min(len(records) - 1, split_idx)) if len(records) > 1 else len(records)
    return records[:split_idx], records[split_idx:]


def _positive_position(ranked_item_ids: list[str], positive_item_id: str) -> int:
    if not positive_item_id:
        return len(ranked_item_ids) + 1
    try:
        return ranked_item_ids.index(positive_item_id)
    except ValueError:
        return len(ranked_item_ids) + 1


def _select_target_ranking(
    *,
    stage: str,
    teacher_ranked: list[str],
    direct_ranked: list[str],
    positive_item_id: str,
) -> tuple[list[str], str, bool]:
    if stage != "v5":
        return teacher_ranked, "teacher", False

    if not direct_ranked:
        return teacher_ranked, "teacher_fallback_no_direct", False

    teacher_pos = _positive_position(teacher_ranked, positive_item_id)
    direct_pos = _positive_position(direct_ranked, positive_item_id)
    teacher_improves_positive = teacher_pos < direct_pos
    if teacher_improves_positive:
        return teacher_ranked, "teacher_gap_positive_gain", True
    return direct_ranked, "direct_anchor_preserved", False


def _candidate_title_map(base_row: dict[str, Any]) -> dict[str, str]:
    item_ids = [str(item_id) for item_id in base_row.get("candidate_item_ids", [])]
    titles = base_row.get("candidate_titles", [])
    return {
        item_id: str(titles[idx]).strip()
        for idx, item_id in enumerate(item_ids)
        if isinstance(titles, list) and idx < len(titles)
    }


def _candidate_score(row: dict[str, Any], item_id: str, key: str, default: float = 0.0) -> float:
    scores = row.get("candidate_scores")
    if not isinstance(scores, dict):
        return default
    item_score = scores.get(item_id)
    if not isinstance(item_score, dict):
        return default
    return _as_float(item_score.get(key), default=default)


def _append_preference_pair(
    pairs: list[dict[str, Any]],
    *,
    chosen: str,
    rejected: str,
    source: str,
    teacher_ranked: list[str],
    title_by_id: dict[str, str],
    weight: float,
) -> None:
    if not chosen or not rejected or chosen == rejected:
        return
    existing = {(pair["chosen_item_id"], pair["rejected_item_id"]) for pair in pairs}
    if (chosen, rejected) in existing:
        return
    pairs.append(
        {
            "chosen_item_id": chosen,
            "rejected_item_id": rejected,
            "chosen_item_title": title_by_id.get(chosen, ""),
            "rejected_item_title": title_by_id.get(rejected, ""),
            "preference_source": source,
            "preference_weight": round(weight, 6),
            "chosen_teacher_rank": _positive_position(teacher_ranked, chosen),
            "rejected_teacher_rank": _positive_position(teacher_ranked, rejected),
        }
    )


def _build_dpo_style_preferences(
    *,
    stage: str,
    base_row: dict[str, Any],
    teacher_row: dict[str, Any],
    teacher_ranked: list[str],
    direct_ranked: list[str],
    target_ranked: list[str],
    positive_item_id: str,
    disagreement: bool,
    uncertainty_summary: dict[str, float],
    preference_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if stage != "v6" or not bool(preference_cfg.get("enabled", True)):
        return []

    max_pairs = int(preference_cfg.get("max_pairs_per_event", 4))
    if max_pairs <= 0:
        return []

    min_effective_uncertainty = float(preference_cfg.get("min_effective_uncertainty", 0.0))
    min_risk_weight = float(preference_cfg.get("min_risk_weight", 0.0))
    require_gap_or_uncertainty = bool(preference_cfg.get("require_gap_or_uncertainty", True))
    uncertainty_trigger = (
        uncertainty_summary["mean_effective_uncertainty"] >= min_effective_uncertainty
        and uncertainty_summary["mean_risk_weight"] >= min_risk_weight
    )
    if require_gap_or_uncertainty and not (disagreement or uncertainty_trigger):
        return []

    pairs: list[dict[str, Any]] = []
    title_by_id = _candidate_title_map(base_row)
    base_weight = float(preference_cfg.get("base_pair_weight", 1.0))

    if bool(preference_cfg.get("include_positive_gain_pair", True)) and positive_item_id:
        teacher_pos = _positive_position(teacher_ranked, positive_item_id)
        direct_pos = _positive_position(direct_ranked, positive_item_id)
        if teacher_pos < direct_pos and direct_ranked:
            rejected = direct_ranked[max(0, min(direct_pos - 1, len(direct_ranked) - 1))]
            _append_preference_pair(
                pairs,
                chosen=positive_item_id,
                rejected=rejected,
                source="teacher_positive_gap_gain",
                teacher_ranked=teacher_ranked,
                title_by_id=title_by_id,
                weight=base_weight + 0.2,
            )

    if bool(preference_cfg.get("include_teacher_order_pairs", True)):
        topn = min(int(preference_cfg.get("teacher_topn", 4)), len(target_ranked))
        min_score_gap = float(preference_cfg.get("min_score_gap", 0.0))
        for left_idx in range(topn):
            for right_idx in range(left_idx + 1, min(len(target_ranked), topn + 2)):
                chosen = target_ranked[left_idx]
                rejected = target_ranked[right_idx]
                chosen_score = _candidate_score(teacher_row, chosen, "final_score")
                rejected_score = _candidate_score(teacher_row, rejected, "final_score")
                if chosen_score and rejected_score and chosen_score - rejected_score < min_score_gap:
                    continue
                risk_weight = _candidate_score(teacher_row, chosen, "risk_weight")
                _append_preference_pair(
                    pairs,
                    chosen=chosen,
                    rejected=rejected,
                    source="structured_risk_teacher_order",
                    teacher_ranked=teacher_ranked,
                    title_by_id=title_by_id,
                    weight=base_weight + 0.1 * risk_weight,
                )
                if len(pairs) >= max_pairs:
                    return pairs[:max_pairs]

    return pairs[:max_pairs]


def build_srpd_rank_data(config_path: str | Path) -> dict[str, Any]:
    config = _load_yaml_config(config_path)
    stage = str(config.get("srpd_stage", "v1")).strip().lower()
    if stage not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        raise ValueError(f"Unsupported srpd_stage: {stage}")

    base_rows = load_jsonl(config["base_input_path"])
    teacher_rows = load_jsonl(config["structured_risk_teacher_path"])
    base_by_event = {str(row.get("source_event_id", "")): row for row in base_rows}
    pairwise_by_event = _load_pairwise_preferences(config.get("pairwise_preference_path"))

    weight_cfg = config.get("weighting", {}) or {}
    max_pairwise_preferences = int(config.get("max_pairwise_preferences_per_event", 12))
    records: list[dict[str, Any]] = []
    missing_base_count = 0

    for teacher_row in teacher_rows:
        event_id = str(teacher_row.get("source_event_id", "")).strip()
        base_row = base_by_event.get(event_id)
        if base_row is None:
            missing_base_count += 1
            continue

        teacher_ranked = [
            str(item_id)
            for item_id in teacher_row.get("pred_ranked_item_ids", teacher_row.get("topk_item_ids", []))
        ]
        direct_ranked = [str(item_id) for item_id in teacher_row.get("original_pred_ranked_item_ids", [])]
        if not teacher_ranked:
            continue

        pairwise_preferences = pairwise_by_event.get(event_id, [])[:max_pairwise_preferences]
        uncertainty_summary = _candidate_score_summary(teacher_row)
        disagreement = bool(teacher_ranked != direct_ranked)
        positive_item_id = str(base_row.get("positive_item_id", "")).strip()
        target_ranked, target_source, teacher_improves_positive = _select_target_ranking(
            stage=stage,
            teacher_ranked=teacher_ranked,
            direct_ranked=direct_ranked,
            positive_item_id=positive_item_id,
        )
        dpo_style_preferences = _build_dpo_style_preferences(
            stage=stage,
            base_row=base_row,
            teacher_row=teacher_row,
            teacher_ranked=teacher_ranked,
            direct_ranked=direct_ranked,
            target_ranked=target_ranked,
            positive_item_id=positive_item_id,
            disagreement=disagreement,
            uncertainty_summary=uncertainty_summary,
            preference_cfg=config.get("preference_training", {}) or {},
        )
        teacher_positive_position = _positive_position(teacher_ranked, positive_item_id)
        direct_positive_position = _positive_position(direct_ranked, positive_item_id)
        target_positive_position = _positive_position(target_ranked, positive_item_id)
        record = dict(base_row)
        record.update(
            {
                "teacher_ranked_item_ids": teacher_ranked,
                "target_ranked_item_ids": target_ranked,
                "srpd_teacher_ranked_item_ids": teacher_ranked,
                "srpd_target_ranked_item_ids": target_ranked,
                "srpd_direct_ranked_item_ids": direct_ranked,
                "srpd_stage": stage,
                "srpd_method_family": "SRPD",
                "srpd_method_variant": str(config.get("method_variant", f"srpd_{stage}")),
                "srpd_teacher_source": str(config["structured_risk_teacher_path"]),
                "srpd_teacher_variant": str(teacher_row.get("rerank_variant", "nonlinear_structured_risk_rerank")),
                "srpd_disagree_with_direct": disagreement,
                "srpd_target_source": target_source,
                "srpd_teacher_improves_positive": teacher_improves_positive,
                "srpd_teacher_positive_position": teacher_positive_position,
                "srpd_direct_positive_position": direct_positive_position,
                "srpd_target_positive_position": target_positive_position,
                "srpd_mean_uncertainty": round(uncertainty_summary["mean_uncertainty"], 6),
                "srpd_mean_effective_uncertainty": round(uncertainty_summary["mean_effective_uncertainty"], 6),
                "srpd_mean_risk_weight": round(uncertainty_summary["mean_risk_weight"], 6),
                "srpd_uncertainty_coverage_rate": round(
                    _as_float(teacher_row.get("event_uncertainty_coverage_rate"), default=uncertainty_summary["matched_uncertainty_rate"]),
                    6,
                ),
                "srpd_pairwise_preferences": pairwise_preferences if stage == "v3" else [],
                "srpd_pairwise_preference_count": len(pairwise_preferences) if stage == "v3" else 0,
                "srpd_dpo_style_preferences": dpo_style_preferences,
                "srpd_dpo_style_preference_count": len(dpo_style_preferences),
            }
        )
        record["srpd_sample_weight"] = _sample_weight(
            stage=stage,
            disagreement=disagreement,
            uncertainty_summary=uncertainty_summary,
            pairwise_preferences=pairwise_preferences,
            weight_cfg=weight_cfg,
        )
        records.append(record)

    train_rows, valid_rows = _split_records(records, float(config.get("train_ratio", 0.8)))
    output_train_path = Path(str(config["output_train_path"]))
    output_valid_path = Path(str(config["output_valid_path"]))
    save_jsonl(train_rows, output_train_path)
    save_jsonl(valid_rows, output_valid_path)

    total_pairwise = sum(int(row.get("srpd_pairwise_preference_count", 0)) for row in records)
    total_dpo_style = sum(int(row.get("srpd_dpo_style_preference_count", 0)) for row in records)
    avg_weight = sum(float(row.get("srpd_sample_weight", 0.0)) for row in records) / max(len(records), 1)
    summary = {
        "run_name": str(config.get("run_name", f"qwen3_rank_beauty_srpd_{stage}")),
        "srpd_stage": stage,
        "method_family": "SRPD",
        "method_variant": str(config.get("method_variant", f"srpd_{stage}")),
        "base_input_path": str(config["base_input_path"]),
        "structured_risk_teacher_path": str(config["structured_risk_teacher_path"]),
        "pairwise_preference_path": str(config.get("pairwise_preference_path", "")),
        "base_rows": len(base_rows),
        "teacher_rows": len(teacher_rows),
        "matched_rows": len(records),
        "missing_base_rows": missing_base_count,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "teacher_match_rate": round(len(records) / max(len(teacher_rows), 1), 6),
        "avg_sample_weight": round(avg_weight, 6),
        "avg_pairwise_preferences": round(total_pairwise / max(len(records), 1), 6),
        "avg_dpo_style_preferences": round(total_dpo_style / max(len(records), 1), 6),
        "output_train_path": str(output_train_path),
        "output_valid_path": str(output_valid_path),
        "status": "srpd_teacher_data_ready" if records else "no_matched_teacher_rows",
        "notes": str(config.get("notes", "")),
    }

    summary_path = Path(str(config.get("summary_path", f"outputs/summary/week7_6_srpd_{stage}_data_summary.csv")))
    _write_csv([summary], summary_path)
    markdown_path = Path(str(config.get("markdown_path", f"outputs/summary/week7_6_srpd_{stage}_data_summary.md")))
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                f"# {summary['run_name']} Data Summary",
                "",
                f"SRPD stage `{stage}` has been aligned to the structured-risk teacher without fabricating training metrics.",
                f"Matched rows: {summary['matched_rows']} / {summary['teacher_rows']}.",
                f"Train rows: {summary['train_rows']}; valid rows: {summary['valid_rows']}.",
                f"Average sample weight: {summary['avg_sample_weight']}.",
                f"Average pairwise preferences per event: {summary['avg_pairwise_preferences']}.",
                f"Average DPO-style preferences per event: {summary['avg_dpo_style_preferences']}.",
                "",
                "This artifact prepares the trainable SRPD path; it is not itself a completed model result.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
