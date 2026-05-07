"""TRUCE-native training data builders for the original Ours framework.

These helpers turn same-candidate recommendation examples into supervision for
an uncertainty-aware generative/reranking adapter. They do not train a model;
they create auditable SFT and scoring contracts for server-side Qwen3 adapter
training.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OursTrainingRow:
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


def item_text(item_id: str, item_lookup: dict[str, dict[str, Any]]) -> str:
    row = item_lookup.get(str(item_id), {})
    title = str(row.get("title") or row.get("raw_text") or item_id)
    category = str(row.get("category") or "")
    brand = str(row.get("brand") or "")
    bits = [title]
    if category:
        bits.append(f"category={category}")
    if brand:
        bits.append(f"brand={brand}")
    return " | ".join(bits)


def history_text(example: dict[str, Any], item_lookup: dict[str, dict[str, Any]], *, max_history: int = 50) -> str:
    history = [str(x) for x in example.get("history") or example.get("history_item_ids") or []][-max_history:]
    if not history:
        return "(empty)"
    return " ; ".join(item_text(item_id, item_lookup) for item_id in history)


def popularity_bucket(item_id: str, train_popularity: dict[str, int]) -> str:
    if not train_popularity:
        return "unknown"
    counts = sorted(train_popularity.values())
    value = int(train_popularity.get(str(item_id), 0))
    if not counts or value <= 0:
        return "tail"
    q80 = counts[max(0, int(0.8 * (len(counts) - 1)))]
    q50 = counts[max(0, int(0.5 * (len(counts) - 1)))]
    if value >= q80:
        return "head"
    if value >= q50:
        return "mid"
    return "tail"


def build_pairwise_prompt(
    example: dict[str, Any],
    *,
    candidate_item_id: str,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> str:
    """Prompt for candidate-level acceptance with uncertainty-aware evidence."""

    candidate = item_text(candidate_item_id, item_lookup)
    bucket = popularity_bucket(candidate_item_id, train_popularity)
    return (
        "TRUCE uncertainty-aware recommendation task.\n"
        "Estimate whether the candidate should be accepted for the user's next interaction.\n"
        "Use user preference evidence, catalog grounding, popularity/long-tail risk, and history repetition risk.\n"
        f"User history: {history_text(example, item_lookup, max_history=max_history)}\n"
        f"Candidate item: {candidate}\n"
        f"Candidate item id: {candidate_item_id}\n"
        f"Train-popularity bucket: {bucket}\n"
        "Answer exactly in JSON with keys: accept, confidence, risk_reason."
    )


def build_pairwise_answer(*, is_positive: bool, candidate_item_id: str, bucket: str) -> str:
    confidence = 0.9 if is_positive else 0.1
    risk = "positive_next_item" if is_positive else f"negative_candidate_{bucket}"
    accept = "true" if is_positive else "false"
    return f'{{"accept": {accept}, "confidence": {confidence:.1f}, "risk_reason": "{risk}"}}'


def build_listwise_prompt(
    example: dict[str, Any],
    *,
    candidate_item_ids: list[str],
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> str:
    lines = [
        "TRUCE listwise reranking task.",
        "Rank the candidate IDs for the user's next interaction.",
        "Prefer grounded preference evidence, but avoid blindly over-trusting popular or history-repetitive items.",
        f"User history: {history_text(example, item_lookup, max_history=max_history)}",
        "Candidates:",
    ]
    for item_id in candidate_item_ids:
        lines.append(
            f"- {item_id}: {item_text(item_id, item_lookup)} | train-popularity={popularity_bucket(item_id, train_popularity)}"
        )
    lines.append("Return JSON with keys: ranked_item_ids, confidence_by_item, risk_notes.")
    return "\n".join(lines)


def build_listwise_answer(*, target_item_id: str, candidate_item_ids: list[str]) -> str:
    ranked = [target_item_id] + [item for item in candidate_item_ids if item != target_item_id]
    confidence = {item: (0.9 if item == target_item_id else 0.1) for item in ranked}
    return (
        '{"ranked_item_ids": '
        + _json_list(ranked)
        + ', "confidence_by_item": '
        + _json_float_map(confidence)
        + ', "risk_notes": "supervised_next_item_target_first"}'
    )


def build_training_rows(
    example: dict[str, Any],
    *,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    negatives_per_example: int = 15,
    include_listwise: bool = True,
    max_history: int = 50,
) -> list[OursTrainingRow]:
    target = str(example.get("target") or example.get("target_item") or "")
    candidates = [str(item) for item in example.get("candidates") or example.get("candidate_items") or []]
    negatives = [item for item in candidates if item != target]
    negatives = _stable_sample(negatives, k=negatives_per_example, key=str(example.get("example_id") or ""))
    selected = ([target] if target else []) + negatives
    rows: list[OursTrainingRow] = []
    for item_id in selected:
        bucket = popularity_bucket(item_id, train_popularity)
        is_positive = item_id == target
        rows.append(
            OursTrainingRow(
                messages=[
                    {
                        "role": "user",
                        "content": build_pairwise_prompt(
                            example,
                            candidate_item_id=item_id,
                            item_lookup=item_lookup,
                            train_popularity=train_popularity,
                            max_history=max_history,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": build_pairwise_answer(
                            is_positive=is_positive,
                            candidate_item_id=item_id,
                            bucket=bucket,
                        ),
                    },
                ],
                metadata={
                    "example_id": str(example.get("example_id") or ""),
                    "target_item_id": target,
                    "candidate_item_id": item_id,
                    "supervision_type": "pairwise_acceptance",
                    "is_positive": is_positive,
                    "popularity_bucket": bucket,
                },
            )
        )
    if include_listwise and target and candidates:
        panel = _listwise_panel(candidates, target=target, train_popularity=train_popularity, key=str(example.get("example_id") or ""))
        rows.append(
            OursTrainingRow(
                messages=[
                    {
                        "role": "user",
                        "content": build_listwise_prompt(
                            example,
                            candidate_item_ids=panel,
                            item_lookup=item_lookup,
                            train_popularity=train_popularity,
                            max_history=max_history,
                        ),
                    },
                    {"role": "assistant", "content": build_listwise_answer(target_item_id=target, candidate_item_ids=panel)},
                ],
                metadata={
                    "example_id": str(example.get("example_id") or ""),
                    "target_item_id": target,
                    "candidate_item_ids": panel,
                    "supervision_type": "listwise_target_first",
                },
            )
        )
    return rows


def build_score_row(
    example: dict[str, Any],
    *,
    candidate_item_id: str,
    item_lookup: dict[str, dict[str, Any]],
    train_popularity: dict[str, int],
    max_history: int = 50,
) -> dict[str, Any]:
    return {
        "example_id": str(example.get("example_id") or ""),
        "user_id": str(example.get("user_id") or ""),
        "prompt": build_pairwise_prompt(
            example,
            candidate_item_id=candidate_item_id,
            item_lookup=item_lookup,
            train_popularity=train_popularity,
            max_history=max_history,
        ),
        "candidate_item_ids": [str(candidate_item_id)],
        "candidate_outputs": ['{"accept": true'],
        "metadata": {
            "event_id": _metadata_value(example, "event_id", str(example.get("example_id") or "")),
            "source_event_id": _metadata_value(example, "source_event_id", str(example.get("example_id") or "")),
            "supervision_type": "pairwise_acceptance_score",
        },
        "scoring_contract": "Score the likelihood that the adapter accepts this candidate under the TRUCE uncertainty-aware prompt.",
    }


def _metadata_value(row: dict[str, Any], key: str, default: str = "") -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get(key) or meta.get(key) or default)


def _stable_sample(items: list[str], *, k: int, key: str) -> list[str]:
    if k < 0 or len(items) <= k:
        return list(items)
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{key}:{item}".encode()).hexdigest())
    return ranked[:k]


def _listwise_panel(items: list[str], *, target: str, train_popularity: dict[str, int], key: str, size: int = 12) -> list[str]:
    chosen = [target] if target in items else []
    rest = [item for item in items if item != target]
    rest.sort(key=lambda item: (-int(train_popularity.get(item, 0)), hashlib.sha256(f"{key}:head:{item}".encode()).hexdigest()))
    chosen.extend(rest[: max(0, size // 2 - len(chosen))])
    tail = sorted(rest, key=lambda item: (int(train_popularity.get(item, 0)), hashlib.sha256(f"{key}:tail:{item}".encode()).hexdigest()))
    for item in tail:
        if item not in chosen:
            chosen.append(item)
        if len(chosen) >= size:
            break
    return chosen[:size]


def _json_list(values: list[str]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False)


def _json_float_map(values: dict[str, float]) -> str:
    import json

    return json.dumps(values, ensure_ascii=False, sort_keys=True)
