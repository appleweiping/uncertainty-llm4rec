"""Lightweight baseline observation runners for title-level recommendation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from storyflow.grounding import TitleGrounder
from storyflow.observation import (
    catalog_records,
    compute_observation_metrics,
    load_catalog_rows,
    observation_metrics_markdown,
    read_jsonl,
    utc_now_iso,
    write_jsonl,
)


@dataclass(frozen=True, slots=True)
class BaselineOutput:
    """A baseline-generated title and confidence proxy."""

    generated_title: str
    confidence: float
    is_likely_correct: str
    selected_item_id: str | None
    score: float
    score_source: str


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return min(0.95, max(0.05, value))


def _catalog_by_id(catalog_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): dict(row) for row in catalog_rows}


class PopularityTitleBaseline:
    """Recommend the most popular unseen catalog title."""

    name = "popularity"

    def __init__(self, catalog_rows: Iterable[dict[str, Any]]) -> None:
        self.catalog_rows = sorted(
            (dict(row) for row in catalog_rows),
            key=lambda row: (
                -int(row.get("popularity") or 0),
                str(row.get("title") or ""),
                str(row.get("item_id") or ""),
            ),
        )
        if not self.catalog_rows:
            raise ValueError("catalog_rows must not be empty")
        self.max_popularity = max(
            (int(row.get("popularity") or 0) for row in self.catalog_rows),
            default=0,
        )

    def predict(self, input_record: dict[str, Any]) -> BaselineOutput:
        history_ids = {str(item_id) for item_id in input_record.get("history_item_ids", [])}
        candidate = next(
            (row for row in self.catalog_rows if str(row.get("item_id")) not in history_ids),
            self.catalog_rows[0],
        )
        popularity = int(candidate.get("popularity") or 0)
        confidence = (
            math.log1p(popularity) / math.log1p(self.max_popularity)
            if self.max_popularity > 0
            else 0.5
        )
        confidence = _clip_probability(confidence)
        return BaselineOutput(
            generated_title=str(candidate.get("title") or ""),
            confidence=confidence,
            is_likely_correct="yes" if confidence >= 0.5 else "no",
            selected_item_id=str(candidate.get("item_id") or ""),
            score=float(popularity),
            score_source="catalog_popularity",
        )


class CooccurrenceTitleBaseline:
    """Train-split item co-occurrence baseline over observation examples."""

    name = "cooccurrence"

    def __init__(
        self,
        *,
        catalog_rows: Iterable[dict[str, Any]],
        observation_examples_jsonl: str | Path,
        train_split: str = "train",
        smoothing: float = 5.0,
    ) -> None:
        self.catalog_rows = list(catalog_rows)
        self.catalog_by_id = _catalog_by_id(self.catalog_rows)
        self.popularity_fallback = PopularityTitleBaseline(self.catalog_rows)
        self.smoothing = smoothing
        self.cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_targets: Counter[str] = Counter()
        for example in read_jsonl(observation_examples_jsonl):
            if str(example.get("split")) != train_split:
                continue
            target_item_id = str(example.get("target_item_id") or "")
            if not target_item_id or target_item_id not in self.catalog_by_id:
                continue
            self.global_targets[target_item_id] += 1
            history_ids = [str(item_id) for item_id in example.get("history_item_ids", [])]
            for history_item_id in history_ids:
                if history_item_id:
                    self.cooccurrence[history_item_id][target_item_id] += 1

    def _score_candidates(self, input_record: dict[str, Any]) -> Counter[str]:
        scores: Counter[str] = Counter()
        history_ids = [str(item_id) for item_id in input_record.get("history_item_ids", [])]
        history_len = max(len(history_ids), 1)
        for index, history_item_id in enumerate(history_ids):
            recency_weight = (index + 1) / history_len
            for candidate_id, count in self.cooccurrence.get(history_item_id, {}).items():
                scores[candidate_id] += count * recency_weight
        return scores

    def predict(self, input_record: dict[str, Any]) -> BaselineOutput:
        history_ids = {str(item_id) for item_id in input_record.get("history_item_ids", [])}
        scores = self._score_candidates(input_record)
        for history_item_id in history_ids:
            scores.pop(history_item_id, None)
        if not scores:
            fallback = self.popularity_fallback.predict(input_record)
            return BaselineOutput(
                generated_title=fallback.generated_title,
                confidence=fallback.confidence,
                is_likely_correct=fallback.is_likely_correct,
                selected_item_id=fallback.selected_item_id,
                score=fallback.score,
                score_source="popularity_fallback",
            )
        selected_item_id, score = max(
            scores.items(),
            key=lambda item: (
                item[1],
                int(self.catalog_by_id.get(item[0], {}).get("popularity") or 0),
                str(self.catalog_by_id.get(item[0], {}).get("title") or ""),
                item[0],
            ),
        )
        selected = self.catalog_by_id[selected_item_id]
        confidence = _clip_probability(float(score) / (float(score) + self.smoothing))
        return BaselineOutput(
            generated_title=str(selected.get("title") or ""),
            confidence=confidence,
            is_likely_correct="yes" if confidence >= 0.5 else "no",
            selected_item_id=selected_item_id,
            score=float(score),
            score_source="train_split_cooccurrence",
        )


def build_baseline(
    baseline: str,
    *,
    catalog_rows: list[dict[str, Any]],
    observation_examples_jsonl: str | Path,
) -> PopularityTitleBaseline | CooccurrenceTitleBaseline:
    if baseline == PopularityTitleBaseline.name:
        return PopularityTitleBaseline(catalog_rows)
    if baseline == CooccurrenceTitleBaseline.name:
        return CooccurrenceTitleBaseline(
            catalog_rows=catalog_rows,
            observation_examples_jsonl=observation_examples_jsonl,
        )
    raise ValueError(f"unknown baseline: {baseline}")


def default_baseline_output_dir(
    *,
    input_jsonl: str | Path,
    baseline: str,
    root: str | Path = ".",
) -> Path:
    input_path = Path(input_jsonl)
    parts = input_path.parts
    dataset = parts[-3] if len(parts) >= 3 else "dataset"
    processed_suffix = parts[-2] if len(parts) >= 2 else "processed"
    run_name = f"{input_path.stem}_{baseline}"
    return Path(root) / "outputs" / "observations" / "baselines" / dataset / processed_suffix / run_name


def _completed_input_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("input_id")) for row in read_jsonl(path)}


def _candidate_dicts(candidates: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates]


def run_baseline_observation(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    baseline: str = "popularity",
    max_examples: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_path = output_path / "raw_responses.jsonl"
    parsed_path = output_path / "parsed_predictions.jsonl"
    grounded_path = output_path / "grounded_predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    report_path = output_path / "report.md"
    manifest_path = output_path / "manifest.json"

    inputs = read_jsonl(input_path)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise ValueError("input_jsonl contains no records to process")

    catalog_csv = inputs[0]["source"]["catalog_csv"]
    examples_jsonl = inputs[0]["source"]["observation_examples"]
    catalog_rows = load_catalog_rows(catalog_csv)
    baseline_model = build_baseline(
        baseline,
        catalog_rows=catalog_rows,
        observation_examples_jsonl=examples_jsonl,
    )
    grounder = TitleGrounder(catalog_records(catalog_rows))
    completed = _completed_input_ids(grounded_path) if resume else set()
    if not resume:
        raw_path.write_text("", encoding="utf-8")
        parsed_path.write_text("", encoding="utf-8")
        grounded_path.write_text("", encoding="utf-8")

    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    grounded_rows: list[dict[str, Any]] = []
    for input_record in inputs:
        input_id = str(input_record["input_id"])
        if input_id in completed:
            continue
        prediction = baseline_model.predict(input_record)
        raw_text = json.dumps(
            {
                "generated_title": prediction.generated_title,
                "is_likely_correct": prediction.is_likely_correct,
                "confidence": prediction.confidence,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        grounded = grounder.ground(
            prediction.generated_title,
            prediction_id=f"baseline:{baseline}:{input_id}",
        )
        correctness = int(
            grounded.is_grounded and grounded.item_id == input_record["target_item_id"]
        )
        raw_rows.append(
            {
                "input_id": input_id,
                "example_id": input_record["example_id"],
                "provider": "baseline",
                "baseline": baseline,
                "raw_text": raw_text,
                "created_at_utc": utc_now_iso(),
                "is_experiment_result": False,
            }
        )
        parsed_rows.append(
            {
                "input_id": input_id,
                "example_id": input_record["example_id"],
                "provider": "baseline",
                "baseline": baseline,
                "generated_title": prediction.generated_title,
                "confidence": prediction.confidence,
                "is_likely_correct": prediction.is_likely_correct,
                "selected_item_id": prediction.selected_item_id,
                "baseline_score": prediction.score,
                "baseline_score_source": prediction.score_source,
                "parse_strategy": "baseline_structured",
                "is_experiment_result": False,
            }
        )
        grounded_rows.append(
            {
                "input_id": input_id,
                "example_id": input_record["example_id"],
                "user_id": input_record["user_id"],
                "split": input_record["split"],
                "prompt_hash": input_record.get("prompt_hash"),
                "provider": "baseline",
                "baseline": baseline,
                "model": baseline,
                "generated_title": prediction.generated_title,
                "confidence": prediction.confidence,
                "is_likely_correct": prediction.is_likely_correct,
                "selected_item_id": prediction.selected_item_id,
                "baseline_score": prediction.score,
                "baseline_score_source": prediction.score_source,
                "target_item_id": input_record["target_item_id"],
                "target_title": input_record["target_title"],
                "target_popularity": input_record["target_popularity"],
                "target_popularity_bucket": input_record["target_popularity_bucket"],
                "grounded_item_id": grounded.item_id,
                "grounding_status": grounded.status.value,
                "grounding_score": grounded.score,
                "grounding_ambiguity": grounded.ambiguity,
                "grounding_second_score": grounded.second_score,
                "grounding_candidates": _candidate_dicts(grounded.candidates),
                "correctness": correctness,
                "parse_strategy": "baseline_structured",
                "api_called": False,
                "is_experiment_result": False,
            }
        )

    append_outputs = resume and grounded_path.exists()
    if raw_rows:
        write_jsonl(raw_path, raw_rows, append=append_outputs and raw_path.exists())
    if parsed_rows:
        write_jsonl(parsed_path, parsed_rows, append=append_outputs and parsed_path.exists())
    if grounded_rows:
        write_jsonl(grounded_path, grounded_rows, append=append_outputs)

    selected_input_ids = {str(input_record["input_id"]) for input_record in inputs}
    all_grounded = [
        row
        for row in read_jsonl(grounded_path)
        if str(row.get("input_id")) in selected_input_ids
    ]
    metrics = compute_observation_metrics(all_grounded)
    metrics.update(
        {
            "provider": "baseline",
            "baseline": baseline,
            "api_called": False,
            "is_experiment_result": False,
            "note": (
                "Lightweight baseline observation sanity metrics. No API, "
                "training, or paper result."
            ),
        }
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        observation_metrics_markdown(metrics, title=f"Baseline Observation Report: {baseline}"),
        encoding="utf-8",
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "provider": "baseline",
        "baseline": baseline,
        "input_jsonl": str(input_path),
        "output_dir": str(output_path),
        "raw_responses": str(raw_path),
        "parsed_predictions": str(parsed_path),
        "grounded_predictions": str(grounded_path),
        "metrics": str(metrics_path),
        "report": str(report_path),
        "requested_input_count": len(inputs),
        "newly_processed_count": len(grounded_rows),
        "total_grounded_count": len(all_grounded),
        "resume": resume,
        "api_called": False,
        "model_training": False,
        "is_experiment_result": False,
        "note": "Baseline observation run only. No external API, model training, or paper result.",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return manifest
