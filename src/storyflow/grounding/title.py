"""Lightweight catalog grounding for generated item titles."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

from storyflow.schemas import (
    GroundedPredictionRecord,
    GroundingCandidate,
    GroundingStatus,
    ItemCatalogRecord,
)

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", flags=re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_MIN_BLOCK_TOKEN_LENGTH = 2
_MAX_FUZZY_BLOCK_CANDIDATES = 1500
_FULL_SCAN_CATALOG_LIMIT = 2000


def normalize_title(title: str) -> str:
    """Normalize title text for transparent exact and fuzzy matching."""

    if title is None:
        raise ValueError("title must not be None")
    normalized = unicodedata.normalize("NFKC", title).casefold()
    normalized = normalized.replace("&", " and ")
    normalized = _NON_WORD_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    normalized = _ARTICLE_RE.sub("", normalized)
    return normalized


def compute_grounding_ambiguity(
    top_score: float,
    second_score: float | None,
) -> float:
    """Return a simple ambiguity proxy from the top-two similarity margin."""

    if second_score is None:
        return 0.0
    margin = max(0.0, min(1.0, top_score - second_score))
    return 1.0 - margin


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _block_tokens(normalized_title: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in normalized_title.split()
        if len(token) >= _MIN_BLOCK_TOKEN_LENGTH
    )


def _trigrams(normalized_title: str) -> tuple[str, ...]:
    compact = normalized_title.replace(" ", "")
    if len(compact) < 3:
        return ()
    return tuple(compact[index : index + 3] for index in range(len(compact) - 2))


def _candidate(
    item: ItemCatalogRecord,
    generated_normalized_title: str,
    rank: int,
) -> GroundingCandidate:
    catalog_normalized_title = normalize_title(item.title)
    score = _ratio(generated_normalized_title, catalog_normalized_title)
    if generated_normalized_title == catalog_normalized_title:
        score = 1.0
    return GroundingCandidate(
        item_id=item.item_id,
        title=item.title,
        normalized_title=catalog_normalized_title,
        score=score,
        rank=rank,
    )


def _candidate_from_normalized(
    item: ItemCatalogRecord,
    generated_normalized_title: str,
    catalog_normalized_title: str,
    rank: int,
) -> GroundingCandidate:
    score = _ratio(generated_normalized_title, catalog_normalized_title)
    if generated_normalized_title == catalog_normalized_title:
        score = 1.0
    return GroundingCandidate(
        item_id=item.item_id,
        title=item.title,
        normalized_title=catalog_normalized_title,
        score=score,
        rank=rank,
    )


def _rank_candidates(
    generated_normalized_title: str,
    catalog: tuple[ItemCatalogRecord, ...],
    limit: int = 5,
) -> tuple[GroundingCandidate, ...]:
    candidates = [
        _candidate(item, generated_normalized_title, rank=1)
        for item in catalog
    ]
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.item_id))
    ranked = [
        GroundingCandidate(
            item_id=candidate.item_id,
            title=candidate.title,
            normalized_title=candidate.normalized_title,
            score=candidate.score,
            rank=index + 1,
        )
        for index, candidate in enumerate(candidates[:limit])
    ]
    return tuple(ranked)


def _record(
    *,
    prediction_id: str,
    generated_title: str,
    normalized_title: str,
    item_id: str | None,
    status: GroundingStatus,
    score: float,
    candidates: tuple[GroundingCandidate, ...],
) -> GroundedPredictionRecord:
    second_score = candidates[1].score if len(candidates) > 1 else None
    ambiguity = compute_grounding_ambiguity(
        candidates[0].score if candidates else 0.0,
        second_score,
    )
    return GroundedPredictionRecord(
        prediction_id=prediction_id,
        generated_title=generated_title,
        normalized_title=normalized_title,
        item_id=item_id,
        status=status,
        score=score,
        ambiguity=ambiguity,
        candidates=candidates,
        second_score=second_score,
    )


def ground_title(
    generated_title: str,
    catalog: Iterable[ItemCatalogRecord],
    *,
    prediction_id: str = "prediction",
    fuzzy_threshold: float = 0.86,
    ambiguity_margin: float = 0.03,
) -> GroundedPredictionRecord:
    """Ground a generated title to a catalog item using transparent matching."""

    catalog_tuple = tuple(catalog)
    if not catalog_tuple:
        raise ValueError("catalog must contain at least one item")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be in [0, 1]")
    if ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be non-negative")

    normalized_title = normalize_title(generated_title)
    candidates = _rank_candidates(normalized_title, catalog_tuple)
    if not normalized_title:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=None,
            status=GroundingStatus.UNGROUNDED,
            score=0.0,
            candidates=candidates,
        )

    exact_matches = [
        item for item in catalog_tuple if item.title == generated_title
    ]
    if len(exact_matches) == 1:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=exact_matches[0].item_id,
            status=GroundingStatus.EXACT,
            score=1.0,
            candidates=candidates,
        )
    if len(exact_matches) > 1:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=None,
            status=GroundingStatus.AMBIGUOUS,
            score=0.0,
            candidates=candidates,
        )

    normalized_matches = [
        item
        for item in catalog_tuple
        if normalize_title(item.title) == normalized_title
    ]
    if len(normalized_matches) == 1:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=normalized_matches[0].item_id,
            status=GroundingStatus.NORMALIZED_EXACT,
            score=0.98,
            candidates=candidates,
        )
    if len(normalized_matches) > 1:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=None,
            status=GroundingStatus.AMBIGUOUS,
            score=0.0,
            candidates=candidates,
        )

    top = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else None
    if top.score < fuzzy_threshold:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=None,
            status=GroundingStatus.UNGROUNDED,
            score=0.0,
            candidates=candidates,
        )
    if second_score is not None and top.score - second_score <= ambiguity_margin:
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=None,
            status=GroundingStatus.AMBIGUOUS,
            score=0.0,
            candidates=candidates,
        )
    return _record(
        prediction_id=prediction_id,
        generated_title=generated_title,
        normalized_title=normalized_title,
        item_id=top.item_id,
        status=GroundingStatus.FUZZY,
        score=top.score,
        candidates=candidates,
    )


class TitleGrounder:
    """Small reusable wrapper around an in-memory item catalog."""

    def __init__(
        self,
        catalog: Iterable[ItemCatalogRecord],
        *,
        fuzzy_threshold: float = 0.86,
        ambiguity_margin: float = 0.03,
    ) -> None:
        self.catalog = tuple(catalog)
        if not self.catalog:
            raise ValueError("catalog must contain at least one item")
        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin
        self._exact_index: dict[str, tuple[ItemCatalogRecord, ...]] = {}
        self._normalized_index: dict[str, tuple[ItemCatalogRecord, ...]] = {}
        exact_index: dict[str, list[ItemCatalogRecord]] = {}
        normalized_index: dict[str, list[ItemCatalogRecord]] = {}
        normalized_catalog: list[tuple[ItemCatalogRecord, str]] = []
        token_index: dict[str, set[int]] = {}
        trigram_index: dict[str, set[int]] = {}
        for item in self.catalog:
            normalized_title = normalize_title(item.title)
            item_index = len(normalized_catalog)
            normalized_catalog.append((item, normalized_title))
            exact_index.setdefault(item.title, []).append(item)
            normalized_index.setdefault(normalized_title, []).append(item)
            for token in set(_block_tokens(normalized_title)):
                token_index.setdefault(token, set()).add(item_index)
            for trigram in set(_trigrams(normalized_title)):
                trigram_index.setdefault(trigram, set()).add(item_index)
        self._normalized_catalog = tuple(normalized_catalog)
        self._token_index = token_index
        self._trigram_index = trigram_index
        self._exact_index = {
            title: tuple(items) for title, items in exact_index.items()
        }
        self._normalized_index = {
            title: tuple(items) for title, items in normalized_index.items()
        }

    def _indexed_candidates(
        self,
        items: tuple[ItemCatalogRecord, ...],
        generated_normalized_title: str,
    ) -> tuple[GroundingCandidate, ...]:
        return tuple(
            _candidate(
                item,
                generated_normalized_title,
                rank=index + 1,
            )
            for index, item in enumerate(sorted(items, key=lambda row: row.item_id))
        )

    def _rank_candidates(
        self,
        generated_normalized_title: str,
        limit: int = 5,
    ) -> tuple[GroundingCandidate, ...]:
        candidate_indices = self._candidate_indices(generated_normalized_title)
        candidates = [
            _candidate_from_normalized(
                item,
                generated_normalized_title,
                normalized_title,
                rank=1,
            )
            for index in candidate_indices
            for item, normalized_title in (self._normalized_catalog[index],)
        ]
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.item_id))
        return tuple(
            GroundingCandidate(
                item_id=candidate.item_id,
                title=candidate.title,
                normalized_title=candidate.normalized_title,
                score=candidate.score,
                rank=index + 1,
            )
            for index, candidate in enumerate(candidates[:limit])
        )

    def _candidate_indices(self, generated_normalized_title: str) -> tuple[int, ...]:
        if len(self._normalized_catalog) <= _FULL_SCAN_CATALOG_LIMIT:
            return tuple(range(len(self._normalized_catalog)))

        weights: dict[int, int] = {}
        for token in set(_block_tokens(generated_normalized_title)):
            for index in self._token_index.get(token, ()):
                weights[index] = weights.get(index, 0) + 4
        for trigram in set(_trigrams(generated_normalized_title)):
            for index in self._trigram_index.get(trigram, ()):
                weights[index] = weights.get(index, 0) + 1

        if not weights:
            return ()

        generated_length = len(generated_normalized_title)
        ranked = sorted(
            weights,
            key=lambda index: (
                -weights[index],
                abs(len(self._normalized_catalog[index][1]) - generated_length),
                self._normalized_catalog[index][0].item_id,
            ),
        )
        return tuple(ranked[:_MAX_FUZZY_BLOCK_CANDIDATES])

    def ground(
        self,
        generated_title: str,
        *,
        prediction_id: str = "prediction",
    ) -> GroundedPredictionRecord:
        normalized_title = normalize_title(generated_title)
        if not normalized_title:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.UNGROUNDED,
                score=0.0,
                candidates=(),
            )

        exact_matches = self._exact_index.get(generated_title, ())
        if len(exact_matches) == 1:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=exact_matches[0].item_id,
                status=GroundingStatus.EXACT,
                score=1.0,
                candidates=self._indexed_candidates(exact_matches, normalized_title),
            )
        if len(exact_matches) > 1:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.AMBIGUOUS,
                score=0.0,
                candidates=self._indexed_candidates(exact_matches, normalized_title),
            )

        normalized_matches = self._normalized_index.get(normalized_title, ())
        if len(normalized_matches) == 1:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=normalized_matches[0].item_id,
                status=GroundingStatus.NORMALIZED_EXACT,
                score=0.98,
                candidates=self._indexed_candidates(normalized_matches, normalized_title),
            )
        if len(normalized_matches) > 1:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.AMBIGUOUS,
                score=0.0,
                candidates=self._indexed_candidates(normalized_matches, normalized_title),
            )

        candidates = self._rank_candidates(normalized_title)
        if not candidates:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.UNGROUNDED,
                score=0.0,
                candidates=(),
            )
        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else None
        if top.score < self.fuzzy_threshold:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.UNGROUNDED,
                score=0.0,
                candidates=candidates,
            )
        if second_score is not None and top.score - second_score <= self.ambiguity_margin:
            return _record(
                prediction_id=prediction_id,
                generated_title=generated_title,
                normalized_title=normalized_title,
                item_id=None,
                status=GroundingStatus.AMBIGUOUS,
                score=0.0,
                candidates=candidates,
            )
        return _record(
            prediction_id=prediction_id,
            generated_title=generated_title,
            normalized_title=normalized_title,
            item_id=top.item_id,
            status=GroundingStatus.FUZZY,
            score=top.score,
            candidates=candidates,
        )
