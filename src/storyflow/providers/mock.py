"""Deterministic mock provider for no-API observation pipeline tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ParsedProviderResponse:
    """Parsed generated recommendation response."""

    generated_title: str
    confidence: float
    is_likely_correct: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MockProviderOutput:
    """Raw and parsed output from the mock provider."""

    provider: str
    provider_mode: str
    raw_text: str
    parsed: ParsedProviderResponse


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        number = float(value)
        if number > 1.0:
            number = number / 100.0
    else:
        number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be in [0, 1] or [0, 100]")
    return number


def parse_provider_response(raw_text: str) -> ParsedProviderResponse:
    """Parse a JSON provider response into generated title and confidence."""

    payload = json.loads(raw_text)
    generated_title = str(payload.get("generated_title", "")).strip()
    if not generated_title:
        raise ValueError("provider response missing generated_title")
    confidence = _coerce_confidence(payload.get("confidence"))
    yes_no = str(payload.get("is_likely_correct", "")).strip().lower()
    if yes_no not in {"yes", "no"}:
        raise ValueError("is_likely_correct must be yes or no")
    return ParsedProviderResponse(
        generated_title=generated_title,
        confidence=confidence,
        is_likely_correct=yes_no,
        raw=payload,
    )


class MockProvider:
    """No-API provider with deterministic oracle, popularity, and random modes."""

    provider = "mock"

    def __init__(
        self,
        catalog_items: Iterable[dict[str, Any]],
        *,
        mode: str = "popularity_biased",
        seed: int = 13,
    ) -> None:
        if mode not in {"oracle-ish", "popularity_biased", "random"}:
            raise ValueError("mode must be oracle-ish, popularity_biased, or random")
        self.mode = mode
        self.seed = int(seed)
        self.catalog_items = sorted(
            (dict(item) for item in catalog_items),
            key=lambda item: (
                -int(float(item.get("popularity") or 0)),
                str(item.get("item_id", "")),
            ),
        )
        if not self.catalog_items:
            raise ValueError("catalog_items must not be empty")

    def generate(self, input_record: dict[str, Any]) -> MockProviderOutput:
        """Generate a mock title/confidence response for one observation input."""

        digest = _stable_int(
            f"{self.seed}:{self.mode}:{input_record['input_id']}:{input_record['prompt_hash']}"
        )
        if self.mode == "oracle-ish":
            payload = self._oracleish_payload(input_record, digest)
        elif self.mode == "popularity_biased":
            payload = self._popularity_payload(input_record, digest)
        else:
            payload = self._random_payload(digest)
        raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return MockProviderOutput(
            provider=self.provider,
            provider_mode=self.mode,
            raw_text=raw_text,
            parsed=parse_provider_response(raw_text),
        )

    def _oracleish_payload(self, input_record: dict[str, Any], digest: int) -> dict[str, Any]:
        bucket = digest % 10
        if bucket < 7:
            return self._payload(input_record["target_title"], 0.82, "yes", "oracle_target")
        if bucket < 9:
            title = self.catalog_items[digest % min(10, len(self.catalog_items))]["title"]
            return self._payload(title, 0.64, "yes", "oracle_popular_fallback")
        return self._payload("A Catalog Title That Does Not Exist", 0.36, "no", "oracle_abstain")

    def _popularity_payload(self, input_record: dict[str, Any], digest: int) -> dict[str, Any]:
        bucket = digest % 10
        if bucket < 7:
            title = self.catalog_items[digest % min(10, len(self.catalog_items))]["title"]
            return self._payload(title, 0.88, "yes", "popular_prior")
        if bucket < 9:
            return self._payload(input_record["target_title"], 0.70, "yes", "target_fallback")
        return self._payload("A Popular Sounding Non Catalog Title", 0.78, "yes", "hallucinated")

    def _random_payload(self, digest: int) -> dict[str, Any]:
        item = self.catalog_items[digest % len(self.catalog_items)]
        confidence = 0.2 + ((digest // 13) % 61) / 100.0
        yes_no = "yes" if confidence >= 0.5 else "no"
        return self._payload(item["title"], round(confidence, 2), yes_no, "random_catalog")

    @staticmethod
    def _payload(
        generated_title: str,
        confidence: float,
        yes_no: str,
        strategy: str,
    ) -> dict[str, Any]:
        return {
            "generated_title": generated_title,
            "is_likely_correct": yes_no,
            "confidence": confidence,
            "mock_strategy": strategy,
        }
