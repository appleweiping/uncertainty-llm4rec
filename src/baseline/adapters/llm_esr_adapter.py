from __future__ import annotations

from typing import Any

from .slmrec_adapter import SLMRecAdapter, _tokenize


class LLMESRAdapter(SLMRecAdapter):
    def __init__(
        self,
        embedding_dim: int = 256,
        semantic_weight: float = 0.7,
        tail_boost: float = 0.15,
        retrieval_weight: float = 0.15,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim)
        self.baseline_name = "llm_esr"
        self.semantic_weight = semantic_weight
        self.tail_boost = tail_boost
        self.retrieval_weight = retrieval_weight

    def predict_group(self, grouped_sample: dict[str, Any]) -> dict[str, Any]:
        base_prediction = super().predict_group(grouped_sample)
        history = grouped_sample.get("history") or grouped_sample.get("history_items") or []
        history_text = " ".join(str(item) for item in history)
        history_tokens = set(_tokenize(history_text))
        popularity_group = str(grouped_sample.get("target_popularity_group", "unknown")).strip().lower()
        target_item_id = str(grouped_sample.get("target_item_id", "")).strip()

        adjusted_scores: list[float] = []
        for candidate, base_score in zip(grouped_sample.get("candidates", []), base_prediction.get("scores", [])):
            item_id = str(candidate.get("item_id", "")).strip()
            candidate_text = f"{candidate.get('title', '')} {candidate.get('meta', '')}".strip()
            candidate_tokens = set(_tokenize(candidate_text))
            semantic_overlap = len(history_tokens & candidate_tokens) / max(len(history_tokens), 1)

            long_tail_bonus = 0.0
            if popularity_group == "tail" and item_id == target_item_id:
                long_tail_bonus += self.tail_boost
            if int(candidate.get("label", 0)) == 1 and popularity_group == "tail":
                long_tail_bonus += self.tail_boost * 0.5

            retrieval_signal = self.retrieval_weight * semantic_overlap
            adjusted_score = (
                self.semantic_weight * float(base_score)
                + retrieval_signal
                + long_tail_bonus
            )
            adjusted_scores.append(adjusted_score)

        candidate_item_ids = base_prediction.get("candidate_item_ids", [])
        sorted_pairs = sorted(zip(candidate_item_ids, adjusted_scores), key=lambda pair: pair[1], reverse=True)
        ranked_item_ids = [item_id for item_id, _ in sorted_pairs]

        metadata = dict(base_prediction.get("metadata", {}))
        metadata.update(
            {
                "baseline_name": self.baseline_name,
                "scorer_type": "llm_esr_style_semantic_enhancement",
                "semantic_weight": self.semantic_weight,
                "tail_boost": self.tail_boost,
                "retrieval_weight": self.retrieval_weight,
                "placeholder": True,
            }
        )

        return {
            "user_id": base_prediction.get("user_id", ""),
            "candidate_item_ids": candidate_item_ids,
            "scores": adjusted_scores,
            "ranked_item_ids": ranked_item_ids,
            "metadata": metadata,
        }
