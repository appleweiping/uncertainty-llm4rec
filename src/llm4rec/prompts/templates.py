"""Versioned prompt templates for Phase 3 LLM baselines."""

from __future__ import annotations

GENERATIVE_TITLE_TEMPLATE_ID = "phase3.generative_title.v1"
RERANK_TEMPLATE_ID = "phase3.candidate_rerank.v1"
YES_NO_VERIFY_TEMPLATE_ID = "phase3.yes_no_verify.v1"
CANDIDATE_NORMALIZED_TEMPLATE_ID = "phase3.candidate_normalized.v1"


GENERATIVE_TITLE_TEMPLATE = """You are a recommendation model.
Task: generate one catalog item title that may fit the user's next interaction.

Rules:
- Use only the user history and visible candidate titles below.
- Do not assume access to the true next item.
- Output valid JSON only.

User history titles:
{history_titles}

Visible candidate titles:
{candidate_titles}

Return this JSON schema:
{{
  "recommendation": "generated item title",
  "confidence": 0.0,
  "reason": "...",
  "uncertainty_reason": "..."
}}
"""


RERANK_TEMPLATE = """You are a candidate reranking model.
Task: rank only the visible candidate titles for the user's next interaction.

Rules:
- Rank only titles that appear in the visible candidate list.
- Do not introduce new titles.
- Do not assume access to the true next item.
- Output valid JSON only.

User history titles:
{history_titles}

Visible candidate titles:
{candidate_titles}

Return this JSON schema:
{{
  "ranked_items": [
    {{"title": "candidate title", "confidence": 0.0}}
  ],
  "reason": "..."
}}
"""


YES_NO_VERIFY_TEMPLATE = """You are checking a generated recommendation.
Task: decide whether the generated recommendation is likely to match the user's next interaction.

Rules:
- Use only the user history and generated recommendation below.
- Do not assume access to the true next item.
- Output valid JSON only.

User history titles:
{history_titles}

Generated recommendation title:
{generated_title}

Grounded catalog item, if available:
{grounded_title}

Return this JSON schema:
{{
  "answer": "yes",
  "confidence": 0.0,
  "reason": "..."
}}
"""


CANDIDATE_NORMALIZED_TEMPLATE = """You are estimating recommendation confidence over plausible item alternatives.
Task: assign normalized confidence to the generated recommendation and visible alternatives.

Rules:
- Use only the user history, generated recommendation, and visible alternatives below.
- Do not assume access to the true next item.
- Output valid JSON only.

User history titles:
{history_titles}

Generated recommendation title:
{generated_title}

Visible alternative titles:
{candidate_titles}

Return this JSON schema:
{{
  "options": [
    {{"title": "candidate title", "confidence": 0.0}}
  ],
  "normalized": true
}}
"""
