from __future__ import annotations

from llm4rec.prompts.builder import (
    build_candidate_normalized_prompt,
    build_generative_title_prompt,
    build_rerank_prompt,
    build_yes_no_verification_prompt,
)


CATALOG = [
    {"item_id": "i1", "title": "Alpha Movie", "category": "Drama"},
    {"item_id": "i2", "title": "Beta Movie", "category": "Comedy"},
    {"item_id": "i3", "title": "Gamma Movie", "category": "Sci-Fi"},
]
EXAMPLE = {
    "example_id": "e1",
    "user_id": "u1",
    "history": ["i1"],
    "target": "i3",
    "candidates": ["i2", "i3"],
    "split": "test",
    "domain": "tiny",
}


def test_generative_prompt_excludes_target_title() -> None:
    prompt = build_generative_title_prompt(
        example=EXAMPLE,
        item_catalog=CATALOG,
        candidate_items=["i2", "i3"],
        exclude_item_ids={"i3"},
    )
    assert "Alpha Movie" in prompt.prompt
    assert "Beta Movie" in prompt.prompt
    assert "Gamma Movie" not in prompt.prompt
    assert prompt.metadata["target_excluded_from_prompt"] is True
    assert prompt.prompt_hash


def test_prompt_builders_apply_target_exclusion_by_default() -> None:
    repeated_target = {
        **EXAMPLE,
        "history": ["i1", "i3"],
    }
    generative = build_generative_title_prompt(
        example=repeated_target,
        item_catalog=CATALOG,
        candidate_items=["i2", "i3"],
    )
    rerank = build_rerank_prompt(
        example=repeated_target,
        item_catalog=CATALOG,
        candidate_items=["i2", "i3"],
    )
    verify = build_yes_no_verification_prompt(
        example=repeated_target,
        item_catalog=CATALOG,
        generated_title="Beta Movie",
    )
    normalized = build_candidate_normalized_prompt(
        example=repeated_target,
        item_catalog=CATALOG,
        generated_title="Beta Movie",
        candidate_items=["i2", "i3"],
    )
    for prompt in [generative, rerank, verify, normalized]:
        assert "Gamma Movie" not in prompt.prompt
        assert "i3" not in prompt.prompt
        assert prompt.metadata["target_excluded_from_prompt"] is True
        assert prompt.metadata["history_excluded_item_ids"] == ["i3"]


def test_all_phase3_prompt_types_have_template_ids() -> None:
    rerank = build_rerank_prompt(
        example=EXAMPLE,
        item_catalog=CATALOG,
        candidate_items=["i2", "i3"],
        exclude_item_ids={"i3"},
    )
    verify = build_yes_no_verification_prompt(
        example=EXAMPLE,
        item_catalog=CATALOG,
        generated_title="Beta Movie",
    )
    normalized = build_candidate_normalized_prompt(
        example=EXAMPLE,
        item_catalog=CATALOG,
        generated_title="Beta Movie",
        candidate_items=["i2", "i3"],
        exclude_item_ids={"i3"},
    )
    assert rerank.prompt_template_id.endswith("candidate_rerank.v1")
    assert verify.prompt_template_id.endswith("yes_no_verify.v1")
    assert normalized.prompt_template_id.endswith("candidate_normalized.v1")
    assert "Gamma Movie" not in rerank.prompt
    assert "Gamma Movie" not in normalized.prompt
