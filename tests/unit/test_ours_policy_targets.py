from llm4rec.methods.ours_framework import candidate_policy_target


ITEMS = {
    "hist": {"title": "Rose Serum", "category": "beauty", "brand": "A"},
    "pos": {"title": "Gentle Cleanser", "category": "beauty", "brand": "B"},
    "head": {"title": "Rose Serum Plus", "category": "beauty", "brand": "A"},
    "tail": {"title": "Niche Mineral Mask", "category": "skin", "brand": "C"},
}


def test_policy_target_promotes_positive_with_bounded_risk() -> None:
    example = {
        "example_id": "e1",
        "history": ["hist"],
        "target": "pos",
        "candidates": ["pos", "head", "tail"],
    }
    target = candidate_policy_target(
        example,
        candidate_item_id="pos",
        target_item_id="pos",
        candidate_item_ids=["pos", "head", "tail"],
        item_lookup=ITEMS,
        train_popularity={"hist": 5, "pos": 2, "head": 100, "tail": 1},
    )

    assert target.policy_action == "promote"
    assert 0.0 <= target.candidate_normalized_utility <= 1.0
    assert 0.0 <= target.popularity_residual_utility <= 1.0
    assert target.harm_risk < 0.55


def test_policy_target_suppresses_echo_or_head_bias_negative() -> None:
    example = {
        "example_id": "e1",
        "history": ["hist"],
        "target": "pos",
        "candidates": ["pos", "head", "tail"],
    }
    positive = candidate_policy_target(
        example,
        candidate_item_id="pos",
        target_item_id="pos",
        candidate_item_ids=["pos", "head", "tail"],
        item_lookup=ITEMS,
        train_popularity={"hist": 5, "pos": 2, "head": 100, "tail": 1},
    )
    target = candidate_policy_target(
        example,
        candidate_item_id="head",
        target_item_id="pos",
        candidate_item_ids=["pos", "head", "tail"],
        item_lookup=ITEMS,
        train_popularity={"hist": 5, "pos": 2, "head": 100, "tail": 1},
    )

    assert target.policy_action == "suppress"
    assert target.harm_risk > positive.harm_risk
    assert "risk" in target.policy_reason


def test_policy_target_is_candidate_order_stable() -> None:
    example = {
        "example_id": "e1",
        "history": ["hist"],
        "target": "pos",
        "candidates": ["pos", "head", "tail"],
    }
    kwargs = {
        "example": example,
        "candidate_item_id": "tail",
        "target_item_id": "pos",
        "item_lookup": ITEMS,
        "train_popularity": {"hist": 5, "pos": 2, "head": 100, "tail": 1},
    }

    first = candidate_policy_target(candidate_item_ids=["pos", "head", "tail"], **kwargs)
    second = candidate_policy_target(candidate_item_ids=["tail", "pos", "head"], **kwargs)

    assert first == second
