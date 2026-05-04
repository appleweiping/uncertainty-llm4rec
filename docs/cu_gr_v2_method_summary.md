# CU-GR v2 Method Summary

Working method name: CU-GR v2: Candidate-Normalized Calibrated Preference Fusion for Generative Recommendation.

## Input

Each example contains a user history, a held-out target item, and a sampled candidate set generated under the shared R3 protocol.

## Candidate Panel

The panel is built locally from valid candidate items only. It includes high fallback ranks, mid-rank contrasts, popularity/tail contrasts, optional sequential candidates when available, and deterministic fills. Panel items are shown with anonymous labels A/B/C/... rather than global item identifiers.

## Listwise Preference Prompt

The prompt asks the LLM to rank candidate-local panel labels for the user's next-item preference. It does not identify the target item and does not expose a global target item ID.

## Parser

The parser accepts JSON listwise responses, maps labels back to panel item IDs, rejects invalid labels, rejects duplicate labels, tracks partial rankings, and preserves raw outputs for audit.

## Fusion Formula

`score = alpha * normalized_fallback_score + beta * normalized_llm_score + gamma * llm_confidence - lambda * popularity_penalty`.

MovieLens selected `alpha=0.5, beta=0.7, gamma=0.2, lambda=0.05`; Amazon Beauty selected `alpha=0.5, beta=0.3, gamma=0.0, lambda=0.1`.

## Train / Validation / Test Split

Fusion grid selection is trained on seed13, validated on seed21, and reported on held-out seed42. The grid is not selected on seed42.

## Safety Constraints

Safety analysis tracks parse success, invalid labels, duplicate labels, candidate adherence, harmful swaps, and safe-fusion threshold behavior. The validation constraint requires harmful_swap_rate <= 0.05.

## Inference Algorithm

1. Rank the full candidate set with fallback BM25.
2. Build a deterministic candidate-local panel.
3. Query the LLM with anonymous labels in JSON mode.
4. Parse and validate label rankings.
5. Normalize fallback and LLM panel scores.
6. Fuse scores with validation-selected weights.
7. Replace panel ordering inside the fallback ranking while non-panel candidates retain fallback order.
8. Evaluate with the shared ranking evaluator.
