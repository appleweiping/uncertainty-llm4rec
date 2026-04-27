# Framework-Observation-Day1d Logit-Based Confidence Report

## Scope

Pending server run. Day1d will run local Qwen-LoRA on Beauty valid/test 200/200 only. It does not train, use evidence, use CEP, call external APIs, or run four domains.

## Method

The prompt asks only for a binary `recommend` JSON decision. The script extracts confidence from model token probabilities for `true` versus `false` at the `{"recommend": ...}` position.

Implementation note: Day1d uses constrained true/false continuation scoring rather than free-form generation. This avoids an extra generation pass and keeps the score definition tied directly to token probabilities.

Two scores are reported:

- `positive_relevance_score = P(recommend=true)` for relevance / label prediction.
- `decision_confidence = max(P(true), P(false))` for confidence in the chosen binary decision.

## Status

Awaiting Day1d server smoke results.
