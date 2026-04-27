# Framework-Observation-Day1e Self-Consistency Plan

## Trigger

Run Day1e only if Day1d logit-based confidence still has weak relevance/correctness signal or unstable calibration. Do not run Day1e before reading Day1d Beauty 200/200 results.

## Scope

- Local Qwen-LoRA only.
- Beauty only.
- Smoke size: valid/test 100/100 because sampling is more expensive.
- No training, no evidence, no CEP, no external APIs, no four-domain run.

## Method

For each user-candidate pair, sample the same binary recommendation prompt `n=5` or `n=10` times with controlled stochastic decoding.

Derived scores:

- `recommend_true_frequency`: fraction of samples voting `recommend=true`; use as the relevance score.
- `majority_vote_rate`: max fraction of true/false votes; use as decision confidence.
- `uncertainty`: `1 - majority_vote_rate`.

## Diagnostics

Report parse/schema rate, recommend true rate, accuracy, AUROC/Brier/ECE for `recommend_true_frequency` against label, and AUROC/Brier/ECE for `majority_vote_rate` against decision correctness.

## Decision Rule

If self-consistency improves ranking/relevance signal or decision correctness calibration over Day1d, consider a larger Beauty smoke. If it remains weak, the next route should be pairwise/listwise context rather than more confidence elicitation.
