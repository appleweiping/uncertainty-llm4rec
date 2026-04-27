# Framework-Observation-Day1f Self-Consistency Plan

## Trigger

Run this only if Day1e shows that logit P(true) is still weak as a ranking/relevance signal or the go/no-go remains ambiguous.

## Scope

- Beauty 100/100 smoke only.
- Local Qwen-LoRA only.
- No training, no evidence, no CEP, no external APIs, no four-domain run.

## Method

For each user-candidate pair, sample the binary recommendation prompt `n=5` or `n=10` times with stochastic decoding.

Derived scores:

- `recommend_true_frequency`: fraction of samples voting `recommend=true`; use as relevance score.
- `decision_confidence`: `majority_vote_rate = max(true_votes, false_votes) / n`.
- `uncertainty`: `1 - majority_vote_rate`.

## Evaluation

Compare self-consistency against Day1d/Day1e logit P(true):

- AUROC, Brier, ECE for `recommend_true_frequency` against label.
- AUROC, Brier, ECE for `decision_confidence` against decision correctness.
- User-level NDCG/MRR/HR using `recommend_true_frequency` as ranking score.

Do not run self-consistency full until the 100/100 smoke beats or clarifies logit P(true).
