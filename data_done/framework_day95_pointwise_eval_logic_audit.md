# Framework-Day9.5 Pointwise Eval Logic Audit

## Score Source

The Day9 pointwise evaluator uses `relevance_score` parsed from model output if present; otherwise it converts parsed `relevance_label` into score `1.0` or `0.0`.

## Parse Failure Fallback

In the original Day9 evaluator, `parse_relevance()` returns score `0.0` on parse failure. This means failed candidates are demoted, not promoted.

## Tie-Breaker

The original Day9 evaluator sorts by `(-score, local_idx)`, where `local_idx` is the candidate's original order within the 6-candidate group. This is unsafe if candidate order is label-biased.

## Candidate Order

The pointwise files are grouped in candidate-pool order. Current local audit shows the test positive-at-position-1 rate is `1.0`. If this is high, a tie or parse-failure-heavy evaluator that preserves original order can produce inflated HR@1/NDCG.

## Label Leakage Status

Static code inspection found no direct use of `target_label` during ranking score construction, but the original tie-breaker can leak through order bias if positives are consistently first. Day9.5 independent safe eval therefore replaces original-order tie-break with lexical/random-neutral tie-break and recomputes metrics from prediction outputs when server predictions are available.
