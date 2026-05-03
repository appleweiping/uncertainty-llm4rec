# R3 OursMethod Error Decomposition and Offline Policy Refinement

## 1. Summary

This report uses only existing R3 MovieLens 1M artifacts under `outputs/runs/r3_movielens_1m_real_llm_full_candidate500_*`. No DeepSeek API calls were made, no prompts were changed, and no evaluator, split, or candidate protocol definitions were changed.

Ours full does not beat fallback-only/BM25 in R3. Across seeds `[13, 21, 42]`, Ours full has Recall@10 `0.0524`, while fallback-only/BM25 has Recall@10 `0.0591`.

Real DeepSeek confidence is badly miscalibrated under this protocol. Ours full has mean confidence `0.8510`, ECE `0.8510`, Brier `0.7253`, and `17,963` high-confidence wrong cases.

LLM direct generation/reranking are not useful under the current MovieLens candidate-500 protocol. Direct generative and confidence-observation paths are zero-hit; `llm_rerank_real` is also zero-hit in the saved R3 artifacts.

Current evidence supports an uncertainty-observation story more strongly than a method-improvement story. No paper claim that OursMethod improves over fallback-only is supported by R3.

The policy sweep is diagnostic-only, not paper evidence, because the available R3 artifacts are test artifacts. Seed `13` was used as a selection slice and seeds `21/42` as confirmation slices.

## 2. Decision-Level Attribution

| decision | count | Recall@10 | NDCG@10 | MRR@10 | mean confidence | ECE | Brier | high-conf wrong | delta hit@10 vs fallback | help | hurt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| accept | 1,533 | 0.0000 | 0.0000 | 0.0000 | 0.8478 | 0.8478 | 0.7194 | 1,499 | -122 | 0 | 122 |
| fallback | 16,237 | 0.0566 | 0.0275 | 0.0189 | 0.8511 | 0.8511 | 0.7255 | 16,114 | 0 | 0 | 0 |
| rerank | 350 | 0.0857 | 0.0464 | 0.0343 | 0.8589 | 0.8589 | 0.7384 | 350 | 0 | 0 | 0 |
| abstain | 0 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 0.0000 | 0 | 0 | 0 | 0 |
| Ours full | 18,120 | 0.0524 | 0.0255 | 0.0176 | 0.8510 | 0.8510 | 0.7253 | 17,963 | -122 | 0 | 122 |
| fallback-only | 18,120 | 0.0591 | 0.0292 | 0.0203 | 0.0000 | 0.0000 | 0.0000 | 0 | n/a | n/a | n/a |

Attribution:

- Accepted examples are the main source of harm: `1,533` accepted overrides, zero Recall@10, `122` cases where fallback hit and Ours missed, and no cases where Ours accepted output helped.
- Fallback decisions preserve fallback top-10 exactly and therefore have zero delta versus fallback-only.
- Rerank decisions in Ours full do not change top-10 versus fallback in the saved artifacts, so they do not help or hurt Ours full. Their higher within-slice Recall@10 comes from fallback/BM25 being stronger on that slice, not from LLM reranking.
- Ours full top-10 is identical to fallback-only for `91.54%` of examples.

Popularity:

- Accepted targets: head `906`, mid `466`, tail `161`.
- Accepted grounded/predicted items: head `1,384`, mid `138`, tail `11`, showing a strong head-item bias among accepted LLM-grounded outputs.
- Ours full target buckets: head `11,028`, mid `5,337`, tail `1,755`.
- Ours full grounded/predicted buckets: head `17,193`, mid `969`, tail `246`.

## 3. Confidence Threshold Sweep

The offline sweep used existing metadata only:

- `min_accept_confidence`: `[0.7, 0.8, 0.85, 0.9, 0.95]`
- `min_grounding_score`: `[0.7, 0.8, 0.9, 1.0]`
- accept on/off
- rerank on/off
- fallback-on-any-uncertainty
- candidate-adherent gates
- confidence plus candidate-adherent gates
- candidate-normalized gate when available

Key all-seed results:

| policy | overrides | override rate | Recall@10 | NDCG@10 | MRR@10 | high-conf wrong | tail Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fallback-only reference | 0 | 0.0000 | 0.0591 | 0.0292 | 0.0203 | 0 | 0.0923 |
| Ours full reference | 18,120 | 1.0000 | 0.0524 | 0.0255 | 0.0176 | 17,963 | 0.0792 |
| fallback-on-any-uncertainty | 0 | 0.0000 | 0.0591 | 0.0292 | 0.0203 | 0 | 0.0923 |
| candidate-adherent only | 1,539 | 0.0849 | 0.0524 | 0.0255 | 0.0176 | 1,505 | 0.0792 |
| candidate-adherent + conf >= 0.90 | 21 | 0.0012 | 0.0591 | 0.0292 | 0.0203 | 21 | 0.0923 |
| candidate-adherent + conf >= 0.95 | 12 | 0.0007 | 0.0591 | 0.0292 | 0.0203 | 12 | 0.0923 |

Confidence is not useful for selecting beneficial overrides in this R3 protocol. Raising confidence thresholds reduces the number of harmful overrides, but the surviving overrides are still high-confidence wrong. No swept accept/rerank policy exceeds fallback-only.

## 4. Fallback Parity Policy

The conservative policy is named `ours_conservative_uncertainty_gate` and is config-driven in `configs/methods/ours_conservative_uncertainty_gate.yaml`.

Policy:

```text
Use fallback ranking by default.
Only override fallback when the original Ours decision is accept, confidence >= 0.95,
grounding_score >= 1.0, candidate adherence passes, and candidate-normalized evidence exists.
Do not use rerank overrides.
```

All-seed offline result:

| policy | overrides | Recall@10 | NDCG@10 | MRR@10 | validity | hallucination | high-conf wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| fallback-only reference | 0 | 0.0591 | 0.0292 | 0.0203 | 1.0000 | 0.0000 | 0 |
| ours_conservative_uncertainty_gate | 12 | 0.0591 | 0.0292 | 0.0203 | 1.0000 | 0.0000 | 12 |

The conservative gate matches fallback-only on aggregate ranking metrics, validity, hallucination, and tail Recall@10. It does not exceed fallback-only. It still allows `12` high-confidence wrong overrides, although these do not change aggregate @10 metrics in the saved artifacts.

The only mechanically fallback-safe policy in this sweep is `fallback_on_any_uncertainty`, which keeps fallback top-10 unchanged for every example. The conservative gate should be treated as a candidate replay policy, not as a validated new research feature.

## 5. Rerank Audit

`llm_rerank_real` has zero Recall@10 in all saved R3 artifacts.

Audit result:

| seed | count | Recall@10 | parse success | candidate-adherent | all-zero-score rows | input-order top10 rows | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| 13 | 6,040 | 0.0000 | 61 (1.01%) | 6,040 (100%) | 5,982 | 5,927 | parser/truncated JSON failure |
| 21 | 6,040 | 0.0000 | 61 (1.01%) | 6,040 (100%) | 5,982 | 5,927 | parser/truncated JSON failure |
| 42 | 6,040 | 0.0000 | 61 (1.01%) | 6,040 (100%) | 5,982 | 5,927 | parser/truncated JSON failure |
| all | 18,120 | 0.0000 | 183 (1.01%) | 18,120 (100%) | 17,946 | 17,781 | parser/truncated JSON failure |

Findings:

- The evaluator reads `predicted_items` correctly.
- Candidate IDs are preserved and predictions are candidate-adherent.
- Most rerank rows collapse to original candidate order with all-zero scores.
- Raw outputs often contain a usable `ranked_items` array followed by a truncated trailing explanation, so the strict parser rejected the whole response.

A narrow parser fix was added: `parse_rerank_response` now recovers a closed `ranked_items` array from an otherwise truncated JSON object. This does not change the current R3 saved artifacts; a cache replay is required before rerank metrics can be re-evaluated.

## 6. Case Studies

Exported files:

- `outputs/tables/r3_case_studies_high_conf_wrong.csv`: 50 rows
- `outputs/tables/r3_case_studies_accept_hurts.csv`: 50 rows
- `outputs/tables/r3_case_studies_accept_helps.csv`: 0 rows
- `outputs/tables/r3_case_studies_fallback_saves.csv`: 50 rows
- `outputs/tables/r3_case_studies_tail_underconfident.csv`: 0 rows

The empty accept-help file is informative: among accepted examples in the saved R3 artifacts, no case improved Recall@10 over fallback-only. The empty tail-underconfident file is also consistent with the confidence pathology: the model is generally overconfident, not underconfident, including on tail targets.

## Interpretation

Evidence supported by R3 refinement:

- Observation-first story: supported.
- Fallback-safe uncertainty gate story: partially supported. A no-override gate exactly preserves fallback; the named conservative gate matches fallback aggregate metrics but still admits a small number of high-confidence wrong overrides.
- Method-improvement story: not supported.

Recommended config-driven change:

- Keep OursMethod core unchanged.
- Add `ours_conservative_uncertainty_gate` only as an offline/cache-replay policy candidate.
- Do not claim improvement unless a subsequent cache replay or real run shows improvement beyond fallback-only.

Next recommended action:

Run R3b conservative policy real-LLM replay from cache.

## R3b conservative gate cache replay (observation-first pivot)

Config: `configs/experiments/r3b_movielens_1m_conservative_gate_cache_replay.yaml`.

Requirements:

- Same MovieLens 1M R2 processed data and candidate-500 protocol as R3
  (`include_target: true`, candidate seed `13` on disk).
- `llm.cache.require_hit: true` and `safety.allow_api_calls: false` (no DeepSeek
  calls; no `DEEPSEEK_API_KEY` required).
- Methods: `bm25`, `ours_fallback_only`, `ours_uncertainty_guided_real`,
  `ours_conservative_uncertainty_gate`, `ours_ablation_no_uncertainty`,
  `llm_generative_real`.
- Seeds `[13, 21, 42]`; stop with error if any cache entry is missing.

After successful runs:

```text
python scripts/export_tables.py --input outputs/runs --output outputs/tables
python scripts/aggregate_runs.py --input outputs/runs --output outputs/tables
python scripts/export_r3b_tables.py --runs outputs/runs --output outputs/tables
```

Expected tables: `r3b_conservative_gate_main.csv`,
`r3b_conservative_gate_ablation.csv`, `r3b_conservative_gate_decision_stats.csv`,
`r3b_observation_failures.csv`. Populate numeric claims in this section **only**
from those files (or from `metrics.json` per run), never from memory.

