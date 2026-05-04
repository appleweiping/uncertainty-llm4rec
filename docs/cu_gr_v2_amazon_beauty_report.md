# CU-GR v2 Amazon Beauty Second-Domain Gate Report

## Verdict

PASS: CU-GR v2 generalizes to the Amazon Beauty second-domain gate under the configured held-out seed42 criteria.

Held-out seed42:

- fallback_only NDCG@10 = 0.14248601
- fusion_train_best NDCG@10 = 0.15421699
- delta NDCG@10 = +0.01173098
- harmful_swap_rate = 0.015
- parser_success_rate = 0.95
- target_in_panel_rate = 0.18

## Domain Selected

Amazon Reviews 2023 All_Beauty, converted from the local prepared observation artifacts at `data/processed/amazon_reviews_2023_beauty/full`.

No data was downloaded. The local catalog has 479 items, so the requested candidate_size=500 was reduced to the largest feasible target-included candidate pool, candidate_size=479.

## Dataset / Protocol

- dataset: `amazon_reviews_2023_beauty_cu_gr_v2`
- processed_dir: `data/processed/amazon_reviews_2023_beauty/cu_gr_v2`
- item_count: 479
- interactions: 3315
- examples: 2244
- split_counts: train 1795, val 224, test 225
- subset_size: 200 per seed
- seeds: [13, 21, 42]
- candidate_protocol: sampled, include_target=true
- candidate_size_requested: 500
- candidate_size_effective: 479
- panel_size: 15
- train/tune/test: seed13 train, seed21 validation, seed42 held-out test

## Provider

DeepSeek OpenAI-compatible API:

- model: `deepseek-v4-flash`
- base_url: `https://api.deepseek.com`
- JSON response format enabled
- cache/resume enabled
- raw outputs, prompt hashes, token usage, latency, and cache keys preserved
- API key value was not printed, inspected, or committed

## Panel Feasibility

Panel_size=15 passed the offline feasibility gate by positive oracle NDCG gain:

- target_in_panel_rate = 0.18
- fallback_hit@10 = 0.175
- coverage minus fallback_hit@10 = +0.005
- oracle NDCG@10 upper bound = 0.18
- oracle gain vs fallback = +0.03751399

The coverage margin alone did not reach +0.03, and oracle beneficial swap opportunities were limited. The gate proceeded because oracle gain was clearly positive.

## Main Results

Aggregate across seeds:

| method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 | delta NDCG vs fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| fallback_only | 0.175000 | 0.142486 | 0.132310 | 0.175000 | 0.000000 |
| llm_listwise_panel | 0.163333 | 0.135149 | 0.126403 | 0.163333 | -0.007337 |
| fusion_fixed_grid | 0.175000 | 0.147209 | 0.138393 | 0.175000 | +0.004723 |
| fusion_train_best | 0.175000 | 0.152265 | 0.144893 | 0.175000 | +0.009779 |
| safe_fusion | 0.161667 | 0.145336 | 0.140342 | 0.161667 | +0.002850 |

Held-out seed42:

| method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 | delta NDCG vs fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| fallback_only | 0.175000 | 0.142486 | 0.132310 | 0.175000 | 0.000000 |
| llm_listwise_panel | 0.160000 | 0.134076 | 0.125917 | 0.160000 | -0.008410 |
| fusion_fixed_grid | 0.175000 | 0.148086 | 0.139548 | 0.175000 | +0.005600 |
| fusion_train_best | 0.175000 | 0.154217 | 0.147548 | 0.175000 | +0.011731 |
| safe_fusion | 0.160000 | 0.145535 | 0.141125 | 0.160000 | +0.003049 |

References available from local artifacts:

- BM25_reference equals fallback_only.
- sequential_markov_reference Recall@10/NDCG@10/MRR@10 = 0.0/0.0/0.0 on this subset.
- popularity_reference Recall@10/NDCG@10/MRR@10 = 0.0/0.0/0.0 on this subset.

## Fusion Weights

Selected by seed21 validation, not by seed42:

- alpha = 0.5
- beta = 0.3
- gamma = 0.0
- lambda = 0.1
- train seed13 NDCG@10 = 0.15166766
- validation seed21 NDCG@10 = 0.15090912
- test seed42 NDCG@10 = 0.15421699

Safe fusion thresholds selected on seed21:

- margin = 0.03
- confidence_min = 0.5

## Parser Stats

| seed | parser_success_rate | invalid_label_rate | duplicate_label_rate | partial_ranking_rate | confidence_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.955000 | 0.040000 | 0.005000 | 0.000000 | 0.522593 |
| 21 | 0.955000 | 0.040000 | 0.005000 | 0.000000 | 0.528953 |
| 42 | 0.950000 | 0.045000 | 0.005000 | 0.000000 | 0.519004 |
| aggregate | 0.953333 | 0.041667 | 0.005000 | 0.000000 | 0.523524 |

## Panel Coverage

| seed | target_in_panel_rate | fallback target rank in panel | LLM target rank in panel | LLM top1 panel hit rate | panel oracle NDCG upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 0.180000 | 2.558824 | 2.794118 | 0.735294 | 0.180000 |
| 21 | 0.180000 | 2.352941 | 2.558824 | 0.764706 | 0.180000 |
| 42 | 0.180000 | 2.558824 | 2.970588 | 0.735294 | 0.180000 |
| aggregate | 0.180000 | 2.490196 | 2.774510 | 0.745098 | 0.180000 |

## Swap Analysis

Held-out seed42:

- llm_listwise_panel: beneficial/harmful/neutral = 7/9/184, harmful_swap_rate=0.045
- fusion_fixed_grid: beneficial/harmful/neutral = 9/3/188, harmful_swap_rate=0.015
- fusion_train_best: beneficial/harmful/neutral = 9/3/188, harmful_swap_rate=0.015
- safe_fusion: beneficial/harmful/neutral = 7/4/189, harmful_swap_rate=0.020

Aggregate fusion_train_best:

- beneficial/harmful/neutral = 25/8/567
- harmful_swap_rate = 0.013333
- avg NDCG delta per example = +0.009779
- top10_changed_rate = 0.95

## Cost / Latency

Aggregate:

- live_requests = 600
- cache_hits = 0
- total_tokens = 1,287,713
- effective_cost_usd = 0.22858864
- p50_latency_seconds = 7.551839
- p95_latency_seconds = 8.395560
- retry_count = 0
- timeout_count = 0
- rate_limit_429_count = 0

## Leakage / Fairness Audit

- Candidate panels contain only items from the per-example candidate set.
- Prompts use anonymous labels A/B/C/... and do not expose the held-out target as a target.
- The global target item ID is not included as target metadata in the prompt.
- Histories are sourced from prepared past-interaction observations.
- Candidate construction excludes history items except the held-out target and records this in the manifest.
- The same evaluator and prediction schema were used for fallback, BM25, fusion policies, and available references.
- No MockLLM outputs were used as evidence.

## Interpretation

method story strengthened

The second-domain result validates the calibrated fusion part of CU-GR v2 on held-out Amazon Beauty seed42. The raw listwise panel policy alone underperforms fallback on this domain, so the evidence supports candidate-local preference signals only when calibrated through fusion, not as an unconstrained standalone reranker.

## Artifacts

- `outputs/tables/cu_gr_v2_amazon_beauty_main.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_by_seed.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_fusion_weights.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_swap_analysis.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_parser_stats.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_panel_coverage.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_cost_latency.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_failure_cases.csv`

## Next Recommended Action

Prepare paper main experiment tables.
