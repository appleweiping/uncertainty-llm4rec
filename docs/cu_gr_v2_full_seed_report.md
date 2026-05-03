# CU-GR v2 Full Seed Report

Run name: `r3_v2_movielens_preference_signal_subgate_full_seeds`

## Verdict

PASS: CU-GR v2 supports the method story on held-out seed42.

Held-out seed42 `fusion_train_best` improves over `fallback_only` by +0.05672119 NDCG@10, with harmful swap rate 0.035 and parser success rate 0.99.

## Provider

DeepSeek OpenAI-compatible API:

- model: `deepseek-v4-flash`
- base URL: `https://api.deepseek.com`
- API key source: `DEEPSEEK_API_KEY`
- JSON response format enabled
- cache/resume enabled
- API key value was not inspected, printed, logged in this report, or committed

## Dataset / Protocol

- dataset: MovieLens R3, `data/processed/movielens_1m/r2_full_single_dataset`
- split/evaluator lineage: existing R3 artifacts
- candidate protocol: sampled, candidate_size=500, include_target=true
- panel_size: 15
- subset_size/max_examples: 200 per seed
- seeds: 13, 21, 42
- prompt panel policy: anonymous A/B/C labels, no target title as target, no global target ID in prompt

## Commands Run

- `py -3 scripts/validate_experiment_ready.py --config configs/experiments/r3_v2_movielens_preference_full_seeds.yaml`
- `py -3 scripts/list_required_artifacts.py --config configs/experiments/r3_v2_movielens_preference_full_seeds.yaml`
- `git diff --check`
- `py -3 scripts/run_all.py --config configs/experiments/r3_v2_movielens_preference_full_seeds.yaml`
- `py -3 scripts/build_preference_dataset.py --runs outputs/runs --output outputs/tables/cu_gr_v2_preference_dataset.csv`
- `py -3 scripts/train_preference_fusion.py --input outputs/tables/cu_gr_v2_preference_dataset.csv --output outputs/models/cu_gr_v2_preference_fusion`
- `py -3 scripts/replay_preference_fusion.py --input outputs/tables/cu_gr_v2_preference_dataset.csv --model outputs/models/cu_gr_v2_preference_fusion --output outputs/tables`
- `py -3 scripts/export_tables.py --input outputs/runs --output outputs/tables`
- `py -3 scripts/aggregate_runs.py --input outputs/runs --output outputs/tables`
- `py -3 -m pytest tests/unit/test_candidate_panel.py`
- `py -3 -m pytest tests/unit/test_preference_prompt.py`
- `py -3 -m pytest tests/unit/test_preference_parser.py`
- `py -3 -m pytest --basetemp outputs/pytest-basetemp`

## Test Results

- candidate panel tests: 3 passed
- preference prompt test: 1 passed
- preference parser tests: 4 passed
- full regression: 318 passed

The first full pytest attempts hit Windows temp-directory permission errors, then passed after running with an explicit pytest base temp directory and elevated permission for pytest temp cleanup.

## Parser Stats

| seed | parse_success_rate | invalid_label_rate | duplicate_label_rate | partial_ranking_rate |
|---:|---:|---:|---:|---:|
| 13 | 0.99 | 0.005 | 0.005 | 0.0 |
| 21 | 0.99 | 0.005 | 0.005 | 0.0 |
| 42 | 0.99 | 0.005 | 0.005 | 0.0 |
| aggregate | 0.99 | 0.005 | 0.005 | 0.0 |

## Panel Coverage

| seed | target_in_panel_rate | fallback target rank in panel | LLM target rank in panel | LLM top1 panel hit rate | panel oracle NDCG upper bound |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.30 | 10.3167 | 5.3000 | 0.1167 | 0.3015 |
| 21 | 0.30 | 10.4333 | 5.9833 | 0.1000 | 0.3015 |
| 42 | 0.30 | 10.3333 | 4.9500 | 0.1167 | 0.3015 |
| aggregate | 0.30 | 10.3611 | 5.4111 | 0.1111 | 0.3015 |

## Main Results

Aggregate over seeds 13/21/42:

| method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 | delta NDCG vs fallback |
|---|---:|---:|---:|---:|---:|
| fallback_only | 0.0900 | 0.049339 | 0.036889 | 0.0900 | 0.000000 |
| llm_listwise_panel | 0.2433 | 0.131868 | 0.096867 | 0.2433 | +0.082529 |
| fusion_fixed_grid | 0.1217 | 0.068136 | 0.051827 | 0.1217 | +0.018796 |
| fusion_train_best | 0.1933 | 0.103809 | 0.076615 | 0.1933 | +0.054470 |
| safe_fusion | 0.1750 | 0.089869 | 0.063895 | 0.1750 | +0.040529 |

## Held-out Seed42 Result

| method | Recall@10 | NDCG@10 | MRR@10 | HitRate@10 | delta NDCG vs fallback |
|---|---:|---:|---:|---:|---:|
| fallback_only | 0.0900 | 0.049339 | 0.036889 | 0.0900 | 0.000000 |
| llm_listwise_panel | 0.2500 | 0.138267 | 0.102788 | 0.2500 | +0.088928 |
| fusion_fixed_grid | 0.1150 | 0.064796 | 0.049484 | 0.1150 | +0.015456 |
| fusion_train_best | 0.2000 | 0.106060 | 0.077544 | 0.2000 | +0.056721 |
| safe_fusion | 0.1800 | 0.093082 | 0.066450 | 0.1800 | +0.043743 |

Success criteria:

- fusion_train_best NDCG@10 > fallback_only NDCG@10: pass
- delta_NDCG@10 > 0.01 absolute: pass
- harmful_swap_rate <= 0.05: pass, 0.035
- parser_success_rate >= 0.95: pass, 0.99
- target_in_panel_rate reported and non-trivial: pass, 0.30

## Fusion Weights

`fusion_train_best` was selected on seed21 validation, with seed42 held out:

- alpha: 0.5
- beta: 0.7
- gamma: 0.2
- lambda: 0.05
- train seed13 NDCG@10: 0.10727346
- validation seed21 NDCG@10: 0.09809310
- validation harmful swap rate: 0.035
- test seed42 NDCG@10: 0.10606046
- test seed42 delta vs LLM-listwise NDCG@10: -0.03220651
- test seed42 delta vs R3 Ours v1 NDCG@10: +0.05964648

`safe_fusion` thresholds were tuned on seed21 only:

- margin: 0.05
- confidence_min: 0.5
- seed42 NDCG@10: 0.09308180

## Swap Analysis

Held-out seed42:

| method | beneficial | harmful | neutral | harmful_swap_rate | avg NDCG delta | top10_changed_rate |
|---|---:|---:|---:|---:|---:|---:|
| llm_listwise_panel | 41 | 12 | 147 | 0.060 | +0.088928 | 0.99 |
| fusion_fixed_grid | 15 | 6 | 179 | 0.030 | +0.015456 | 0.99 |
| fusion_train_best | 32 | 7 | 161 | 0.035 | +0.056721 | 0.99 |
| safe_fusion | 22 | 4 | 174 | 0.020 | +0.043743 | 1.00 |

Aggregate `fusion_train_best`: 91 beneficial / 20 harmful / 489 neutral, harmful_swap_rate 0.033333.

## Cost / Latency

| seed | live_requests | cache_hits | total_tokens | effective_cost_usd | p50_latency_seconds | p95_latency_seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 0 | 200 | 306041 | 0.05890682 | 0.0113 | 0.0264 |
| 21 | 200 | 0 | 306029 | 0.05889576 | 7.5642 | 8.3179 |
| 42 | 200 | 0 | 306653 | 0.05908322 | 7.3745 | 7.9506 |
| aggregate | 400 | 200 | 918723 | 0.17688580 | 4.9833 | 5.4316 |

Retry, timeout, and 429 counts were all recorded as 0.

## Interpretation

method story revived.

The result supports the claim under test: free-form direct generation and verbal confidence remain problematic, but candidate-local listwise preference signals are useful, and calibrated fusion improves the fallback on held-out seed42 without exceeding the harmful swap constraint.

## Artifacts

- `outputs/tables/cu_gr_v2_full_seed_main.csv`
- `outputs/tables/cu_gr_v2_full_seed_by_seed.csv`
- `outputs/tables/cu_gr_v2_full_seed_fusion_weights.csv`
- `outputs/tables/cu_gr_v2_full_seed_swap_analysis.csv`
- `outputs/tables/cu_gr_v2_full_seed_parser_stats.csv`
- `outputs/tables/cu_gr_v2_full_seed_panel_coverage.csv`
- `outputs/tables/cu_gr_v2_full_seed_cost_latency.csv`
- `outputs/tables/cu_gr_v2_full_seed_failure_cases.csv`
- `outputs/tables/cu_gr_v2_preference_dataset.csv`
- `outputs/models/cu_gr_v2_preference_fusion/model.json`

## Next Recommended Action

Run CU-GR v2 on a second dataset/domain.
