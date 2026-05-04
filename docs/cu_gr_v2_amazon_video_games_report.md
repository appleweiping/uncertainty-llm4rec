# CU-GR v2 Amazon Video Games Report

Run name: `r3_v2_amazon_video_games_preference_full_seeds`

## Verdict

MAJOR FIXES REQUIRED. Amazon Video Games did not pass the third-domain held-out gate because seed42 fusion_train_best NDCG@10 improved by 0.00813984 over fallback, below the required +0.01 absolute threshold.

## Provider

DeepSeek OpenAI-compatible API, model `deepseek-v4-flash`; cache/resume enabled. API key value was not inspected or written.

## Dataset / Protocol

Amazon Reviews 2023 Video Games local processed domain, processed directory `data/processed/amazon_reviews_2023_video_games/cu_gr_v2`, sampled candidate_size=500, include_target=true, panel_size=15, subset_size=200 per seed, seeds `[13, 21, 42]`.

Panel feasibility passed before API calls by oracle-gain evidence. Panel size 15 had target_in_panel_rate 0.195 on seeds 13/21 and 0.190 on seed42. The coverage lift over fallback_hit@10 was below +0.03, but oracle NDCG gain was clearly positive.

## Fusion Weights

Selected alpha=0.7, beta=0.1, gamma=0.2, lambda=0.0 on seed21 validation. Safe fusion thresholds: margin=0.05, confidence_min=0.5.

## Held-out Seed42

- fallback_only NDCG@10=0.09606638
- llm_listwise_panel NDCG@10=0.08813102
- fusion_train_best NDCG@10=0.10420622
- delta vs fallback=+0.00813984
- harmful_swap_rate=0.020000
- parser_success_rate=0.975
- target_in_panel_rate=0.190

The run satisfies parser, harmful-swap, and panel-reporting requirements, but fails the required delta_NDCG@10 > 0.01 criterion.

## Cost / Latency

The run used 600 live requests, 972258 total tokens, estimated effective cost 0.18421452 USD, p50 latency 7.6860 seconds, p95 latency 8.2973 seconds, and no retries, timeouts, or 429s.

## Interpretation

MovieLens and Amazon Beauty remain validated method-supporting domains. Amazon Video Games is third-domain diagnostic evidence showing that CU-GR v2 is promising but domain-dependent under the current panel/fusion design.

## Artifacts

- `outputs/tables/cu_gr_v2_amazon_video_games_main.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_by_seed.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_fusion_weights.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_swap_analysis.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_parser_stats.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_panel_coverage.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_cost_latency.csv`
- `outputs/tables/cu_gr_v2_amazon_video_games_failure_cases.csv`

## Next Recommended Action

Refine fusion model.
