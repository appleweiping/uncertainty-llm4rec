# Paper Results Summary

Artifact base commit: `6b5301afeab1c69051ac23dd8cfcfd3bc779986a`

## Dataset Summaries

- MovieLens 1M: candidate_size=500, panel_size=15, subset_size=200 per seed, seeds [13,21,42].
- Amazon Beauty: local Amazon Reviews 2023 All_Beauty, candidate_size=479 because the local catalog has 479 items, panel_size=15, subset_size=200 per seed, seeds [13,21,42].

## Method Summaries

- Free-form LLM title generation and verbalized confidence are retained as negative/motivating evidence from existing MovieLens R3 artifacts.
- CU-GR v2 uses candidate-local listwise LLM preferences over anonymous panels and calibrated fusion with fallback ranking.
- Fusion weights are selected on seed21 validation after seed13 training and evaluated on seed42.

## Key Results

- MovieLens 1M seed42: CU-GR v2 fusion NDCG@10=0.10606 vs fallback=0.049339.
- Amazon Beauty seed42: CU-GR v2 fusion NDCG@10=0.154217 vs fallback=0.142486.

## Key Table References

- `outputs/tables/paper_main_results.csv`
- `outputs/tables/paper_ablation.csv`
- `outputs/tables/paper_uncertainty_observation.csv`
- `outputs/tables/paper_panel_analysis.csv`
- `outputs/tables/paper_cost_latency.csv`
- `outputs/tables/figure_calibration_reliability.csv`
- `outputs/tables/figure_risk_coverage.csv`
- `outputs/tables/figure_delta_vs_fallback_by_dataset.csv`
- `outputs/tables/figure_swap_outcomes.csv`
- `outputs/tables/figure_panel_coverage.csv`
- `outputs/tables/cu_gr_v2_full_seed_main.csv`
- `outputs/tables/cu_gr_v2_amazon_beauty_main.csv`

## Claim Boundary

The artifacts support an observation-motivated method framing across MovieLens 1M and Amazon Beauty. They do not support claims about full-ranking evaluation, more than two domains, local open-source LLMs, or production-scale inference.
