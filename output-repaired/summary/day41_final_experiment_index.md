# Day41 Final Experiment Index

## Week1-Week4 / raw confidence observation

- key_files: `output-repaired/summary/day29b_beauty_multimodel_raw_confidence_diagnostics.csv; output-repaired/summary/day29b_observation_raw_llm_confidence_miscalibration_report.md`
- main_claim: Raw verbalized confidence/relevance signals are informative but miscalibrated across multiple LLMs; they should not be used directly as probabilities.
- claim_level: `primary_observation`
- limitations: Observation is strongest on Beauty diagnostics; raw confidence is not evaluated as a final recommender by itself.

## Day6 decision confidence repair

- key_files: `output-repaired/summary/day26_final_claim_map.md; prior Day6 report referenced in day26_experiment_index.md`
- main_claim: Evidence-risk can strongly support yes/no decision reliability reranking.
- claim_level: `primary_method`
- limitations: Decision reliability is a different setting from candidate relevance posterior ranking.

## Day9 full Beauty candidate relevance posterior

- key_files: `output-repaired/summary/day29b_beauty_relevance_probability_diagnostics.csv; output-repaired/summary/day26_final_claim_map.md`
- main_claim: Raw candidate relevance probability is miscalibrated; valid-set calibration produces a useful calibrated relevance posterior.
- claim_level: `primary_method`
- limitations: Calibration improves probability quality; AUROC is not expected to transform dramatically.

## Day10 list-level first-pass boundary

- key_files: `output-repaired/summary/day26_final_claim_map.md; output-repaired/summary/day26_paper_results_section_draft.md`
- main_claim: Evidence decomposition is better as post-hoc/hybrid decision support than as a heavy first-pass generation burden.
- claim_level: `boundary_analysis`
- limitations: Boundary is about first-pass list generation, not CEP plug-in reranking.

## Day20/23/25 Beauty full three-backbone multi-seed

- key_files: `output-repaired/summary/day26_three_backbone_external_plugin_main_table_metric_repaired.csv; output-repaired/summary/day26_component_attribution_summary_metric_repaired.csv`
- main_claim: CEP plug-in improves NDCG/MRR across SASRec-style, GRU4Rec, and Bert4Rec full Beauty multi-seed.
- claim_level: `primary_performance`
- limitations: Beauty full is primary evidence but not universal SOTA across all domains/backbones.

## Day29b observation consolidation

- key_files: `output-repaired/summary/day29b_beauty_multimodel_calibration_effect.csv; output-repaired/summary/day29b_paper_motivation_snippet.md`
- main_claim: Multi-model diagnostics support the motivation that raw LLM confidence is informative but unreliable without calibration.
- claim_level: `primary_observation`
- limitations: Aggregates existing outputs; does not add new model inference.

## Day29c backbone score calibration diagnostic

- key_files: `output-repaired/summary/day29c_backbone_score_calibration_diagnostics.csv; output-repaired/summary/day29c_backbone_score_miscalibration_report.md`
- main_claim: Backbone scores are useful ranking logits but are not calibrated probabilities.
- claim_level: `diagnostic_only`
- limitations: Diagnostic does not claim backbone methods fail; it separates ranking ability from uncertainty estimation.

## Day30 CEP robustness

- key_files: `output-repaired/summary/day30_cep_robustness_metrics.csv; output-repaired/summary/day30_sasrec_cep_robustness_grid.csv; output-repaired/summary/day30_cep_backbone_robustness_report.md`
- main_claim: CEP and SASRec+CEP remain useful under controlled input perturbations; D degradation is bounded in the 500-user robustness setting.
- claim_level: `robustness_support`
- limitations: Beauty 500-user robustness first run, primarily SASRec.

## Day31/37/39 small-domain CEP calibration

- key_files: `output-repaired/summary/day31_movies_medium5_calibration_comparison.csv; output-repaired/summary/day37_movies_small_calibration_comparison.csv; output-repaired/summary/day39_books_electronics_small_calibration_comparison.csv`
- main_claim: Movies/books/electronics small results support cross-domain calibration consistency and directionality.
- claim_level: `cross_domain_sanity`
- limitations: Small-domain candidate pool is 6, so HR@10 is trivial; backbone fallback caveats apply.

## Day38/40 small-domain fallback sensitivity

- key_files: `output-repaired/summary/day40_small_domains_fallback_sensitivity_summary.csv; output-repaired/summary/day40_books_electronics_fallback_sensitivity_report.md`
- main_claim: Small-domain gains are not explained by fallback flag alone, but many are best interpreted as fallback/cold compensation or sample-limited directionality.
- claim_level: `cross_domain_sanity`
- limitations: Do not describe small-domain results as fully healthy ID-backbone benchmarks.

## Day34/35 regular-medium cold-start/content-carrier route

- key_files: `output-repaired/summary/day34_movies_cold_content_carrier_report.md; output-repaired/summary/day35_cross_domain_route_decision_report.md`
- main_claim: Regular medium domains reveal real cold-start issues for ID-only sequential backbones and motivate content-aware/cold-aware carriers.
- claim_level: `boundary_analysis`
- limitations: Content carrier is diagnostic/cold-aware, not a SOTA recommender claim.
