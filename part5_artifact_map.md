# Part5 Artifact Map

This map records the compact week5-week6 Part5 evidence bundle. It is intentionally an artifact guide rather than a full experiment diary: the goal is to make the main entrances, tables, figures, notes, and reports traceable without exposing every implementation detail.

## Main Entrances

`main_build_multitask_samples.py` builds the shared pointwise, candidate-ranking, and pairwise-preference inputs. In the final Part5 version it also supports coverage-oriented pairwise sample generation through configurable pair generation mode and per-event pair limits.

`main_rank_rerank.py` applies uncertainty-aware ranking decisions to candidate-ranking outputs. For the current compact evidence version, `nonlinear_structured_risk_rerank` is the default current best family; `local_margin_swap_rerank` and `structured_risk_plus_local_margin_swap_rerank` remain retained exploratory families.

`main_pairwise_rank.py` turns pairwise preference predictions into ranking candidates. It supports the weighted mechanism line and a plain win-count baseline used for non-uncertainty comparison, while pairwise itself remains mechanism-layer evidence rather than the main ranking replacement.

`main_compare_multitask.py --finalize_part5` refreshes the final Part5 multitask result table and narrative summary.

`main_part5_artifacts.py --build_figures --build_pairwise_coverage` rebuilds the paper-facing Part5 figure pack and pairwise coverage evidence pack from existing summary artifacts.

`main_literature_baseline.py` runs the compact candidate-ranking baseline group over the same candidate set, split, and metrics.

`src/analysis/aggregate_cross_domain_structured_risk_results.py` consolidates the four-domain direct ranking vs structured-risk current best comparison.

`src/analysis/build_pairwise_coverage_upgrade.py` builds the final pairwise coverage upgrade table, plot source, figure, and notes.

`src/analysis/aggregate_literature_baseline_compare.py` consolidates the baseline group, direct ranking, and structured-risk current best family into one defensive compare table.

`src/analysis/build_part5_consolidated_tables.py` builds the paper-ready Part5 table skeletons for single-domain, four-domain, and pairwise-boundary writing.

## Main Tables

`outputs/summary/part5_multitask_final_results.csv` is the original Part5 final multitask result table. It is the method-closure table for pointwise diagnosis, candidate ranking, pairwise mechanism evidence, same-task baseline, and retained variants.

`outputs/summary/week6_final_4domain_structured_risk_compare.csv` is the four-domain structured-risk landing table. It compares direct DeepSeek candidate ranking and structured-risk current best reranking for Movies, Beauty, Books, and Electronics.

`outputs/summary/week6_final_pairwise_coverage_upgrade.csv` is the final pairwise coverage table. It records total/supported ranking events, supported event fraction, pair coverage, overlap/expanded metrics, and plain-vs-weighted gap.

`outputs/summary/week6_final_literature_baseline_compare.csv` is the compact baseline compare table. It places candidate-order, popularity-prior, longtail-prior, direct ranking, and structured-risk current best under the same Beauty candidate-ranking metric schema.

`outputs/summary/part5_single_domain_main_table.csv` is the paper-ready single-domain Part5 main table skeleton. It centralizes current line, same-task baseline, retained line, and role labels across the three task layers.

`outputs/summary/part5_4domain_main_table.csv` is the paper-ready four-domain main table skeleton. It connects pointwise diagnosis, direct ranking, structured-risk reranking, and baseline references under the compact DeepSeek cross-domain setting.

`outputs/summary/part5_pairwise_boundary_table.csv` is the paper-ready pairwise boundary table. It is used to state why pairwise is currently mechanism-layer evidence rather than a main decision replacement.

## Main Figures

`outputs/summary/figures/part5/reliability_diagram_part5.png` and `outputs/summary/figures/part5/confidence_histogram_part5.png` support the pointwise diagnosis-layer claim.

`outputs/summary/figures/part5/popularity_confidence_distribution_part5.png` supports the claim that uncertainty behavior should be read together with exposure and popularity structure.

`outputs/summary/figures/part5/part5_family_compare.png` supports the structured-risk current best family decision while keeping retained exploratory families visible.

`outputs/summary/figures/part5/part5_pairwise_coverage.png` and `outputs/summary/figures/part5/part5_pairwise_scope_compare.png` record the original pairwise coverage boundary.

`outputs/summary/figures/part5/part5_pairwise_coverage_upgraded.png` records the final coverage upgrade, including supported-event improvement and direct/plain/weighted comparison.

## Notes And Reports

`outputs/summary/part5_multitask_final_summary.md` is the original final Part5 summary.

`outputs/summary/part5_figure_pack.md` explains the role of each paper-facing figure.

`outputs/summary/week6_final_4domain_structured_risk_notes.md` explains the four-domain structured-risk current best landing.

`outputs/summary/week6_final_pairwise_coverage_upgrade_notes.md` explains how pairwise coverage was repaired and what boundary remains.

`outputs/summary/week6_final_literature_baseline_notes.md` explains the baseline selection logic and current defensive value.

`outputs/summary/part5_consolidated_tables.md` explains the role of the three paper-ready consolidated tables.

`Paper/pony2/week5-week6 Part5阶段完整收口总结.md` records the complete stage-level interpretation in Chinese.

`Paper/pony2/April 19th week6 magic 8th Part5最终升级与完整收口日报.md` records the final upgrade work and its current boundary.

## Current Boundary

This artifact bundle is a compact 100-sample real-API Part5 evidence version. It is designed to be complete, traceable, and defensible enough to hand off to week7, but it does not claim final large-scale stability. It does not expand real API samples to 1000, does not enter server or local-HF execution, does not add new ranking families, and does not upgrade local swap or fully fused into the default main line.
