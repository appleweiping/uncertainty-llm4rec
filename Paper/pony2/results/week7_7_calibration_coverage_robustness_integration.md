# Week7.7 Calibration, Coverage, And Robustness Integration

This note records how the current Week7-7.7 SRPD line absorbs the earlier uncertainty, coverage, and robustness work instead of treating it as a separate side project.

## What Is Already Covered

The current SRPD evidence is no longer only an NDCG/MRR table. I have now promoted the four-domain SRPD result matrix into `week7_7_four_domain_srpd_full_metric_matrix.csv` and `week7_7_four_domain_srpd_full_metric_matrix.md`, where every locally verified `direct / structured_risk / SRPD-v1 / SRPD-v2 / SRPD-v3 / SRPD-v4 / SRPD-v5` row carries `coverage@10`, `head_exposure_ratio@10`, `longtail_coverage@10`, `parse_success_rate`, and `out_of_candidate_rate`. This matters because SRPD is an uncertainty-aware recommendation method: if a version improves NDCG but destroys coverage, increases head exposure, or becomes parse-unstable, it should not be treated as a clean win.

The calibration side is also recoverable from existing pointwise teacher files. I have generated `week7_7_teacher_reliability_summary.csv` and `week7_7_teacher_reliability_summary.md` from the four DeepSeek pointwise uncertainty diagnostics: Beauty full5838, Books full3000, Electronics full3000, and Movies full3000. These rows carry `accuracy`, `avg_confidence`, `Brier`, `ECE`, `MCE`, and `AUROC`, and they should be treated as Teacher Reliability Layer evidence. In the current interpretation, Beauty and Books provide positive-domain distillation evidence, Electronics is a repair-domain case, and Movies is the strongest failure-boundary case because its teacher reliability signal is weaker and its SRPD distillation variants do not yet beat direct/structured risk.

The robustness and ablation pieces are partly covered by the SRPD variant ladder itself. `SRPD-v1` is the plain structured-risk teacher SFT ablation; `SRPD-v2` adds uncertainty weighting; `SRPD-v4` adds gap-gated reliability control; `SRPD-v5` adds direct-anchor fallback; `SRPD-v6` adds DPO-style preference SFT. Therefore the version ladder already forms a mechanism ablation path. The early robustness assets under `outputs/robustness/`, including DeepSeek noise-curve summaries, should be cited as existing robustness infrastructure and then extended in Week8/Week9 to the final SRPD-System if needed.

## What Still Needs To Be Extended

The current four-domain full-metric matrix covers the locally synced v1-v5 results. `SRPD-v6` should not be mixed into that formal file until the server outputs are synced back locally. The server-reported Electronics v6 result is already known to be weaker than v5 (`NDCG@10 = 0.6442090733041972`, `MRR = 0.5305`), which means preference SFT is not automatically better than direct-anchor repair on Electronics. Once Movies/Beauty/Books v6 are run and synced, v6 should be added to a refreshed full-metric matrix with the same coverage/exposure/parse columns.

The remaining robustness section should be framed as mechanism-level ablation rather than random stress testing. The core comparisons are: remove uncertainty weighting (`v1` or ordinary SFT vs `v2`), remove gap gate (`v2` vs `v4`), remove direct anchor (`v4` vs `v5`), remove preference pairs (`v4/v5` vs `v6`), and compare across domains as a domain-shift robustness check. If Week8 adds same-schema NH proxy baselines, they should also report the same side metrics so that outer compare is not reduced to utility only.

## Paper-Level Contribution

The important story is that Week1-Week6 assets are not discarded. Calibration becomes Teacher Reliability evidence; coverage and head/long-tail exposure become side-effect metrics; robustness and sensitivity become ablation infrastructure; baseline aggregation becomes the Week8 outer-compare bridge; and pointwise/pairwise/ranking become the multi-signal foundation for listwise and preference-based SRPD. This lets the paper claim a complete uncertainty-aware LLM4Rec evaluation stack: reliability, utility, exposure, robustness, and cross-domain boundary analysis.
