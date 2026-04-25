# Day9/Day10 Task-Specific Repair Report

## 1. Formulation Correction

Day6, Day9, and Day10 should not be forced into one identical formula. Day6 is yes/no decision confidence repair, Day9 is candidate relevance posterior calibration, and Day10 is list-level recommendation selection. Scheme four is the shared component, but its role changes by task: reliability repair in Day6, calibrated relevance posterior in Day9, and post-hoc risk audit/reranking for Day10.

## 2. Day10 Plain Base + Evidence Risk

Using the reconstructed Day10 plain rank score as the base ranking gives NDCG@10 `0.624715` and MRR@10 `0.504916`. The best plain-base evidence-risk rerank gives NDCG@10 `0.625871` and MRR@10 `0.506423` with lambda `0.2` and normalization `zscore`. Relative to the reconstructed rank-score baseline, changes are NDCG `0.0019` and MRR `0.0030`.

However, the originally reported Day10 full plain direct list remains NDCG@10 `0.625992` and MRR@10 `0.506458`. Relative to that reported plain list, the best plain-base evidence-risk rerank changes NDCG by `-0.0002` and MRR by `-0.0001`. This distinction matters because converting a generated list into a rank-score table introduces a small reconstruction gap.

## 3. Day9 Stronger Base Ablation

The best stronger-base setting is `D_hybrid_alpha_0.25` with lambda `0.1` and normalization `zscore`, reaching NDCG@10 `0.628194` and MRR@10 `0.509301`. Relative to the reported Day10 plain list, this is NDCG `0.0035` and MRR `0.0056`. This table compares relevance_probability, calibrated_relevance_probability, Day10 plain_rank_score, and hybrid calibrated relevance plus plain rank score.

## 4. Diagnostics

The diagnostic table checks whether the base score itself is strong, whether evidence_risk is aligned with misranked negatives, and whether promoted/demoted items have the expected risk pattern. For the best hybrid setting, evidence_risk AUROC for misranked negatives is `0.490721`, which is close to random. This is the key guardrail: the current improvement should be interpreted as a small task-specific hybrid/risk effect, not as proof that evidence_risk alone robustly detects all wrong high-rank items.

## 5. Conclusion

The repaired formulation is better stated as task-specific: Day10 first-pass evidence list generation should not be tuned further, but a hybrid of plain list rank score and Day9 calibrated relevance with a mild evidence-risk penalty gives the best current full Beauty result. The gain is positive but modest and below the 5% external-backbone target.

## 6. Output Files

- `output-repaired/summary/beauty_day10_plain_base_evidence_risk_rerank_grid.csv`
- `output-repaired/summary/beauty_day9_stronger_base_evidence_risk_ablation.csv`
- `output-repaired/summary/beauty_day9_day10_repair_diagnostics.csv`
