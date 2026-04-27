# Framework-Observation-Day1f Go/No-Go Decision

## Recommendation

`do_not_use_self_consistency`

## Interpretation

Self-consistency is not the primary confidence line. After the tie-aware ranking fix, self-consistency no longer beats logit P(true). It is more expensive and weaker than logit on the same subset.

Calibration still helps ECE/Brier, but ranking/AUROC signal remains weak. Do not full-run self-consistency, do not continue scalar confidence wording, and do not enter CEP/evidence yet. The next route is pair/list context rather than more scalar confidence wording.

## Test Snapshot

- logit P(true) MRR/AUROC: `0.48083333333333333` / `0.57436`
- self-consistency MRR/AUROC: `0.45764166666666667` / `0.53713`
- logit P(true) HR@1/NDCG@3: `0.26` / `0.4422580581071478`
- self-consistency HR@1/NDCG@3: `0.23833333333333334` / `0.4075760539392899`
