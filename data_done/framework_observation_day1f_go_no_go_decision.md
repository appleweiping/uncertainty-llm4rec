# Framework-Observation-Day1f Go/No-Go Decision

## Recommendation

`do_not_use_self_consistency`

## Interpretation

Self-consistency is compared only against logit P(true) on the exact same Day1f 100-user valid/test subsets. If both methods remain weak, the next route is pair/list context rather than more scalar confidence wording.

## Test Snapshot

- logit P(true) MRR/AUROC: `0.48083333333333333` / `0.57436`
- self-consistency MRR/AUROC: `0.45764166666666667` / `0.53713`
- logit P(true) HR@1/NDCG@3: `0.26` / `0.4422580581071478`
- self-consistency HR@1/NDCG@3: `0.23833333333333334` / `0.4075760539392899`
