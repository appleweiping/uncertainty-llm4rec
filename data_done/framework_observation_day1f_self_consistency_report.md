# Framework-Observation-Day1f Self-Consistency Confidence Report

## Scope

Day1f is a local Beauty 100-user valid/test smoke with complete six-candidate pools. It does not train, use evidence, use CEP, call external APIs, or run four domains.

## Self-Consistency Signal

- test AUROC: `0.53713`
- test ECE/Brier: `0.17099999999999999` / `0.1706`
- test correctness AUROC: `0.5564192949907235`
- test parse success rate: `1.0`

## Ranking

- self-consistency test MRR / random MRR: `0.45764166666666667` / `0.4083333333333333`
- self-consistency test HR@1 / random HR@1: `0.23833333333333334` / `0.16666666666666666`
- self-consistency test NDCG@3 / random NDCG@3: `0.4075760539392899` / `0.35515495892857624`

Ranking metrics are tie-aware: when candidates receive the same self-consistency frequency, the report uses the expected metric under uniform random tie-breaking rather than preserving JSONL row order.

## Logit Comparison

See `data_done/framework_observation_day1f_logit_vs_self_consistency_comparison.csv` for same-subset comparison.

## Interpretation

Self-consistency is not the primary confidence line. After the tie-aware ranking fix, self-consistency no longer beats logit P(true). It is more expensive and weaker than logit on the same subset.

Calibration still helps ECE/Brier, but ranking/AUROC signal remains weak. Do not full-run self-consistency, do not continue scalar confidence wording, and do not enter CEP/evidence yet.

## Recommendation

`do_not_use_self_consistency`
