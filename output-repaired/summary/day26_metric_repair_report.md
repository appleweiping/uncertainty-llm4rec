# Day26 Metric Repair Report

## 1. HR@10 Triviality

Beauty and the first cross-domain medium builder use a 1 positive + 5 negatives candidate pool, so each user has only 6 candidates. Under this setup, HR@10 is trivial because top-10 covers the entire candidate pool. `HR@10 = 1.0` should not be interpreted as recommendation performance.

## 2. Valid Metrics Under 1+5 Candidates

NDCG@10 and MRR remain valid because they distinguish the exact rank of the positive item within the candidate pool. We additionally report HR@1, HR@3, HR@5, NDCG@1, NDCG@3, NDCG@5, positive-rank mean/median, and candidate-pool-size diagnostics.

## 3. Repaired Main Evidence

The main table should use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5. HR@10 is retained only with `hr10_trivial_flag=true`.

- SASRec-style: NDCG@10 `0.6099`, MRR `0.4853`, HR@1 `0.2610`, HR@3 `0.5940`, NDCG@3 `0.4503`, NDCG@5 `0.5666`.
- LLM-ESR GRU4Rec: NDCG@10 `0.6037`, MRR `0.4778`, HR@1 `0.2614`, HR@3 `0.5618`, NDCG@3 `0.4316`, NDCG@5 `0.5570`.
- LLM-ESR Bert4Rec: NDCG@10 `0.5931`, MRR `0.4642`, HR@1 `0.2439`, HR@3 `0.5358`, NDCG@3 `0.4118`, NDCG@5 `0.5385`.

## 4. Claim Text Repair

Use this wording: Scheme 4 / CEP improves NDCG and MRR over three sequential backbones under full Beauty candidate-pool evaluation. HR@10 is not used as primary evidence because the candidate pool contains fewer than 10 negatives.

Do not claim HR@10 improvement as evidence.
