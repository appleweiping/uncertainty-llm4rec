# Day26 Final Claim Map

## 1. Original Observation / Week1-Week4

The original pipeline showed that raw verbalized confidence and raw relevance-style signals are informative, but they are miscalibrated and should not be used directly as decision signals. This observation motivates repair rather than blind trust in self-reported confidence.

## 2. Scheme 4 / CEP

Scheme 4 is not a simple prompt rewrite. It is an evidence-grounded calibrated posterior pipeline. The LLM outputs `relevance_probability`, `positive_evidence`, `negative_evidence`, `ambiguity`, and `missing_information`; valid-set calibration then converts these fields into `calibrated_relevance_probability`. The derived `evidence_risk` is used as a risk regularizer rather than as the primary scorer.

## 3. Day6

In the yes/no decision reliability setting, evidence risk can directly represent decision risk. Decoupled reranking has clear payoff because the task is explicitly about whether a recommendation decision is reliable.

## 4. Day9

In candidate relevance posterior scoring, the main contribution is `calibrated_relevance_probability`, which repairs probability quality. AUROC is useful for discrimination diagnostics, but ECE/Brier are the core uncertainty-quality criteria.

## 5. Day10

In list-level first-pass generation, evidence decomposition is not best used as a generation burden. Plain list generation is a better first-pass base, while Scheme 4 is better positioned as post-hoc or hybrid decision support.

## 6. Day20/23/25

Across three external sequential backbones, Scheme 4 plug-in consistently improves full Beauty multi-seed ranking performance:

- SASRec-style: NDCG@10 `0.6099 +/- 0.0024`, MRR@10 `0.4853 +/- 0.0033`, relative NDCG `11.99%`, relative MRR `21.47%`.
- LLM-ESR GRU4Rec: NDCG@10 `0.6037 +/- 0.0025`, MRR@10 `0.4778 +/- 0.0032`, relative NDCG `12.92%`, relative MRR `23.42%`.
- LLM-ESR Bert4Rec: NDCG@10 `0.5931 +/- 0.0053`, MRR@10 `0.4642 +/- 0.0067`, relative NDCG `13.46%`, relative MRR `24.79%`.

The shared pattern is that `calibrated_relevance_probability` is the primary contributor, while `evidence_risk` is a secondary regularizer. C-only is weaker than B, but D consistently improves over B.

## 7. Final Claim

Scheme 4 / CEP can serve as a plug-in calibrated relevance posterior for LLM-enhanced recommendation. It improves ranking when combined with external sequential backbones, mainly through calibrated relevance posterior and secondarily through evidence-risk regularization.

## 8. Claim Boundary

The current result is Beauty full + three sequential backbones. It is not a universal SOTA claim across all domains, all recommender families, or all generation settings. Natural next steps are cross-domain validation, stronger public backbones, and Qwen-LoRA localization of the evidence generator.
