# Day23 Final Claim Map

## 1. Week1-Week4 / Original Pipeline

The original confidence pipeline showed that raw/verbalized confidence is not pure noise, but it is miscalibrated and cannot be used directly as a trustworthy probability.

## 2. Day6: Yes/No Decision Confidence Repair

The yes/no controlled setting showed that evidence decomposition and decoupled reranking can repair decision reliability. This layer is useful diagnostically, but it is not the final recommendation task form.

## 3. Day9: Candidate Relevance Posterior Calibration

Candidate relevance scoring reframed the task around `relevance_probability`. Valid-set calibration produced `calibrated_relevance_probability`, which became the main Scheme 4 signal.

## 4. Day10: List-level Boundary

Direct evidence-heavy list generation was not the best first-pass decision form. Plain list generation is a better base, while evidence works better as a posterior/risk plug-in.

## 5. Day20: SASRec Full Multi-seed External Plug-in

SASRec-style full multi-seed best method `D_SASRec_plus_calibrated_relevance_plus_evidence_risk` reaches NDCG@10 `0.6099` and MRR@10 `0.4853`.

## 6. Day23: GRU4Rec Full Multi-seed External Plug-in

LLM-ESR GRU4Rec full multi-seed best method `D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk` reaches NDCG@10 `0.6037` and MRR@10 `0.4778`.

## 7. Final Method Position

Scheme 4 is best described as a calibrated evidence posterior plug-in. The primary contribution is calibrated relevance posterior; `evidence_risk` is a secondary risk regularizer. Across two sequential backbones, D generally improves over B, but C-only remains much weaker than B.

## 8. Claim Boundary

The current claim is Beauty full + two sequential backbones. It is not yet a universal SOTA claim across all domains and all recommender families. The next extension is a third backbone or cross-domain validation.
