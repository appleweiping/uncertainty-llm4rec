# Decision Mismatch Analysis

## 1. What This Analysis Explains

This local-only analysis explains why CEP / evidence-risk reranking improves NDCG and MRR. The improvement is not just an abstract metric change. It comes from reducing two kinds of mismatch: probability calibration mismatch and ranking decision mismatch.

## 2. Calibration Mismatch

On full Beauty relevance evidence, raw relevance probability is informative but poorly calibrated: ECE=0.2604, Brier=0.2318, AUROC=0.6007, high-confidence error rate=0.5274. Valid-set calibration reduces this to ECE=0.0095, Brier=0.1347, AUROC=0.5986, high-confidence error rate=0.0000. CEP full reports ECE=0.0116, Brier=0.1348, AUROC=0.6038, high-confidence error rate=0.0000.

This supports the first mismatch claim: raw predicted probability does not match empirical correctness well, while calibrated relevance posterior substantially reduces probability-scale mismatch.

## 3. Ranking Mismatch

Using the best full-grid D setting for each backbone, we compare old ranks from `backbone_score` against new ranks after adding calibrated relevance and evidence risk:

- SASRec-style: positive mean rank 3.4512 -> 3.1017 (improvement 0.3494); inversion rate 0.4902 -> 0.4203 (reduction 0.0699); demoted negative rate=0.8675, promoted positive rate=0.2344.
- GRU4Rec: positive mean rank 3.5437 -> 3.1850 (improvement 0.3587); inversion rate 0.5087 -> 0.4370 (reduction 0.0717); demoted negative rate=0.8639, promoted positive rate=0.2276.
- Bert4Rec: positive mean rank 3.8623 -> 3.3361 (improvement 0.5262); inversion rate 0.5725 -> 0.4672 (reduction 0.1052); demoted negative rate=0.8710, promoted positive rate=0.2276.

Positive items move earlier on average, and fewer negative candidates remain ranked above the positive candidate. This is the ranking-decision side of the NDCG/MRR gain.

## 4. Evidence-Risk Mechanism

The demotion/promotion diagnostics show whether reranking is random or mechanism-consistent. A useful pattern is: demoted candidates should be mostly negatives and higher-risk; promoted candidates should contain more positives. The table reports `demoted_negative_rate`, `high_risk_demotion_precision`, and `promoted_positive_rate` for each backbone.

## 5. Interpretation

NDCG/MRR improves because the method reduces two mismatch types:

- Calibration mismatch: raw relevance probability is not a reliable probability, but calibrated relevance posterior and CEP full reduce ECE/Brier.
- Ranking decision mismatch: high-risk negative candidates are pushed down, positive candidates are moved earlier, and pairwise inversions decrease.

This preserves the intended method boundary: backbone scores provide ranking ability; calibrated relevance probability provides posterior relevance; evidence risk supplies secondary risk regularization.

## Local-Only Execution Note

This analysis used existing Day9 relevance outputs and Day19/22/25 full backbone joined candidates. It did not call APIs, retrain models, alter prompts/parsers/formulas, or touch the running Movies Day29 pipeline.
