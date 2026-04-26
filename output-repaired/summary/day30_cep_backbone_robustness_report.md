# Day30 CEP + Backbone Robustness Report

## 1. Motivation

Week1-Week4 included robustness for the old raw-confidence pipeline. Day30 adds robustness for the CEP route after Day9/Day10 and for the external backbone plug-in setting.

## 2. Setup

The experiment uses a fixed Beauty 500-user subset sampled with seed 42 from the full Beauty test users. The candidate pool is unchanged from the Beauty full setting: 1 positive plus 5 negatives per user. Noise types are `history_dropout`, `candidate_text_dropout`, and `history_swap_noise` with levels 0.1, 0.2, and 0.3. The prompt and scoring formula are unchanged.

Because the candidate pool has 6 items, HR@10 is trivial and is not used as primary evidence. The report focuses on NDCG@10, MRR, HR@1/HR@3, NDCG@3, and NDCG@5.

## 3. CEP Signal Robustness

Clean raw relevance has ECE `0.2585` and Brier `0.2285`. Clean calibrated relevance has ECE `0.0065` and Brier `0.1338`. Noisy rows use the clean Beauty valid calibrator applied to noisy test outputs; no noisy test fit is used. The minimum noisy parse success rate is `1.0000`.

## 4. Backbone Plug-in Robustness

Clean SASRec-only has NDCG@10 `0.5502`, MRR `0.4068`, and HR@3 `0.5220`. Clean best method is `B_SASRec_plus_calibrated_relevance` with NDCG@10 `0.6226`, MRR `0.5021`, HR@3 `0.6140`. SASRec-only is fixed across noise because the candidate pool does not change; B/D rows use noisy CEP fields.

Top noisy rows by NDCG/MRR:

| condition | method | NDCG@10 | MRR | HR@3 | degradation_NDCG | degradation_MRR |
| --- | --- | --- | --- | --- | --- | --- |
| history_dropout_0.2 | D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6188 | 0.4971 | 0.6000 | 0.0029 | 0.0040 |
| history_dropout_0.3 | D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6175 | 0.4952 | 0.6060 | 0.0043 | 0.0059 |
| history_dropout_0.1 | D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6163 | 0.4937 | 0.5920 | 0.0055 | 0.0074 |

## 5. Key Finding

Across the completed noisy settings, the best noisy rows remain close to clean CEP performance and stay above the fixed SASRec-only baseline. The largest observed D-setting drop is NDCG `0.0207` and MRR `0.0276`. This supports the limited claim that CEP remains useful under these light perturbations and that evidence risk can act as a secondary regularizer. Interpret the results as 500-user controlled robustness, not a full robustness benchmark.

## 6. Limitations

This is not full Beauty robustness and only prioritizes SASRec. GRU4Rec/Bert4Rec can be added later if the first run is stable.

## 7. Day31 Recommendation

If the noisy rows remain positive relative to SASRec-only and degradation is moderate, Day31 should extend robustness to GRU4Rec/Bert4Rec or fold this into the final robustness section. If noisy CEP degrades sharply, the next step should be noisy-aware calibration rather than prompt tuning.
