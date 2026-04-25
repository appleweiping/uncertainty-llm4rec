# Day29 Movies medium_5neg_2000 Report

## 1. Why Pause 20neg_2000

Movies medium_20neg_2000 preflight and tiny smoke passed, and partial valid predictions were preserved. Full 20neg_2000 inference was intentionally paused because API/runtime cost was too high for this stage.

## 2. Completed 20neg_2000 Artifacts

The preserved artifacts include preflight, smoke report, runtime monitor, and partial `valid_raw.jsonl`. They can be resumed later and are not deleted.

## 3. Why Switch To 5neg_2000

Movies medium_5neg_2000 uses the same candidate-pool size as the Beauty main experiments: 1 positive + 5 negatives per user. This gives a lighter cross-domain consistency check before returning to 20neg ranking-strength evaluation.

## 4. Metric Protocol

Because each user has 6 candidates, HR@10 is trivial and is not used as primary evidence. Main metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5. All plug-in rows include `hr10_trivial_flag=true`.

## 5. Relevance Calibration

Raw relevance ECE is `0.2913` and Brier is `0.2202`. Calibrated relevance ECE is `0.0044` and Brier is `0.1297`. Minimal evidence posterior ECE is `0.0053`; full evidence posterior ECE is `0.0063`.

## 6. Field Diagnostics

On test, relevance_probability mean is `0.4578`, std `0.1237`, near_one_rate `0.0008`. evidence_risk mean is `0.6622`, std `0.1648`.

## 7. SASRec Plug-in

Best method is `B_SASRec_plus_calibrated_relevance` with normalization `zscore`, alpha `0.5`, beta `0.5`, lambda `0.0`. SASRec-only NDCG@10 is `0.5499`, MRR `0.4095`, HR@3 `0.4295`. Best plug-in NDCG@10 is `0.6728`, MRR `0.5673`, HR@3 `0.6965`. Relative gains are NDCG `22.35%` and MRR `38.53%`.

## 8. Direction Against Beauty

If calibrated relevance or the D combination improves NDCG/MRR over SASRec-only, the cross-domain direction is consistent with Beauty: calibrated relevance posterior is the main contributor, evidence_risk is a secondary regularizer.

## 9. Day30 Recommendation

If Movies 5neg is positive, Day30 can either run Books/Electronics 5neg for cross-domain consistency or return to Movies 20neg_2000 as a stronger ranking benchmark when API budget allows.
