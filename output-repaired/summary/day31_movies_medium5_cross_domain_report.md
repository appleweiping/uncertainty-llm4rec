# Day31 Movies medium_5neg_2000 Cross-Domain Report

## 1. Why Movies Medium5

Movies medium_5neg_2000 comes from the regular Movies processed domain, keeps the Beauty-compatible pointwise schema, and is much cheaper than medium_20neg_2000. This makes it a useful first cross-domain consistency check before spending more API budget.

## 2. Data Scale And Metric Protocol

Valid and test inference are complete: valid `12000` rows, test `12000` rows. Each user has 1 positive plus 5 negatives, so HR@10 is trivial and is not used as primary evidence. Main metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 3. Movies Relevance Calibration

On test, raw relevance has ECE `0.2913`, Brier `0.2202`, and AUROC `0.6271`. Calibrated relevance has ECE `0.0044`, Brier `0.1297`, and AUROC `0.6334`. Minimal evidence posterior ECE is `0.0053` and full evidence posterior ECE is `0.0063`. This matches the Beauty-side observation: raw relevance probability is informative but miscalibrated, while calibrated relevance posterior repairs probability quality.

## 4. Movies SASRec Plug-in

SASRec-only reaches NDCG@10 `0.5499`, MRR `0.4095`, HR@1 `0.2035`, and HR@3 `0.4295`. The best plug-in row is `B_SASRec_plus_calibrated_relevance` with NDCG@10 `0.6728`, MRR `0.5673`, HR@1 `0.3615`, and HR@3 `0.6965`. Best B-only reaches NDCG@10 `0.6728` and MRR `0.5673`; best D reaches NDCG@10 `0.6728` and MRR `0.5673`.

## 5. Important Backbone Health Caveat

Join coverage is `1.0000`, but SASRec fallback_rate is `0.9698` with positive fallback `0.8965`. This means the Movies SASRec score export is not a healthy external-backbone conclusion yet; many candidates rely on fallback scores, likely because the regular Movies IDs/history are sparse or not mapped into the trained sequence vocabulary. Therefore, the plug-in gain should be treated as cross-domain CEP consistency, not as a final Movies external backbone result.

## 6. Direction Against Beauty

The calibration result is consistent with Beauty: raw relevance is miscalibrated and calibrated relevance posterior sharply improves ECE/Brier. The plug-in table is directionally positive, but because fallback is high, Day32 should either repair Movies backbone mapping or run Books medium5 if its backbone coverage is healthier.

## 7. Day32 Recommendation

Do not open LoRA yet. Recommended next step: inspect why Movies SASRec fallback is high. If it is a Movies mapping issue, repair backbone export before claiming Movies external plug-in. In parallel, Books medium5 can be the next cross-domain target if its processed schema and backbone mapping are healthier.
