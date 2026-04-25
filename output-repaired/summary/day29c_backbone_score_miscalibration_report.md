# Day29c Backbone Score Calibration Diagnostic

## 1. Motivation

Day29b consolidated the observation that raw LLM confidence and raw relevance probability are informative but miscalibrated. Day29c checks the analogous question for recommender backbone scores: can SASRec-style, GRU4Rec, or Bert4Rec `backbone_score` be directly used as confidence or probability?

## 2. Clarification

The answer should be no by design. These backbone scores are ranking logits, dot products, or sequence scores. They are effective for ordering candidates, but they are not calibrated estimates of `P(relevant)`. This diagnostic is therefore not a critique that the backbones are broken; it is a reminder not to interpret raw ranking scores as probabilities.

## 3. Diagnostic Result

If we map backbone scores into probability-like quantities with naive transformations such as global sigmoid, global min-max, per-user min-max, or per-user softmax, calibration remains unreliable:

- SASRec-style: best naive Brier proxy is `softmax_user_score` (ECE=0.0826, Brier=0.1494, AUROC=0.5076); best ranking proxy is `softmax_user_score` (NDCG@10=0.5438, MRR=0.3983).
- GRU4Rec: best naive Brier proxy is `softmax_user_score` (ECE=0.0860, Brier=0.1532, AUROC=0.4751); best ranking proxy is `softmax_user_score` (NDCG@10=0.5360, MRR=0.3886).
- Bert4Rec: best naive Brier proxy is `softmax_user_score` (ECE=0.0820, Brier=0.1499, AUROC=0.4181); best ranking proxy is `softmax_user_score` (NDCG@10=0.5056, MRR=0.3503).

The candidate pool size is approximately 6 for the Beauty full candidate-pool evaluation, so HR@10 is trivial and not used as evidence. NDCG and MRR remain meaningful because they depend on the positive item's exact rank.

## 4. Ranking vs Calibration

The backbone score's value is ranking ability, measured by NDCG/MRR/HR@1/HR@3. CEP's value is calibrated relevance posterior quality, measured by ECE/Brier, and then its usefulness as a plug-in decision signal. CEP calibrated relevance posterior on the Beauty candidate pool reports ECE=0.0096, Brier=0.1349, NDCG@10=0.6707, and MRR=0.5641.

## 5. Connection to Main Method

The final method does not replace the external backbone with CEP and does not treat backbone scores as confidence. The intended decomposition is:

- `backbone_score` provides ranking ability.
- `calibrated_relevance_probability` provides calibrated posterior relevance.
- `evidence_risk` provides secondary risk regularization.

This is the same interpretation used in the Day20/Day23/Day25 external backbone results.

## 6. Claim Boundary

Valid split backbone scores are not available for the fixed full Beauty candidate-score files, so this report does not fit a proper backbone calibrator. It only reports naive probability diagnostics. We therefore do not claim that all backbone scores have been fully calibrated. The scoped conclusion is diagnostic: raw recommender scores should not be used as calibrated confidence without calibration, and CEP is better positioned as a calibrated posterior plug-in.

## Local-Only Execution Note

This analysis read existing `candidate_scores.csv`, joined-candidate summaries, and Day29b tables only. It did not call DeepSeek, did not train any backbone, did not change prompt/parser/formula code, and did not touch the running Day29 Movies inference process.
