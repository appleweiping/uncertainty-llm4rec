# Day25 Bert4Rec Full Multi-seed Report

## 1. Day24 Recap

Day24 showed a positive 100-user smoke for LLM-ESR Bert4Rec. It is a third external backbone: not minimal SASRec-style and not GRU4Rec, while still exposing real candidate logits via `Bert4Rec.predict()`.

## 2. Day25 Setup

Full Beauty candidate pool is scored with LLM-ESR Bert4Rec trained only on the Beauty train split. No DeepSeek API calls, prompt changes, LoRA, or formula changes are used. Day9 full evidence is reused for calibrated relevance and evidence risk.

## 3. Backbone Health

Users: `973`.

Candidate rows: `5838`.

Join coverage: `1.0000`.

Fallback rate: `0.1336`.

Positive fallback rate: `0.1984`.

Negative fallback rate: `0.1207`.

Bert4Rec-only NDCG@10: `0.5020`.

Bert4Rec-only MRR@10: `0.3457`.

## 4. Full Plug-in Grid

Best method: `D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk`.

Best NDCG@10: `0.5871`.

Best MRR@10: `0.4567`.

Relative NDCG improvement vs Bert4Rec-only: `0.1697`.

Relative MRR improvement vs Bert4Rec-only: `0.3210`.

Best row per method:

| method | HR@10 | NDCG@10 | MRR@10 | relative_NDCG_vs_backbone | relative_MRR_vs_backbone | lambda | alpha | beta | normalization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Bert4Rec_only | 1.0000 | 0.5020 | 0.3457 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | minmax |
| B_Bert4Rec_plus_calibrated_relevance | 1.0000 | 0.5756 | 0.4410 | 0.1466 | 0.2757 | 0.0000 | 0.5000 | 0.5000 | minmax |
| C_Bert4Rec_plus_evidence_risk | 1.0000 | 0.5216 | 0.3714 | 0.0391 | 0.0745 | 0.5000 | 1.0000 | 0.0000 | zscore |
| D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk | 1.0000 | 0.5871 | 0.4567 | 0.1697 | 0.3210 | 0.5000 | 0.5000 | 0.5000 | minmax |

## 5. Multi-seed Stability

Fixed settings selected from the seed-42 full grid, then reused for seeds 42/43/44:

| method | normalization | alpha | beta | lambda |
| --- | --- | --- | --- | --- |
| A_Bert4Rec_only | none | 1.0000 | 0.0000 | 0.0000 |
| B_Bert4Rec_plus_calibrated_relevance | minmax | 0.5000 | 0.5000 | 0.0000 |
| C_Bert4Rec_plus_evidence_risk | minmax | 1.0000 | 0.0000 | 0.5000 |
| D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk | minmax | 0.5000 | 0.5000 | 0.5000 |

Multi-seed summary:

| method | HR@10_mean | HR@10_std | NDCG@10_mean | NDCG@10_std | MRR@10_mean | MRR@10_std | Recall@10_mean | Recall@10_std | relative_NDCG_vs_bert4rec_mean | relative_NDCG_vs_bert4rec_std | relative_MRR_vs_bert4rec_mean | relative_MRR_vs_bert4rec_std | fallback_rate_mean | fallback_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Bert4Rec_only | 1.0000 | 0.0000 | 0.5231 | 0.0185 | 0.3728 | 0.0238 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1336 | 0.0000 |
| B_Bert4Rec_plus_calibrated_relevance | 1.0000 | 0.0000 | 0.5844 | 0.0099 | 0.4521 | 0.0128 | 1.0000 | 0.0000 | 0.1178 | 0.0256 | 0.2148 | 0.0536 | 0.1336 | 0.0000 |
| C_Bert4Rec_plus_evidence_risk | 1.0000 | 0.0000 | 0.5364 | 0.0136 | 0.3903 | 0.0174 | 1.0000 | 0.0000 | 0.0258 | 0.0105 | 0.0478 | 0.0211 | 0.1336 | 0.0000 |
| D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk | 1.0000 | 0.0000 | 0.5931 | 0.0053 | 0.4642 | 0.0067 | 1.0000 | 0.0000 | 0.1346 | 0.0306 | 0.2479 | 0.0639 | 0.1336 | 0.0000 |

## 6. Component Attribution

| method | NDCG@10_mean | MRR@10_mean | relative_NDCG_vs_bert4rec_mean | relative_MRR_vs_bert4rec_mean | delta_NDCG_vs_A_mean | delta_MRR_vs_A_mean | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_Bert4Rec_only | 0.5231 | 0.3728 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | external Bert4Rec backbone only |
| B_Bert4Rec_plus_calibrated_relevance | 0.5844 | 0.4521 | 0.1178 | 0.2148 | 0.0613 | 0.0793 | calibrated relevance posterior contribution |
| C_Bert4Rec_plus_evidence_risk | 0.5364 | 0.3903 | 0.0258 | 0.0478 | 0.0134 | 0.0175 | evidence risk as standalone regularizer |
| D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk | 0.5931 | 0.4642 | 0.1346 | 0.2479 | 0.0700 | 0.0914 | posterior plus evidence-risk regularization |

## 7. Comparison With SASRec And GRU4Rec

SASRec full multi-seed best `D_SASRec_plus_calibrated_relevance_plus_evidence_risk`: NDCG `0.6099`, MRR `0.4853`. GRU4Rec full multi-seed best `D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk`: NDCG `0.6037`, MRR `0.4778`.

The expected pattern is consistent if B carries most of the improvement, C remains weaker, and D is at least competitive with or above B. This supports the position that calibrated relevance posterior is the main contribution and evidence risk is a secondary regularizer.

## 8. Day26 Recommendation

Day26 should build the three-backbone final table and final paper-facing claim map. Keep the claim bounded to Beauty full and these external/sequential backbones unless cross-domain experiments are added.
