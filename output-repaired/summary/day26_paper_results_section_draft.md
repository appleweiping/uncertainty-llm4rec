# Day26 Paper Results Section Draft

## 1. Uncertainty Quality

The early confidence pipeline shows that raw LLM confidence is not pure noise, but it is not a calibrated probability. In the candidate relevance setting, the calibrated evidence posterior (CEP) repairs this problem by fitting a valid-set calibration model over evidence-derived features. We therefore evaluate uncertainty quality with calibration-sensitive metrics such as ECE and Brier score, rather than treating raw self-reported confidence as directly decision-ready.

## 2. Decision Repair

The yes/no decision setting provides a controlled diagnostic environment. In that setting, evidence risk naturally represents decision unreliability, and a decoupled risk-aware reranking formulation can change decisions. This result is useful for validating the risk component, but it is not the final form of the recommendation task.

## 3. Candidate Relevance Posterior

For candidate-level recommendation, CEP reframes the model output as relevance posterior estimation. The primary signal is `calibrated_relevance_probability`; it is not the same as raw confidence and is not obtained by directly trusting the model's self-report. The evidence fields are used to construct a posterior that is calibrated on validation data and then evaluated on held-out test data.

## 4. External Backbone Plug-in

We evaluate CEP as a plug-in on three sequential recommendation backbones under full Beauty multi-seed validation. No additional DeepSeek API calls are made during these plug-in experiments; all runs reuse the Day9 full evidence table.

| backbone | method | NDCG@10_mean | NDCG@10_std | MRR@10_mean | MRR@10_std | relative_NDCG_vs_backbone_mean | relative_MRR_vs_backbone_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SASRec-style | Backbone only | 0.5446 | 0.0028 | 0.3996 | 0.0038 | 0.0000 | 0.0000 |
| SASRec-style | Backbone + calibrated relevance | 0.6053 | 0.0016 | 0.4790 | 0.0023 | 0.1114 | 0.1988 |
| SASRec-style | Backbone + evidence risk | 0.5560 | 0.0048 | 0.4146 | 0.0065 | 0.0210 | 0.0378 |
| SASRec-style | Backbone + calibrated relevance + evidence risk | 0.6099 | 0.0024 | 0.4853 | 0.0033 | 0.1199 | 0.2147 |
| LLM-ESR GRU4Rec | Backbone only | 0.5347 | 0.0066 | 0.3873 | 0.0088 | 0.0000 | 0.0000 |
| LLM-ESR GRU4Rec | Backbone + calibrated relevance | 0.5920 | 0.0036 | 0.4622 | 0.0049 | 0.1072 | 0.1938 |
| LLM-ESR GRU4Rec | Backbone + evidence risk | 0.5454 | 0.0033 | 0.4018 | 0.0044 | 0.0201 | 0.0378 |
| LLM-ESR GRU4Rec | Backbone + calibrated relevance + evidence risk | 0.6037 | 0.0025 | 0.4778 | 0.0032 | 0.1292 | 0.2342 |
| LLM-ESR Bert4Rec | Backbone only | 0.5231 | 0.0185 | 0.3728 | 0.0238 | 0.0000 | 0.0000 |
| LLM-ESR Bert4Rec | Backbone + calibrated relevance | 0.5844 | 0.0099 | 0.4521 | 0.0128 | 0.1178 | 0.2148 |
| LLM-ESR Bert4Rec | Backbone + evidence risk | 0.5364 | 0.0136 | 0.3903 | 0.0174 | 0.0258 | 0.0478 |
| LLM-ESR Bert4Rec | Backbone + calibrated relevance + evidence risk | 0.5931 | 0.0053 | 0.4642 | 0.0067 | 0.1346 | 0.2479 |

The best D setting improves over the corresponding backbone-only baseline on all three backbones:

- SASRec-style: NDCG@10 `0.6099 +/- 0.0024`, MRR@10 `0.4853 +/- 0.0033`, relative NDCG `11.99%`, relative MRR `21.47%`.
- LLM-ESR GRU4Rec: NDCG@10 `0.6037 +/- 0.0025`, MRR@10 `0.4778 +/- 0.0032`, relative NDCG `12.92%`, relative MRR `23.42%`.
- LLM-ESR Bert4Rec: NDCG@10 `0.5931 +/- 0.0053`, MRR@10 `0.4642 +/- 0.0067`, relative NDCG `13.46%`, relative MRR `24.79%`.

## 5. Component Attribution

The component pattern is consistent. B (backbone + calibrated relevance) accounts for most of the gain. C (backbone + evidence risk only) is positive but weaker. D (backbone + calibrated relevance + evidence risk) consistently improves over B, indicating that evidence risk is best interpreted as a secondary regularizer rather than as the main scorer.

## 6. Limitations

These results are currently limited to Amazon Beauty and sequential/backbone-level validation. The evidence generator uses DeepSeek API outputs from Day9; future work can localize it through Qwen-LoRA. The current results should not be claimed as universal SOTA across all domains or all recommendation settings until cross-domain and stronger-public-backbone validations are complete.
