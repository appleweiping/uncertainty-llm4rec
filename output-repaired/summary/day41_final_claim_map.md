# Day41 Final Claim Map

## 1. Starting Observation

LLM recommendation confidence and relevance signals are informative but miscalibrated. Multi-model Beauty diagnostics show this is not a single-model accident: raw verbalized confidence and raw relevance probability carry signal, but ECE/Brier/high-confidence error indicate they are unreliable as direct probabilities.

## 2. Method

CEP / Scheme 4 is an evidence-grounded calibrated posterior, not a prompt rewrite. It asks for relevance and evidence fields, then uses valid-set calibration to produce `calibrated_relevance_probability` and derives `evidence_risk` from ambiguity, missing information, and evidence margin.

## 3. Task-Specific Formulation

Day6 treats evidence risk as a decision-reliability signal for yes/no reranking. Day9 treats candidate-level output as relevance posterior calibration. Day10 establishes that evidence decomposition is not ideal as a heavy first-pass list-generation burden. External backbone experiments use CEP as a plug-in calibrated posterior with secondary risk regularization.

## 4. Main Performance Evidence

The main performance evidence is Beauty full + three sequential backbones + multi-seed: SASRec-style, GRU4Rec, and Bert4Rec. In this setting, CEP improves NDCG/MRR consistently, with calibrated relevance as the main contributor and evidence risk as a secondary regularizer.

## 5. Robustness Evidence

Day30 shows CEP does not collapse under controlled noisy input on a Beauty 500-user subset. Noisy D remains close to clean CEP, and observed NDCG/MRR drops are bounded in this first robustness run.

## 6. Cross-Domain Evidence

Small-domain Movies/Books/Electronics support calibration consistency and directionality, but backbone fallback is non-trivial. These results are cross-domain sanity / continuity evidence, not fully healthy external-backbone proof. Regular medium analysis reveals realistic cold-start issues and motivates content-aware/cold-aware carriers.

## 7. Boundary

Do not claim universal SOTA. Do not claim evidence risk is the main scorer. Do not use HR@10 as primary evidence when the candidate pool has six items. Do not describe fallback-heavy small-domain results as fully healthy backbone benchmarks.

## 8. Next Phase

Two reasonable next directions are: (1) Qwen-LoRA / local evidence-generator framework, if the goal is system ownership and cost reduction; or (2) stronger content-aware/cold-aware cross-domain backbone, if the goal is extending beyond Beauty into regular medium domains.
