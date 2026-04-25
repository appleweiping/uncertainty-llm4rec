# Day29b Observation: Raw LLM Confidence Is Informative but Miscalibrated

## 1. Motivation

The starting question is whether an LLM's verbalized confidence in a recommendation task can be used directly as a decision signal. The consolidated evidence says no: the signal is useful, but its probability scale is not reliable enough to use without calibration.

## 2. Raw Confidence Is Informative but Miscalibrated

Raw confidence is not pure noise. Across Beauty diagnostics, AUROC is often meaningfully above chance, indicating that confidence has a relationship with correctness. At the same time, ECE, Brier score, and high-confidence error behavior show that the raw confidence values should not be interpreted as calibrated probabilities.

## 3. Multi-Model Evidence

Beauty raw confidence diagnostics across five LLMs:

- DeepSeek: diagnostic ECE=0.3025, Brier=0.2972, AUROC=0.7458, mean confidence=0.7945
- Doubao: diagnostic ECE=0.3458, Brier=0.3331, AUROC=0.7754, mean confidence=0.8658
- GLM: diagnostic ECE=0.3790, Brier=0.3620, AUROC=0.7788, mean confidence=0.8490
- Kimi: diagnostic ECE=0.3163, Brier=0.3244, AUROC=0.7036, mean confidence=0.7993
- Qwen: diagnostic ECE=0.3706, Brier=0.3616, AUROC=0.8328, mean confidence=0.9306

Mean Beauty diagnostic ECE across the five model runs is 0.3428. This supports the observation that miscalibration is not a single-model artifact. The broader output table also includes Books-small, Electronics-small, and Movies-small where available.

## 4. Calibration Helps but Does Not Create Ranking Ability

Valid-set calibration usually reduces probability-scale error. On Beauty, the mean ECE reduction across the five model runs is 0.2383, and the mean Brier reduction is 0.1416. AUROC changes are much smaller on average (-0.0007), which is expected: calibration mainly repairs probability scale rather than training a new ranker.

This distinction matters for the paper framing. CEP should not be described as a standalone ranking model. Its core role is to turn informative but miscalibrated LLM signals into calibrated posterior scores that can support downstream decisions.

## 5. Relevance Setting Evidence

The same phenomenon appears after moving from yes/no confidence to candidate relevance scoring. On full Beauty candidate relevance evidence, raw relevance probability has ECE=0.2604, Brier=0.2318, and AUROC=0.6007. Calibrated relevance probability reduces this to ECE=0.0095, Brier=0.1347, and AUROC=0.5986. The full evidence posterior variant reports ECE=0.0116, Brier=0.1348, and AUROC=0.6038.

## 6. Why Scheme 4 / CEP Is Needed

The consolidated conclusion is: raw LLM confidence or relevance signal is informative but unreliable as a probability. Therefore, before using it in recommendation decisions, we need an evidence-grounded calibrated posterior. Scheme 4 / CEP supplies that bridge by combining relevance probability, positive/negative evidence, ambiguity, missing information, and valid-set calibration.

## 7. Connection to External Backbone Plug-In

The later Day20/Day23/Day25 results should be read through this lens. The method does not replace SASRec, GRU4Rec, or Bert4Rec with raw LLM confidence. Instead, it plugs a calibrated relevance posterior into external sequential backbones, with evidence risk acting as a secondary regularizer.

## Local-Only Execution Note

This Day29b consolidation used only existing files under `outputs/` and `output-repaired/summary/`. It did not call DeepSeek, did not train a backbone, did not change prompts/parsers/formulas, and did not touch the running Day29 Movies inference process.
