# Framework-Observation-Day1c Decision-Forced Confidence Report

## Scope

Day1c is a local Qwen-LoRA confidence elicitation smoke on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or training.

## Prediction Directory

`output-repaired\framework_observation\beauty_qwen_lora_confidence_decision_forced\predictions`

## Decision / Confidence Diagnostics

- test rows / valid rows: `0` / `0`
- test parse success: `0.0`
- test schema valid: `0.0`
- test accuracy: `0.0`
- test AUROC: `NA`
- test ECE: `0.0`
- test Brier: `0.0`
- recommend true / false rate: `0.0` / `0.0`
- confidence mean / std: `0.0` / `0.0`
- confidence unique count: `0`
- decision_basis valid rate: `0.0`
- decision_basis distribution: `{"insufficient_information": 0, "strong_match": 0, "unrelated": 0, "weak_match": 0}`

## Calibration

- best calibrated score: `NA`
- best calibrated ECE: `NA`
- best calibrated Brier: `NA`

## Interpretation

- collapse_type: `format_failure`
- recommendation: `needs_prompt_redesign`

If `recommend_true_rate` remains above 0.9, decision collapse is still unresolved and Day1c should not be scaled to full Beauty. If the decision rate becomes reasonable but confidence remains nearly constant, the decision prompt may be usable but verbalized scalar confidence is not; the next route should be logit/probability confidence or self-consistency rather than more confidence wording.
