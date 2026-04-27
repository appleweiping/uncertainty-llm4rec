# Framework-Observation-Day1c Decision-Forced Confidence Report

## Scope

Day1c is a local Qwen-LoRA confidence elicitation smoke on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or training.

## Prediction Directory

`output-repaired\framework_observation\beauty_qwen_lora_confidence_decision_forced\predictions`

## Decision / Confidence Diagnostics

- test rows / valid rows: `200` / `200`
- test parse success: `1.0`
- test schema valid: `1.0`
- test accuracy: `0.785`
- test AUROC: `0.29558583913494296`
- test ECE: `0.04244999999999999`
- test Brier: `0.1761945`
- recommend true / false rate: `0.165` / `0.835`
- confidence mean / std: `0.8271499999999999` / `0.01781509191668683`
- confidence unique count: `3`
- decision_basis valid rate: `1.0`
- decision_basis distribution: `{"insufficient_information": 0, "strong_match": 33, "unrelated": 167, "weak_match": 0}`

## Calibration

- best calibrated score: `logistic_calibrated_confidence`
- best calibrated ECE: `0.02978774715311816`
- best calibrated Brier: `0.16899198210470728`

## Interpretation

- collapse_type: `low_variance_decision_anchored_confidence`
- recommendation: `switch_to_logit_confidence`

Decision collapse is fixed when `recommend_true_rate` drops out of the >0.9 regime. In this run, scalar verbalized confidence remains low-variance and not informative, so Day1c should not be scaled to full Beauty. The next route should be logit/probability confidence or self-consistency rather than more confidence wording.

Summary: decision collapse fixed, but scalar verbalized confidence remains low-variance and not informative.
