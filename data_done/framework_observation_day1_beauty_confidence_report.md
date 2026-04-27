# Framework-Observation-Day1 Beauty Local Qwen-LoRA Confidence Report

## Scope

This is a local Qwen/Qwen-LoRA confidence observation on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or ranking-baseline repair.

## Prediction Directory

`output-repaired\framework_observation\beauty_qwen_lora_confidence\predictions`

## Raw Confidence Diagnostics

- status: `pending_predictions`
- test parse success: `0.0`
- test schema valid: `0.0`
- test accuracy: `0.0`
- test ECE: `0.0`
- test Brier: `0.0`
- test AUROC: `NA`
- test high-confidence error rate: `0.0`

## Calibration

- best available calibrated score: `NA`
- best calibrated ECE: `NA`
- best calibrated Brier: `NA`

Calibration is fit on valid and evaluated on test. Raw confidence is verbalized confidence, not a calibrated probability.

Metrics calibrate confidence as confidence in the model's binary decision being correct (`recommend == label`), not as direct `P(relevant)`.

## Relation To Week1-Week4

This stage is a cleaner local-framework continuation of the earlier confidence observation. If raw confidence is informative but miscalibrated, it supports bringing confidence calibration into the later framework design.

## Recommendation

If parse/schema are stable and calibration reduces ECE/Brier, proceed to four-domain local confidence observation. If parsing is weak, repair the confidence prompt/parser before scaling.
