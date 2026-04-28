# Framework-Observation-Day1 Beauty Local Qwen-LoRA Confidence Report

## Scope

This is a local Qwen/Qwen-LoRA confidence observation on `data_done/beauty` 5neg pointwise samples. It does not use external APIs, evidence fields, CEP fusion, or ranking-baseline repair.

## Prediction Directory

`output-repaired/framework_observation/beauty_qwen_lora_confidence_refined_vllm/predictions`

## Raw Confidence Diagnostics

- status: `ready_for_interpretation`
- test parse success: `1.0`
- test schema valid: `1.0`
- test accuracy: `0.19230769230769232`
- test ECE: `0.45769230769230773`
- test Brier: `0.36480769230769233`
- test AUROC: `0.5`
- test high-confidence error rate: `0.0`

## Confidence Collapse / Saturation Diagnostics

- collapse status: `low_unique_confidence_values`
- test confidence mean: `0.65`
- test confidence std: `0.0`
- test confidence min/max: `0.65` / `0.65`
- test confidence unique count: `1`
- test confidence at 1.0 rate: `0.0`
- test confidence >= 0.90 rate: `0.0`
- test confidence >= 0.97 rate: `0.0`
- recommend true/false rate: `0.9807692307692307` / `0.019230769230769232`
- confidence correct/wrong mean: `0.65` / `0.65`
- confidence gap correct-minus-wrong: `0.0`

If confidence mass concentrates near `0.97` or `1.0`, this should be interpreted as confidence collapse/saturation, not as method success. If confidence has meaningful variance but ECE/Brier are poor, the signal is informative but miscalibrated.

## Calibration

- best available calibrated score: `isotonic_calibrated_confidence`
- best calibrated ECE: `0.0`
- best calibrated Brier: `0.15532544378698226`
- interpretation: valid-set calibration reduces ECE/Brier and makes the raw confidence signal more usable.

Calibration is fit on valid and evaluated on test. Raw confidence is verbalized confidence, not a calibrated probability.

Metrics calibrate confidence as confidence in the model's binary decision being correct (`recommend == label`), not as direct `P(relevant)`.

## Relation To Week1-Week4

This stage is a cleaner local-framework continuation of the earlier confidence observation. If raw confidence is informative but miscalibrated, it supports bringing confidence calibration into the later framework design.

## Recommendation

If the original prompt shows confidence collapse/saturation, run the optional Day1b refined 200/200 smoke before spending more full-runtime. If parse/schema are stable, confidence is not collapsed, and calibration reduces ECE/Brier, proceed to broader local confidence observation. If parsing is weak, repair the confidence prompt/parser before scaling.
