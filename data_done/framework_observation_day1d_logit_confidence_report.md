# Framework-Observation-Day1d Logit-Based Confidence Report

## Scope

Day1d is a local Qwen-LoRA Beauty 200/200 smoke. It does not train, use evidence, use CEP, call external APIs, or run four domains.

The model is prompted only for a binary `recommend` decision. Confidence is extracted from model token probabilities for `true` versus `false`, not from verbalized scalar confidence.

## Prediction Directory

`output-repaired/framework_observation/beauty_qwen_lora_logit_confidence/predictions`

## Relevance / Label Prediction

- test rows / valid rows: `200` / `200`
- parse/schema: `1.0` / `1.0`
- recommend true rate: `0.04`
- accuracy at threshold 0.5: `0.83`
- positive relevance AUROC: `0.5887668320340185`
- positive relevance Brier: `0.1429006312394226`
- positive relevance ECE: `0.11125464997727197`
- positive relevance score mean/std: `0.13393764701965` / `0.17010084593581187`

## Decision Correctness Confidence

- correctness AUROC: `0.5965627214741318`
- correctness Brier: `0.1429006312394226`
- correctness ECE: `0.0847636449604948`
- high-confidence error rate: `0.11`
- decision confidence mean/std/min/max: `0.8839778485314717` / `0.12448676995862419` / `0.5094134374789216` / `0.9941465518261811`
- confidence unique count: `200`

## Calibration

- best relevance calibrated score: `logistic_calibrated` with ECE `0.034089472091289215` and Brier `0.1345724812871104`
- best correctness calibrated score: `logistic_calibrated` with ECE `0.03629381237551008` and Brier `0.13772766390758376`

## Interpretation

- collapse_type: `usable_miscalibrated_signal`
- recommendation: `switch_to_logit_confidence`

Logit/token probability fixes scalar verbalized confidence collapse and gives a usable-but-weak miscalibrated signal. However, the hard recommend=true/false decision remains conservative and under-recommending at the default 0.5 threshold, so we should not full-run yet. The next audit should treat P(true) as a continuous relevance score rather than over-reading the fixed-threshold hard decision.
