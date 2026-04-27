# Framework-Observation-Day1b Refined Confidence Prompt Plan

## Purpose

Day1b is an optional measurement refinement for local Qwen-LoRA raw verbalized confidence. It does not replace the original Day1 prompt and is not a CEP method contribution.

## Trigger

Run the refined prompt only if the original prompt shows confidence collapse/saturation or very low confidence diversity, for example high `confidence_ge_0.97_rate`, high `confidence_at_1_rate`, or low `confidence_unique_count`.

## Refined Schema

```json
{
  "recommend": true,
  "confidence_level": "medium",
  "confidence": 0.65,
  "reason": "one short sentence"
}
```

The refined prompt separates confidence elicitation from recommendation relevance and explicitly warns that confidence is not a relevance score, popularity score, fluency score, calibrated probability, or evidence field.

## Confidence Level Mapping

- `low`: 0.50 to 0.60
- `medium`: 0.60 to 0.75
- `high`: 0.75 to 0.90
- `very_high`: 0.90 to 0.97

The prompt discourages `confidence=1.0` because recommendation judgments are not logically certain events.

## Smoke Protocol

Run Beauty only, using `data_done/beauty` 5neg:

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_refined.yaml --split valid --model_variant lora --max_samples 200 --resume
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_refined.yaml --split test --model_variant lora --max_samples 200 --resume
python main_framework_observation_day1_confidence_analysis.py --pred_dir output-repaired/framework_observation/beauty_qwen_lora_confidence_refined/predictions
```

Do not run four domains until Beauty refined smoke confirms stable parsing and non-collapsed confidence distribution.

## Interpretation

If refined confidence has more useful variance while raw ECE/Brier remain poor, the correct conclusion is `informative but miscalibrated`. If valid-set calibration reduces ECE/Brier, the confidence signal becomes more usable for the later framework.
