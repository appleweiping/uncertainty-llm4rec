# Framework-Observation-Day2c Label-Only Fallback Plan

Status: plan only. Do not run unless Day2c-repair still fails parse/schema/generation validity.

## Motivation

Day2c showed that long exact-title copying can truncate JSON when the token budget is too small. Day2c-repair keeps title generation as the main line by increasing token budget and moving `recommended_title` to the last field. If repair still fails, a stricter output-control fallback is label-only generation.

## Fallback Schema

```json
{"selected_label":"A","confidence":0.74}
```

The parser then copies the exact candidate title associated with `selected_label`.

## Why This Is Only A Fallback

This version is more stable, but it weakens the generative-title requirement. It becomes closer to closed-catalog label selection, even though the final evaluation can still attach the selected title. Therefore it should be used as a control/fallback, not as the primary generative title observation.

## Evaluation

Use the same Beauty 100 valid/test subset and report:
- parse/schema validity
- label validity
- selected-label hit rate
- generated/copied title validity
- confidence ECE/Brier/AUROC for hit

If label-only succeeds while title-generation repair fails, the immediate conclusion is that candidate-grounded selection is controllable, but exact title generation needs constrained decoding or shorter catalog handles before uncertainty observation can proceed.
