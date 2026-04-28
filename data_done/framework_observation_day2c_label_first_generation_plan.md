# Framework-Observation-Day2c Label-First Generation Fallback Plan

Status: plan only. Do not run unless Day2b still fails output control.

## Motivation

Day2 exposed placeholder/schema-following failure in candidate-grounded title generation. Day2b removes placeholder examples and requires exact candidate-title copying. If the model still outputs placeholders, explanations, empty titles, or invented titles, the next repair should add a small label-first control channel while preserving title generation.

## Proposed Prompt Format

Input:
- User history.
- Candidate pool with labels A-F and full candidate titles.

Output:
```json
{"selected_label":"C","recommended_title":"Exact Candidate Title","confidence":0.0}
```

Rules:
- `selected_label` must be one of A-F.
- `recommended_title` must exactly copy the title associated with `selected_label`.
- Confidence remains raw verbalized confidence, not calibrated probability.
- No evidence fields.
- No CEP fields.
- No explanations.

## Why This Still Counts As Generative Recommendation

The model still has to produce the recommended title string, and evaluation remains title-level catalog grounding. The label is only an output-control scaffold to prevent placeholder or invented titles.

## Evaluation

Use the same Day2b metrics:
- `generation_valid_rate`
- `placeholder_title_rate`
- `candidate_title_exact_match_rate`
- `matched_title_hit_rate`
- `hallucination_rate`
- confidence ECE/Brier/AUROC for matched-title correctness

Decision:
- If label-first fixes output validity but hit rate remains low, the branch is viable but recommendation quality is weak.
- If label-first also fails, candidate-grounded generation needs stronger constrained decoding or a different base instruction format before larger smoke runs.
