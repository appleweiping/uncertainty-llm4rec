# Framework-Day7 Prompt / Parser Repair Plan

## Diagnosis Inputs

- parse success rate: `0.8671875`
- schema valid rate: `0.8515625`
- empty output rate: `0.0`
- non-JSON rate: `0.1328125`
- missing `ranked_item_ids` rate: `0.015625`
- too-few-items rate: `0.0`
- extra-text rate: `0.0234375`

## Prompt Repair Candidates

1. Make JSON-only instruction stricter and explicitly say: output exactly one JSON object and no explanation.
2. Require `ranked_item_ids` length to equal the candidate pool size.
3. Restate that values must be candidate_item_id strings only, not titles.
4. Keep deterministic generation: `do_sample=false`; avoid temperature/top-p unless needed.
5. Keep max_new_tokens large enough for six IDs; 128 is sufficient for 5neg.

## Parser Repair Candidates

Parser-only compatibility is low-risk if raw outputs commonly use alternate keys. Accept possible aliases such as `ranked_items`, `recommendations`, `ranking`, or `item_ids`, then re-validate against the candidate pool. This should be reported as parser repair, not method improvement.

## Training Boundary

Do not change training data format or enter CEP/confidence/evidence framework during Day7. First determine whether the weak Day6 result is parser/prompt/generation instability or insufficient LoRA training.
