# Framework-Day7 Prompt / Parser Repair Plan

## Diagnosis Inputs

- parse success rate: `NA`
- schema valid rate: `NA`
- empty output rate: `None`
- non-JSON rate: `None`
- missing `ranked_item_ids` rate: `None`
- too-few-items rate: `None`
- extra-text rate: `None`

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
