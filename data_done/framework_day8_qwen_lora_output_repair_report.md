# Framework-Day8 Qwen-LoRA Output Repair Report

## 1. Day7 Recap

Day7 showed that the Day6 Beauty listwise Qwen-LoRA adapter has a small but real signal on 512 samples: NDCG@10 `0.5747`, MRR `0.4197`, and HR@1 `0.2559`, compared with random NDCG@10 `0.5532`, MRR `0.4114`, and HR@1 `0.1680`. However, parse/schema stability remained weak: parse success `0.8320` and schema validity `0.8027`.

## 2. Failure Taxonomy

The observed failures are not dominated by a single harmless parser alias. They include missing or malformed JSON, incomplete ranked lists, schema drift, and outputs that do not exactly match the expected `ranked_item_ids` contract. This means parser repair is useful for robustness, but it is unlikely to fully solve the baseline weakness by itself.

## 3. Parser-Only Repair

Parser-only repair on existing Day7 raw outputs barely changed the result:

- parse success: `0.8320 -> 0.8340`
- NDCG@10: `0.5747 -> 0.5753`
- MRR: `0.4197 -> 0.4206`

This confirms that the main issue is not simply an overly strict parser.

## 4. Generation Config Repair

Generation should be deterministic for evaluation. Day9 removes sampling-only generation flags from deterministic inference: `do_sample=false` is kept, while `temperature`, `top_p`, and `top_k` are not passed as generation kwargs. `max_new_tokens`, EOS, and PAD IDs remain explicit.

## 5. Strict Prompt Re-Eval

Strict prompt plus repaired parser made the existing Day6 adapter worse:

- NDCG@10: `0.5618`
- MRR: `0.3995`
- parse success: `0.8086`
- schema valid: `0.7910`

This is consistent with prompt mismatch: the adapter was trained with the original listwise prompt, so replacing only the inference prompt with a stricter JSON-only variant changes the input distribution.

## 6. Base Qwen Comparison

Base-Qwen comparison remains optional because the current Day8 question was parser/prompt repair. The key result is already clear from the adapter-only comparison: strict inference prompt without matching strict training is not the right fix.

## 7. Day9 Recommendation

Day9 should repair the baseline formulation rather than continue parser-only work:

- keep `listwise-v1` as the existing Day6 baseline;
- train `listwise-v2` with the same JSON-strict prompt used at inference;
- train `pointwise-v1` as a candidate relevance baseline, then aggregate per-candidate scores into rankings.

Do not enter confidence/evidence/CEP framework until the Qwen-LoRA recommendation baseline is more stable.

## 8. Boundary

Day8 performs parser/generation/prompt diagnosis only. It does not train a new adapter, call APIs, or implement confidence/evidence/CEP fusion.
