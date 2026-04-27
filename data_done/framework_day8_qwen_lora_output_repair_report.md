# Framework-Day8 Qwen-LoRA Output Repair Report

## 1. Day7 Recap

Day7 showed Qwen-LoRA has some signal on 512 samples, but parse/schema stability remains weak. Day8 repairs output handling before any longer training or CEP integration.

## 2. Failure Taxonomy

- non-JSON rate: `NA`
- extra explanatory text rate: `NA`
- missing ranked_item_ids rate: `NA`
- alternative key rate: `NA`
- incomplete list rate: `NA`
- title/non-ID output rate: `NA`

## 3. Parser-Only Repair

- raw parse success: `NA`
- repaired parse success: `NA`
- raw NDCG@10: `NA`
- repaired NDCG@10: `NA`
- raw MRR: `NA`
- repaired MRR: `NA`

## 4. Generation Config Repair

Generation is deterministic: `do_sample=false`; no `temperature`, `top_p`, or `top_k`; `max_new_tokens=128`; EOS/PAD IDs are set from the tokenizer.

## 5. Strict Prompt Re-Eval

- strict prompt status: `skipped`
- strict NDCG@10: `NA`
- strict MRR: `NA`
- strict parse success: `NA`
- strict schema valid: `NA`

## 6. Base Qwen Comparison

Status: `skipped_due_to_runtime`.

## 7. Day9 Recommendation

If parse success reaches at least `0.95` and strict prompt metrics clearly beat random, proceed to longer Beauty listwise training. If parse improves but metrics remain weak, revise training length/target before entering CEP. If parse remains poor, further prompt/schema repair is needed.

## 8. Boundary

Day8 performs parser/generation/prompt repair only. It does not train, call APIs, or implement confidence/evidence/CEP framework.
