# Framework-Day3 DecodingMatters Audit

## Summary

DecodingMatters is useful as an implementation reference for local LLM recommendation training and constrained decoding, but it should not be copied as our method. Our contribution remains CEP / calibrated relevance posterior / uncertainty-aware decision support, not LoRA itself.

## Findings

1. `train.py` expects CSV `train_file` and `eval_file`. The dataset rows include columns such as `history_item_title`, `history_item_id`, `item_title`, and `item_id`.
2. `D3Dataset` converts each row into an instruction prompt with user history titles and a target item title as the response. Labels are causal-LM tokens for the target response.
3. The code references `candidate_file=train_file` in `train.py`, but the visible `D3Dataset` constructor does not consume a separate candidate file. Candidate/item catalog constraints mainly appear during evaluation through `info_file`.
4. `category` maps dataset names to natural-language category words. `K` is passed through but is not materially used in the visible dataset prompt path. `version` appears in `train.py` signature but is not used.
5. `base_model` is loaded with `AutoModelForCausalLM.from_pretrained`; tokenizer is loaded separately.
6. The visible training code uses HuggingFace `Trainer` full model training. It does not instantiate PEFT/LoRA adapters in `train.py`.
7. Training output is a saved model directory via `model.save_pretrained(output_dir)`.
8. Evaluation generates item titles with beam search and maps generated titles back to item IDs via `info_file`; metrics are computed by exact title match in `calc.py`.
9. Decoding/framework intervention is in `evaluate.py` and `LogitProcesser.py`: constrained decoding limits next tokens to catalog item-title prefixes, and optional CF logits alter token scores.
10. We can reference its dataset/trainer/evaluation separation and constrained closed-catalog idea. We should not copy its title-generation target, exact-title metric, or CF-logit decoding as our contribution.
