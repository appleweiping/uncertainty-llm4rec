# Framework-Day3 Qwen-LoRA Baseline Task Design

## Boundary

LoRA trains a local Qwen3-8B recommendation baseline. CEP is the framework layer that calibrates relevance posterior and applies evidence-risk decision support on top of model/backbone outputs.

## Format A: Listwise Ranking Instruction

One sample contains user history plus a closed candidate pool. The target is the positive candidate ID / ranked candidate IDs. This is closest to recommender ranking and makes catalog constraints explicit.

Pros: natural ranking baseline, easy to evaluate with NDCG/MRR/HR@1/HR@3, aligns with closed-candidate evaluation, avoids catalog-free hallucination. Cons: fewer training samples and longer context.

## Format B: Pointwise Relevance Instruction

One sample contains user history plus one candidate item. The target is a raw relevance label. This is closer to CEP evidence/relevance posterior, but the LoRA target is still raw relevance, not calibrated probability.

Pros: many samples, simple binary objective, easy to connect to CEP later. Cons: less like a standalone recommender, needs candidate aggregation for ranking, and calibration must remain separate.

## Recommendation

Use listwise as the first Qwen-LoRA recommendation baseline. Keep pointwise as a bridge to CEP/evidence generator design. Do not train LoRA directly on calibrated relevance probabilities; calibration is a valid-set framework operation.
