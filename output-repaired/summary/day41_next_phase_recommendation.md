# Day41 Next Phase Recommendation

## Option A: Cross-Domain Continuation

Regular medium domains are more realistic, but they exposed cold-start issues for ID-only sequential backbones. Continuing this path requires a content-aware or cold-aware backbone rather than directly reusing SASRec/GRU4Rec/Bert4Rec. Good next steps include a stronger text/content carrier, a limited Movies/Books/Electronics medium_20neg run, or a public content-aware recommender that can export candidate-level scores.

## Option B: Qwen-LoRA Framework

Qwen-LoRA is not just API replacement. It requires designing a local evidence generator framework: training data, schema alignment, losses for relevance/evidence fields, calibration evaluation, and downstream plug-in validation. This direction improves system ownership and cost control but is a new method-development phase.

## Recommendation By Goal

If the goal is paper-mainline closure, first write and polish the paper around the current evidence: Beauty full three-backbone multi-seed as primary performance, CEP calibration as core method, robustness as support, and small-domain results as sanity/continuity.

If the goal is continuing system development, prioritize Qwen-LoRA after freezing the current claims.

If the goal is stronger cross-domain evidence, prioritize a content-aware/cold-aware backbone for regular medium domains rather than forcing ID-only sequential backbones into cold candidate pools.
