# Framework-Day1 Protocol Audit

This audit uses local project notes and cloned external repositories. When a local README does not specify a detail, it is marked not specified rather than inferred.

| paper_name | domain/dataset | filtering rule | minimum user interactions | minimum item interactions | k-core | split protocol | negative sampling protocol | evaluation candidate size | leave-one-out | prefix-style | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OpenP5 | Beauty, Movies, Electronics and other public rec datasets | Repo reports fixed preprocessed dataset statistics; README does not specify detailed filtering in the visible local notes. | not specified | not specified | not specified | sequential recommendation supported; local README points to generated dataset scripts. | not specified in local README | not specified in local README | not specified in local README | supported by task generation, but not specified as our eval protocol | External repo requires generated data/checkpoints for generative scoring, so we use it only as protocol context. |
| LLM-ESR | Yelp, fashion, beauty | README states preprocessing filters cold-start users and items. | not specified | not specified | cold-start filtering mentioned; exact k not specified | Uses handled inter_seq format for sequential recommendation. | not specified in local README | not specified in local README | not specified | not specified | Useful evidence that ID-based sequential backbones need cold-start filtering. |
| Project Day20-Day41 CEP validation | Beauty full and small domains | Existing CEP experiments used leave-one-out style user histories and pointwise candidate pools. | at least 3 for train history + valid + test; Framework-Day1 compares >=4/>=5. | not forced in main CEP split | not used for CEP observation-stage main data | chronological leave-one-out, last item test and second last item valid | 1 positive + 5 negatives in primary continuity setting; 20neg variants available for non-trivial HR@10 | 6 in continuity setting; HR@10 trivial and not primary | yes | not for evaluation; can be derived later for LoRA training | This is the closest implemented protocol to preserve continuity while making data_done cleaner. |

## Setting Comparison

- Setting A, user-history leave-one-out: keep users with enough chronological interactions, use the last item as test, the second last item as valid, and previous items as train history. This is the cleanest match for our user-history plus candidate-item CEP schema.
- Setting B, k-core plus leave-one-out: iteratively filters users/items so both sides have enough observations. It is common in recommendation papers but can substantially alter scale and remove long-tail/cold-start behavior.
- Setting C, iterative prefix / multi-instance: useful for local generator or LoRA training data, but it should be derived from train sequences rather than used as the primary evaluation split.

## Recommendation

Framework-Day1 uses Setting A as the first recommended data foundation: user_min4 plus chronological leave-one-out, max 10,000 users per domain, and warm negative sampling for ID-backbone compatibility. Strategy B/C statistics are still reported. Prefix-style examples should be generated later for LoRA training while preserving leave-one-out evaluation.