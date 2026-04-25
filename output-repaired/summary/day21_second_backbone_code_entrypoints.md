# Day21 Second Backbone Code Entrypoints

## Selected Repo

LLM-ESR: `external/LLM-ESR`

## Data Reading

Official data preprocessing is described in `external/LLM-ESR/README.md` and `external/LLM-ESR/data/data_process.py`. The full official pipeline expects handled files such as `inter.txt`, `itm_emb_np.pkl`, `usr_emb_np.pkl`, `pca64_itm_emb_np.pkl`, and `sim_user_100.pkl`.

For Day21 smoke, we do not use the full LLM-ESR enhancement pipeline. We use the external repo's `GRU4Rec` class with our existing Beauty train split.

## Training / Eval Entry

- Official main: `external/LLM-ESR/main.py`
- Official Beauty command: `external/LLM-ESR/experiments/beauty.bash`
- Official trainer: `external/LLM-ESR/trainers/sequence_trainer.py`

## Candidate Score / Logit Entry

`external/LLM-ESR/models/GRU4Rec.py`

The key method is:

`GRU4Rec.predict(seq, item_indices, positions, **kwargs)`

It returns candidate logits with shape `(batch, num_candidates)`, which can be flattened into:

`user_id, candidate_item_id, backbone_score, label`

## Metrics Entry

Official ranking evaluation is in `external/LLM-ESR/trainers/sequence_trainer.py`, which ranks the positive item against negatives and calls `metric_report`. Day21 uses the project evaluator to keep HR@10, NDCG@10, MRR@10, and Recall@10 consistent with Day17-Day20.

## Export Strategy

The adapter trains LLM-ESR GRU4Rec on Beauty train positives only, maps user history titles to item IDs through `items.csv`, scores every candidate in the Day9 evidence-aligned 100-user pool, and writes `output-repaired/backbone/second_backbone_beauty_100/candidate_scores.csv`.
