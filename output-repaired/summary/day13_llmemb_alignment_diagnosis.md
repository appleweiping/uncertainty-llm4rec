# Day13 LLMEmb Alignment Diagnosis

## Status

Blocked before real candidate-score export.

Missing LLMEmb requirements:

- `handled_dir`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled`
- `interaction_file`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\inter.txt`
- `llm_embedding`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\0722_avg_pca.pkl`
- `srs_embedding`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\itm_emb_sasrec.pkl`
- `checkpoint`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\saved\beauty\llmemb_sasrec\llmemb\pytorch_model.bin`

## 1. LLMEmb Required Data Format

LLMEmb reads sequential data from:

`external/LLMEmb/data/<dataset>/handled/<inter_file>.txt`

Each line is expected to contain integer-mapped user and item ids:

`user_id item_id`

The repository README also expects:

- `data/<dataset>/handled/id_map.json`
- `data/<dataset>/handled/<llm_emb_file>.pkl`
- `data/<dataset>/handled/itm_emb_sasrec.pkl`
- `saved/<dataset>/<model_name>/<check_path>/pytorch_model.bin`

The evaluation loader constructs each test candidate pool as:

`item_indices = [positive_item] + sampled_negative_items`

and the true score is produced by:

`self.model.predict(**inputs)`

inside `external/LLMEmb/trainers/sequence_trainer.py`.

## 2. Our Current Beauty Processed Data

Available local project files:

- `data/processed/amazon_beauty/test.jsonl`: `D:\Research\Uncertainty-LLM4Rec\data\processed\amazon_beauty\test.jsonl` exists=True
- `data/processed/amazon_beauty/ranking_test.jsonl`: `D:\Research\Uncertainty-LLM4Rec\data\processed\amazon_beauty\ranking_test.jsonl` exists=True
- `Day9 evidence`: `D:\Research\Uncertainty-LLM4Rec\output-repaired\beauty_deepseek_relevance_evidence_full\calibrated\relevance_evidence_posterior_test.jsonl` exists=True

Our Day9 evidence uses raw Amazon-style ids such as `user_id` and `candidate_item_id`. LLMEmb internally expects contiguous integer ids starting from 1 after its own preprocessing. Therefore, a direct join is only safe if `id_map.json` exposes a reversible mapping between raw ids and mapped integer ids, or if the LLMEmb export writes both raw and mapped ids.

## 3. User/Item Alignment

Current state:

- LLMEmb repo is cloned at `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb`.
- LLMEmb Beauty handled data is not present.
- LLMEmb checkpoint is not present.
- The external repo cannot currently produce real backbone scores locally.

Required Day13/Day14 conversion plan:

1. Build `external/LLMEmb/data/beauty/handled/inter.txt` from a Beauty split whose raw user/item ids can be mapped back to our Day9 ids.
2. Preserve `raw_user_id -> mapped_user_id` and `raw_item_id -> mapped_item_id`.
3. Obtain or generate `0722_avg_pca.pkl` and `itm_emb_sasrec.pkl`, or run a base non-LLMEmb backbone first if the goal is only score-export smoke.
4. Train/load LLMEmb checkpoint at `external/LLMEmb/saved/beauty/llmemb_sasrec/llmemb/pytorch_model.bin`.
5. Export candidate scores with both raw and mapped ids.
6. Join with Day9 evidence on raw `user_id + candidate_item_id`.

## 4. Candidate Pool Compatibility

LLMEmb's default evaluation uses one positive plus `test_neg` sampled negatives. Our Day9 evidence already covers the Beauty candidate rows used by our project. To avoid low join coverage, the preferred path is to make LLMEmb evaluate the same candidate pools as Day9, not random new negatives. If using LLMEmb's random negatives, the generated negative candidate ids may not exist in Day9 evidence and join coverage will drop.

## 5. Can We Join Without Regenerating Day9 Evidence?

Yes, but only if the LLMEmb candidate export uses the same raw user/item ids or a reversible mapping back to them. If the LLMEmb export only contains integer ids, Day9 evidence cannot be joined safely.
