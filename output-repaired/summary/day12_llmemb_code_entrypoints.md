# Day12/Day13 LLMEmb Code Entrypoint Audit

## Status

LLMEmb has now been cloned locally.

Repository:

`https://github.com/Applied-Machine-Learning-Lab/LLMEmb.git`

Local path:

`external/LLMEmb/`

Commit hash:

`3458a5e225062e94b4f1a01e41f3ec82089f0407`

Remote:

`origin https://github.com/Applied-Machine-Learning-Lab/LLMEmb.git`

The repository is readable and Beauty is an official dataset option, but the local clone does not include handled Beauty data, LLM embedding files, SRS item embeddings, or trained checkpoint. Therefore Day13 can locate the true score path and prepare the exporter, but cannot honestly produce real LLMEmb candidate scores yet.

## 1. Main Training And Evaluation Entry

Main recommendation-stage entry:

`external/LLMEmb/main.py`

Control flow:

```python
generator = Seq2SeqGenerator(args, logger, device)  # for sasrec_seq / llmemb_sasrec
trainer = SeqTrainer(args, logger, writer, device, generator)

if args.do_test:
    trainer.test()
else:
    trainer.train()
```

Beauty experiment script:

`external/LLMEmb/experiments/beauty/srs/llmemb.bash`

It runs:

```bash
python main.py --dataset beauty \
  --model_name llmemb_sasrec \
  --hidden_size 128 \
  --train_batch_size 128 \
  --max_len 200 \
  --check_path llmemb \
  --freeze_emb \
  --llm_emb_file 0722_avg_pca
```

## 2. Data Format And Beauty Config

LLMEmb reads Beauty interactions from:

`external/LLMEmb/data/beauty/handled/<inter_file>.txt`

Default `inter_file` is `inter`, so the expected file is:

`external/LLMEmb/data/beauty/handled/inter.txt`

Each line is expected to be:

`mapped_user_id mapped_item_id`

The README also mentions `id_map.json`, but the runtime loader itself mainly consumes the integer interaction file. For our plug-in join, `id_map.json` or an equivalent reversible map is still required because Day9 evidence uses raw Amazon ids, while LLMEmb uses contiguous integer ids.

Required local files for LLMEmb-SASRec Beauty inference:

| Requirement | Expected path | Current status |
|---|---|---|
| handled interactions | `external/LLMEmb/data/beauty/handled/inter.txt` | missing |
| id map | `external/LLMEmb/data/beauty/handled/id_map.json` | missing |
| LLM item embedding | `external/LLMEmb/data/beauty/handled/0722_avg_pca.pkl` | missing |
| SRS item embedding | `external/LLMEmb/data/beauty/handled/itm_emb_sasrec.pkl` | missing |
| checkpoint | `external/LLMEmb/saved/beauty/llmemb_sasrec/llmemb/pytorch_model.bin` | missing |

## 3. Candidate Pool Construction

Evaluation data loader:

`external/LLMEmb/generators/generator.py`

Relevant path:

```python
def make_evalloader(self, test=False):
    if test:
        eval_dataset = concat_data([self.train, self.valid, self.test])
    else:
        eval_dataset = concat_data([self.train, self.valid])
    self.eval_dataset = SeqDataset(eval_dataset, self.item_num, self.args.max_len, self.args.test_neg)
```

Candidate construction:

`external/LLMEmb/generators/data.py::SeqDataset.__getitem__`

Each row returns:

```python
seq, pos, neg, positions
```

where `pos` is the positive target item and `neg` is an array of sampled negative items. With default `test_neg=100`, each user has 101 candidates: 1 positive + 100 negatives.

Important alignment implication:

LLMEmb's default negatives are sampled internally and may not match our Day9 Beauty candidate pool. For high join coverage, Day13/Day14 should either force LLMEmb to evaluate the same candidate pools as Day9 or export raw ids and regenerate Day9 evidence for the same LLMEmb candidate set.

## 4. Score / Logits Generation Location

Main eval function:

`external/LLMEmb/trainers/sequence_trainer.py::SeqTrainer.eval`

Exact score-producing lines:

```python
inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
pred_logits = -self.model.predict(**inputs)

per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
```

This is the correct insertion point for candidate score export. The current code only keeps the positive item's rank, then discards the full candidate score matrix. For our plug-in, we must export every candidate in `item_indices`.

Model-level score function examples:

`external/LLMEmb/models/SASRec.py::SASRec.predict`

```python
log_feats = self.log2feats(seq, positions)
final_feat = log_feats[:, -1, :]
item_embs = self._get_embedding(item_indices)
logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
return logits
```

`external/LLMEmb/models/Bert4Rec.py` and `external/LLMEmb/models/GRU4Rec.py` expose analogous `predict` functions.

## 5. Metric Computation Location

Metric function:

`external/LLMEmb/utils/utils.py::metric_report`

It computes:

```python
NDCG@10
HR@10
```

from the positive item rank. MRR and Recall are not directly reported by the default utility, so our plug-in evaluator should compute HR@10, NDCG@10, MRR@10, and Recall@10 after exporting candidate-level rows.

## 6. Can LLMEmb Export Candidate Scores?

Yes, structurally. The model already computes a score for every candidate item in `item_indices`. The smallest non-invasive export path is:

1. Load model/checkpoint exactly as `SeqTrainer.test()` does.
2. Iterate over `trainer.test_loader`.
3. Build `item_indices = [pos] + neg`.
4. Call `self.model.predict(**inputs)`.
5. Write every candidate score before rank aggregation.

Prepared local adapter:

`main_day13_llmemb_score_export.py`

Target output:

`output-repaired/backbone/llmemb_beauty_100/candidate_scores.csv`

Required schema:

`user_id, candidate_item_id, backbone_score, label`

Optional schema:

`backbone_rank, split, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success`

## 7. Current Blocker

The exporter was run and correctly stopped before producing scores because required LLMEmb files are missing. The current blocked report is:

`output-repaired/summary/day13_llmemb_score_export_report.md`

The alignment diagnosis is:

`output-repaired/summary/day13_llmemb_alignment_diagnosis.md`

No synthetic scores were generated, and Day9 relevance probabilities were not used as LLMEmb scores.

## 8. Next Conversion Plan

The next real step is not to tune prompts or rerank formulas. It is to create a reversible ID-aligned LLMEmb Beauty eval set:

1. Convert our Beauty raw ids to LLMEmb integer ids while preserving `raw_user_id` and `raw_item_id`.
2. Ensure LLMEmb evaluates the same candidate pools as Day9, or explicitly accept low join coverage and regenerate evidence later.
3. Obtain/generate `0722_avg_pca.pkl` and `itm_emb_sasrec.pkl`.
4. Train/load the LLMEmb checkpoint.
5. Run `main_day13_llmemb_score_export.py`.
6. Run `main_day12_llmemb_plugin_smoke.py`.

Only after `join_coverage >= 0.8` should plug-in performance be interpreted.
