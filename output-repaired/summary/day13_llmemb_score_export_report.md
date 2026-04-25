# Day13 LLMEmb Score Export Report

## Status

`blocked`

LLMEmb repo:

`D:\Research\Uncertainty-LLM4Rec\external\LLMEmb`

Commit hash:

`3458a5e225062e94b4f1a01e41f3ec82089f0407`

Target score export:

`D:\Research\Uncertainty-LLM4Rec\output-repaired\backbone\llmemb_beauty_100\candidate_scores.csv`

## Required Files

| Requirement | Path | Exists |
|---|---|---|
| repo | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb` | True |
| handled_dir | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled` | False |
| interaction_file | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\inter.txt` | False |
| id_map | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\id_map.json` | False |
| llm_embedding | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\0722_avg_pca.pkl` | False |
| srs_embedding | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\itm_emb_sasrec.pkl` | False |
| checkpoint | `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\saved\beauty\llmemb_sasrec\llmemb\pytorch_model.bin` | False |

## Located Score Entrypoint

Training/evaluation entry:

`external/LLMEmb/main.py`

Trainer:

`external/LLMEmb/trainers/sequence_trainer.py`

Candidate score generation:

```python
inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
pred_logits = -self.model.predict(**inputs)
```

The model-level candidate score comes from each model's `predict` method. For SASRec-style models, this computes:

```python
item_embs = self._get_embedding(item_indices)
logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
```

Metric computation:

`external/LLMEmb/utils/utils.py::metric_report`

LLMEmb currently keeps only the positive item's rank for HR@10/NDCG@10. To export a plug-in candidate table, the adapter must write every candidate in `item_indices`, not only the final top-10.

## Missing Blockers

- `handled_dir`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled`
- `interaction_file`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\inter.txt`
- `llm_embedding`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\0722_avg_pca.pkl`
- `srs_embedding`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\data\beauty\handled\itm_emb_sasrec.pkl`
- `checkpoint`: `D:\Research\Uncertainty-LLM4Rec\external\LLMEmb\saved\beauty\llmemb_sasrec\llmemb\pytorch_model.bin`

## Outcome

No synthetic scores were generated. If status is `blocked`, the next step is to prepare handled Beauty data, checkpoint, and reversible id mapping before running the exporter again.
