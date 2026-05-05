# Reference Project Baseline Audit

This audit maps the local `references/NH` and `references/NR` paper set to
baseline projects we can reasonably compare against. It is a planning and
framework document, not a result table.

## Goal

The next stage should compare TRUCE/CU-GR against multiple external LLM4Rec
projects, not only classic recommenders. Every external project must be adapted
to the TRUCE artifact contract:

- same processed dataset split;
- same candidate protocol and target inclusion;
- external project trains/generates/scores only;
- predictions imported into `predictions.jsonl`;
- final Recall/NDCG/MRR and audit metrics computed by the TRUCE evaluator;
- no target leakage, no RecBole/project evaluator numbers copied as paper
  metrics, and no project-specific negative sampling that breaks comparability.

## Already Covered

| Family | Current coverage | Gap |
| --- | --- | --- |
| Non-personalized | popularity | adequate as weak control |
| Text retrieval | BM25/fallback | adequate as fallback/control |
| Traditional CF | local MF, RecBole LightGCN | add server-grade reruns |
| Sequential | RecBole SASRec, BERT4Rec adapter added | run BERT4Rec, optionally GRU4Rec/S3-Rec |
| LLM direct/rerank | local API listwise/direct artifacts | add official LLM4Rec projects |

## Must-Run Candidates

### OpenP5 / P5

- Source: OpenP5 paper and code: <https://github.com/agiresearch/OpenP5>.
- Why it matters: canonical open-source platform for developing, training, and
  evaluating LLM-based recommender systems; appears repeatedly in the reference
  set as a baseline family.
- TRUCE adapter target: convert OpenP5 generated item IDs/titles into candidate
  rankings; if it emits free-form text, ground through the TRUCE item catalog
  before evaluation.
- Risk: task templates and dataset preprocessing may not match our candidate
  protocol without a careful adapter.
- Priority: must run if compute and data conversion are feasible.

### TALLRec

- Source: official code: <https://github.com/SAI990323/TALLRec>.
- Why it matters: classic instruction-tuning framework aligning LLaMA-style LLMs
  with recommendation tasks; a common LLM4Rec comparator.
- TRUCE adapter target: train/tune on TRUCE train split, ask for ranked
  candidate recommendation or score candidates, then import into TRUCE schema.
- Risk: original setup expects LLaMA-era checkpoints and may need server/GPU
  environment work. It may be easier to reproduce as an official-project
  baseline than to merge code into this repo.
- Priority: must run or provide a documented blocker.

### BIGRec / DEALRec

- Source: DEALRec/BIGRec code: <https://github.com/Linxyhaha/DEALRec>.
- Why it matters: data-efficient LLM fine-tuning baseline. It is a strong
  comparator when we claim efficient LLM4Rec behavior rather than full-model
  recommender training.
- TRUCE adapter target: few-shot/fine-tuned generation must be grounded to item
  IDs and evaluated on the shared candidate set.
- Risk: prompt/item-index design and pruning policy can introduce incompatible
  training data selection if not audited.
- Priority: must-run candidate after OpenP5/TALLRec.

## High-Value Candidate Projects

### LC-Rec

- Source: <https://github.com/RUCAIBox/LC-Rec/>.
- Why it matters: integrates collaborative semantics into LLM recommendation.
  This is close to the target comparison class for CU-GR because it combines
  collaborative signals and language models.
- TRUCE adapter target: export collaborative embeddings or final rankings, then
  import per-candidate scores.
- Risk: dataset-specific preprocessing and model checkpoints may be heavy.
- Priority: high.

### LLaRA

- Source: paper/code link identifies <https://github.com/ljy0ustc/LLaRA>.
- Why it matters: combines conventional recommenders with LLM item knowledge;
  it is a direct conceptual competitor for methods that use LLM signals on top
  of recommender evidence.
- TRUCE adapter target: treat LLaRA output as an external ranker over TRUCE
  candidates.
- Risk: likely needs GPU and careful item-text conversion.
- Priority: high, choose if LC-Rec is blocked or too expensive.

### CoLLM

- Source: CoLLM paper notes code/data at <https://github.com/zyang1580/CoLLM>.
- Why it matters: injects collaborative embeddings into the LLM token embedding
  space; a strong "LLM + CF" comparator.
- TRUCE adapter target: use its ranked outputs or candidate scoring layer only,
  then TRUCE import/evaluation.
- Risk: larger implementation surface and embedding alignment requirements.
- Priority: high but likely after one easier LLM4Rec project is running.

## Specialized / Robustness Candidates

### SLMRec

- Source: <https://github.com/WujiangXu/SLMRec>.
- Why it matters: ICLR 2025 distillation from LLMs into small sequential
  recommenders. It is useful if the paper claims small-model efficiency or
  sequential recommendation strength.
- TRUCE adapter target: run as an external sequential recommender and import
  ranked candidates.
- Risk: may target different data and sequence preprocessing.
- Priority: medium-high.

### LLM-ESR

- Source: <https://github.com/liuqidong07/LLM-ESR>.
- Why it matters: NeurIPS 2024 long-tailed sequential recommendation. Relevant
  for long-tail claims and Amazon-domain analysis.
- TRUCE adapter target: preserve TRUCE train/valid/test and candidate sets;
  import final rankings and evaluate long-tail slices with TRUCE metrics.
- Risk: notebooks and embedding generation steps may make reproduction slower.
- Priority: medium-high, especially for long-tail experiments.

### TransRec / LETTER / E4SRec Line

- Source examples in the reference set include TransRec/transition paradigm,
  LETTER, and E4SRec-style order-agnostic identifier methods.
- Why it matters: directly attacks item indexing and generation grounding,
  which is close to TRUCE's validity and hallucination framing.
- TRUCE adapter target: map generated identifiers back to canonical item IDs;
  invalid or ungrounded outputs must stay visible in hallucination/validity
  metrics.
- Risk: heavier item-tokenizer work; high chance of candidate-protocol mismatch.
- Priority: medium. Run after OpenP5/TALLRec/BIGRec unless the paper pivots to
  generative-item-indexing as the central comparison.

## Near-Term Traditional Baseline Patch

BERT4Rec and GRU4Rec have been added as RecBole-backed external baseline adapters:

- `configs/baselines/bert4rec_recbole.yaml`
- `configs/baselines/gru4rec_recbole.yaml`
- `configs/experiments/baseline_bert4rec_movielens.yaml`
- `configs/experiments/baseline_bert4rec_amazon_beauty.yaml`
- `configs/experiments/baseline_bert4rec_video_games.yaml`
- `configs/experiments/baseline_gru4rec_movielens.yaml`
- `configs/experiments/baseline_gru4rec_amazon_beauty.yaml`
- `configs/experiments/baseline_gru4rec_video_games.yaml`
- `src/llm4rec/external_baselines/bert4rec_adapter.py`
- `src/llm4rec/external_baselines/gru4rec_adapter.py`

It reuses the chronological sequential benchmark export with
`item_id_list:token_seq`. MovieLens completed end-to-end on CPU with TRUCE
Recall@10 0.199172, NDCG@10 0.107392, and MRR@10 0.079387 for BERT4Rec; GRU4Rec
reached Recall@10 0.160099, NDCG@10 0.088435, and MRR@10 0.066755. Amazon
Beauty also completed end-to-end on CPU: BERT4Rec, GRU4Rec, and LightGCN all
have TRUCE Recall@10, NDCG@10, and MRR@10 of 0.000000 under the current
adapter/config.

Recommended commands:

```powershell
py -3 scripts/export_recbole_data.py --config configs/experiments/baseline_bert4rec_movielens.yaml
py -3 scripts/run_external_baseline.py --config configs/experiments/baseline_bert4rec_movielens.yaml
py -3 scripts/import_external_predictions.py --config configs/experiments/baseline_bert4rec_movielens.yaml

py -3 scripts/export_recbole_data.py --config configs/experiments/baseline_bert4rec_amazon_beauty.yaml
py -3 scripts/run_external_baseline.py --config configs/experiments/baseline_bert4rec_amazon_beauty.yaml
py -3 scripts/import_external_predictions.py --config configs/experiments/baseline_bert4rec_amazon_beauty.yaml
```

Amazon Video Games remains optional/appendix only and must not be tuned to pass
the CU-GR v2 gate.

## Framework Design For Beating External Projects

The comparison framework should be stronger than a table of copied project
metrics:

1. Build one project-specific runner per external baseline that emits either
   per-candidate scores or ranked item IDs.
2. Normalize all outputs into TRUCE prediction schema.
3. Use the same candidate set and target inclusion as CU-GR.
4. Use TRUCE evaluator for accuracy, validity, hallucination, coverage,
   diversity, novelty, long-tail, and runtime/cost.
5. Add a `project_baseline_manifest.json` recording upstream commit, install
   command, model/checkpoint, train/valid/test files, prompt/template if any,
   GPU/CPU, seed, and leakage flags.
6. Compare not only standalone CU-GR v2 against standalone project baselines,
   but also CU-GR fusion using strong recommender scores as base signals. This
   tests whether TRUCE is complementary to strong recommenders rather than only
   a replacement for weak fallback.

## Server Diagnostic Result

Amazon Beauty has a completed TALLRec-style diagnostic run using the server
packet and Qwen3-8B zero-shot likelihood scoring. The output was imported into
the TRUCE schema and evaluated by the TRUCE evaluator on the test split:
Recall@10 0.031111, NDCG@10 0.011949, and MRR@10 0.006321. This is not an
official trained TALLRec result, because LoRA training was not run. Keep it in
appendix/diagnostic tables only.

## Recommended Sequence

1. Run OpenP5 first. OpenP5 has a project-specific sequential packet and should
   be the next official external-project attempt on the server.
2. TALLRec packet support is ready. The existing Beauty Qwen3 scorer is only a
   diagnostic row until official instruction tuning is run.
3. BIGRec/DEALRec, LC-Rec, LLaRA, CoLLM, LLM-ESR, and SLMRec now have generic
   candidate-ranking packet configs for MovieLens and Amazon Beauty. These are
   execution contracts, not results.
4. Use `docs/external_project_baseline_packets.md` as the live status matrix and
   blocker log for external-project runs.
