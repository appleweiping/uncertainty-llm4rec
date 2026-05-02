# Gate R2 MovieLens 1M Protocol

Evidence label:

```text
full single-dataset experiment / candidate paper evidence, pending reviewer approval
```

Mock notice:

```text
MockLLM results are pipeline evidence only, not real LLM evidence.
```

## Dataset

- Dataset: MovieLens 1M.
- Raw path: `data/raw/movielens_1m/ml-1m`.
- R2 canonical source path: `data/processed/movielens_1m/r2_full_single_dataset_source`.
- R2 processed path: `data/processed/movielens_1m/r2_full_single_dataset`.
- Config: `configs/datasets/movielens_1m_r2.yaml`.
- Raw source files: `ratings.dat` and `movies.dat`.
- No download is allowed or required.
- R1 `sanity_50_users` is not used for R2.

Discovered source counts from local raw data:

- Users: 6040.
- Raw movie records: 3883.
- Items with interactions: 3706.
- Interactions: 1000209.
- Minimum user interactions: 20.
- Maximum user interactions: 2314.
- Average user interactions: 165.5975.

## Split

- Split strategy: chronological leave-one-out with the last item as test and the second-last item as validation.
- Timestamp handling: sort by timestamp then item id within each user.
- Future interactions: forbidden.
- Train example policy: one final training supervision example per user, configured as `train_examples_per_user: 1`.
- Expected processed examples: 6040 train, 6040 validation, 6040 test.
- Train popularity source: train split only. The train example history plus train target represent all interactions before each user's validation/test holdouts.

## Candidate Protocol

- Protocol: sampled ranking.
- Candidate size: 100.
- Target inclusion policy: held-out target must be included for every evaluated example.
- Negative sampling seed: 13.
- Sampling pool: catalog items excluding the target and excluding the example history.
- Full-ranking vs sampled-ranking: sampled-ranking only for this R2 run.
- Candidate set saved path: `data/processed/movielens_1m/r2_full_single_dataset/candidate_sets.jsonl`.
- Same candidate set policy: all compared methods share the same preprocessed candidate sets.

Candidate size 100 is a scalability compromise for Gate R2. It is acceptable for protocol validation, but any final paper table should either keep the sampled-ranking label explicit or add a later candidate-sensitivity/full-ranking subgate.

## Metadata Sources

- Item title source: MovieLens `movies.dat` title field.
- Item category/genre source: MovieLens `movies.dat` genre field, copied to `category` and `genres`.
- Domain: `movies`.
- User and item IDs: original MovieLens IDs preserved.

## Methods

R2 method matrix:

- `popularity`
- `bm25`
- `mf`
- `sequential_markov`
- `llm_generative_mock`
- `ours_uncertainty_guided`
- `ours_fallback_only`
- `ours_ablation_no_uncertainty`
- `ours_ablation_no_grounding`
- `ours_ablation_no_popularity_adjustment`
- `ours_ablation_no_echo_guard`

Seed list: `[13, 21, 42]`.

MF configuration: `configs/methods/mf.yaml` with 4 factors and 5 local epochs. This is lightweight local baseline fitting, not LoRA/QLoRA training.

The negative-sampling seed is fixed at 13 because `run_all.py` preprocesses one shared candidate file before child method runs. Method randomness uses the configured method seed list.

## LLM Policy

- Real API calls: disabled.
- Provider for R2 base run: `configs/llm/mock.yaml`.
- MockLLM status: pipeline diagnostic only, not real LLM evidence.
- HF downloads: disabled.
- LoRA/QLoRA training: disabled.
- Follow-up real LLM config: `configs/experiments/r2_movielens_1m_real_llm_api_followup.yaml`.
- Follow-up config default: `dry_run: true`, `requires_confirm: true`, `allow_api_calls: false`.

## Metrics

Required metrics:

- Ranking: Recall@K, HitRate@K, MRR@K, NDCG@K.
- Validity: validity rate, hallucination rate, candidate adherence.
- LLM/generative: parse success rate and grounding success rate when applicable.
- Confidence/calibration: mean confidence, ECE, Brier, risk-coverage when confidence exists.
- Beyond accuracy: coverage, diversity, novelty, long-tail metrics.
- Slices: user-history and item-popularity slices when produced by the evaluator.
- Efficiency: latency, token usage, cost proxy fields when available.

## Artifact Contract

Each run under `outputs/runs/<run_id>/` must include:

- `resolved_config.yaml`
- `environment.json`
- `logs.txt`
- `predictions.jsonl`
- `metrics.json`
- `metrics.csv`
- `cost_latency.json`
- `artifacts/`

`git_info.json` is optional if the git commit is captured in `environment.json`.

Table outputs under `outputs/tables/` must include:

- `aggregate_metrics.csv`
- `aggregate_metrics.md`
- `aggregate_metrics.tex`
- `experiment_summary.json`
- confidence artifacts where confidence data exists.

## Leakage Safeguards

- Target item ID is excluded from LLM prompt candidate IDs.
- Target title is excluded from prompt history/candidate titles.
- Future interactions are not included in histories.
- Popularity, novelty, and long-tail statistics use train examples only.
- Grounding uses the item catalog only.
- Confidence policy must not inspect target correctness.
- Fallback methods use the same candidate set.
- Ablations disable only their configured components.
- All methods use the shared prediction schema and evaluator.

## Evidence Status

This R2 run can validate full single-dataset protocol, shared candidate files, shared evaluator behavior, non-API baselines, and MockLLM pipeline plumbing. It cannot support paper claims about real LLM uncertainty or real generative recommendation behavior without the real-LLM subgate.
