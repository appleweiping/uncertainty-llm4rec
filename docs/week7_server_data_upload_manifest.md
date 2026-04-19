# Week7 Server Data Upload Manifest

This manifest records the local processed data that should be uploaded before continuing Week7 server runs. It intentionally does not assume that the current server-side `data/` directory is valid, because that directory was identified as an old partial upload rather than the active pony data state.

## Scope

The current Week7 execution line remains unchanged: Qwen3-8B is the local server backend, structured risk is still the current best ranking family, pairwise remains the mechanism line, and local swap / fully fused remain retained exploratory families. This manifest only fixes the data handoff layer.

Do not rerun preprocessing for this step. Upload the local processed files that already exist under `D:/Research/Uncertainty-LLM4Rec/data/processed`.

## Old Partial Data To Ignore

The following local files are old root-level toy or noisy pointwise artifacts and should not be treated as pony mainline data:

- `data/processed/test.jsonl`
- `data/processed/test_noisy.jsonl`
- `data/processed/test_noisy_metadata.json`

The following directories are also not the immediate Week7 server input despite being present locally:

- `data/processed/movielens_1m`: old Movielens branch, not the active Amazon pony mainline.
- `data/processed/amazon_books`, `data/processed/amazon_electronics`, `data/processed/amazon_movies`: large intermediate CSV-oriented directories that currently do not contain the full multitask `ranking_*` and `pairwise_*` jsonl set required by the current Week5-Week7 pony pipeline.

## Minimal Upload For Immediate Week7 Qwen3 Smoke And Medium

Upload the Beauty processed directory below. It is the minimum practical set for the current Qwen3 smoke configs, literature baseline entry, medium-scale pointwise / ranking / pairwise runs, and later structured-risk rerank dependency chain.

- `data/processed/amazon_beauty/train.jsonl`
- `data/processed/amazon_beauty/valid.jsonl`
- `data/processed/amazon_beauty/test.jsonl`
- `data/processed/amazon_beauty/ranking_valid.jsonl`
- `data/processed/amazon_beauty/ranking_test.jsonl`
- `data/processed/amazon_beauty/pairwise_valid.jsonl`
- `data/processed/amazon_beauty/pairwise_test.jsonl`
- `data/processed/amazon_beauty/pairwise_coverage_valid.jsonl`
- `data/processed/amazon_beauty/pairwise_coverage_test.jsonl`
- `data/processed/amazon_beauty/items.csv`
- `data/processed/amazon_beauty/users.csv`
- `data/processed/amazon_beauty/interactions.csv`
- `data/processed/amazon_beauty/popularity_stats.csv`

Observed local size: about 68 MB. Key line counts: `test.jsonl` 5,838; `ranking_test.jsonl` 973; `pairwise_coverage_test.jsonl` 500; `pairwise_test.jsonl` 4,865.

## Recommended Complete Compact Pony Upload

For a clean server state that can support Week7 handoff and Week8 cross-domain extension without another data interruption, upload these processed directories:

- `data/processed/amazon_beauty`
- `data/processed/amazon_movies_small`
- `data/processed/amazon_books_small`
- `data/processed/amazon_electronics_small`

Observed local total size: about 179.5 MB.

Each directory should preserve the relative path under the repository root. The `_small` directories are the current compact cross-domain processed results used by the Week6/Week7 pony plan for Movies, Books, and Electronics. They contain the required `train.jsonl`, `valid.jsonl`, `test.jsonl`, `ranking_valid.jsonl`, `ranking_test.jsonl`, `pairwise_valid.jsonl`, `pairwise_test.jsonl`, `items.csv`, `users.csv`, `interactions.csv`, and `popularity_stats.csv` files.

## Optional Later Upload

Only upload noisy Beauty directories if the next run explicitly targets robustness/noisy validation:

- `data/processed/amazon_beauty_noisy`
- `data/processed/amazon_beauty_noisy_nl10`
- `data/processed/amazon_beauty_noisy_nl20`
- `data/processed/amazon_beauty_noisy_nl30`

These are not required for the immediate Qwen3 smoke, parser/thinking compatibility check, or Week7 medium-scale Beauty batch.

## Upload Command Shape

Use an upload tool that preserves relative paths. Example command shape from the local repository root:

```bash
rsync -av data/processed/amazon_beauty/ <server_repo>/data/processed/amazon_beauty/
rsync -av data/processed/amazon_movies_small/ <server_repo>/data/processed/amazon_movies_small/
rsync -av data/processed/amazon_books_small/ <server_repo>/data/processed/amazon_books_small/
rsync -av data/processed/amazon_electronics_small/ <server_repo>/data/processed/amazon_electronics_small/
```

Do not place server credentials, SSH passwords, temporary URLs, or tokens in repository files.
