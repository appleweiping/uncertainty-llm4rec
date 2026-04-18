# 2026-04-18 Part 2: Candidate Ranking Minimal Loop

## Stage Goal

Part 2 only delivers the **minimal candidate-ranking closed loop** for Version 3.

The accepted target is:

- keep legacy `pointwise_yesno` inference alive
- add config-driven `candidate_ranking`
- connect `prompt_builder -> main_infer -> parser -> structured prediction output`

This stage does **not** try to finish ranking evaluation, ranking-side calibration, ranking-side reranking, or baseline-side confidence validation.

## Files Actually Modified

Modified:

- `main_infer.py`
- `src/llm/prompt_builder.py`
- `src/llm/parser.py`
- `README.md`

Added:

- `prompts/candidate_rank_topk.txt`
- `prompts/candidate_score_list.txt`
- `docs/version3/2026-04-18_part2_candidate_ranking_minimal_loop.md`

## How Task Switching Works

`main_infer.py` now supports two task types:

- `pointwise_yesno`
- `candidate_ranking`

The switch is controlled through config or CLI:

```yaml
task_type: pointwise_yesno
```

or

```yaml
task_type: candidate_ranking
```

Legacy behavior is preserved:

- if `task_type` is omitted, `main_infer.py` still defaults to `pointwise_yesno`

For ranking mode, `main_infer.py` also supports:

- `ranking_mode: score_list`
- `ranking_mode: rank_topk`

If `ranking_mode` is omitted, it is inferred from the prompt path:

- `prompts/candidate_score_list.txt` -> `score_list`
- `prompts/candidate_rank_topk.txt` -> `rank_topk`

## What Was Added

### 1. Candidate-ranking control flow in `main_infer.py`

`main_infer.py` now contains a separate ranking path that:

1. reads `task_type`
2. groups existing flattened pointwise rows by `user_id`
3. constructs per-user candidate sets
4. builds a ranking prompt
5. calls the backend once per user-level candidate set
6. parses ranking output
7. writes structured ranking predictions to JSONL

This was done without changing `src/data/*`.

### 2. Ranking prompt construction

`src/llm/prompt_builder.py` now supports:

- `build_pointwise_prompt(...)`
- `build_candidate_ranking_prompt(...)`

The ranking prompt path uses:

- `history_block`
- `candidate_block`
- `candidate_count`

### 3. Ranking parser closed loop

`src/llm/parser.py` now supports:

- `parse_candidate_score_list_response(...)`
- `parse_candidate_rank_topk_response(...)`
- `parse_ranking_response(...)`

The old yes/no parser remains intact:

- `parse_response(...)`

### 4. New ranking prompt templates

Two prompt templates were added:

- `prompts/candidate_score_list.txt`
- `prompts/candidate_rank_topk.txt`

They are both structured-output prompts and are designed for stable parsing rather than free-form generation.

## Ranking Output Schema

The minimal `candidate_ranking` output row now looks like this:

```yaml
task_type: candidate_ranking
ranking_mode: score_list | rank_topk
user_id: ...
target_item_id: ...
target_popularity_group: ...
candidate_count: ...
candidates:
  - item_id: ...
    title: ...
    label: ...
selected_item_id: ...
top_k_item_ids: [...]
ranked_item_ids: [...]
candidate_scores:
  - item_id: ...
    score: ...
    reason: ...
reason: ...
prompt: ...
raw_response: ...
response_latency: ...
response_model_name: ...
response_provider: ...
response_backend_type: ...
response_usage: {...}
```

Notes:

- `score_list` mode is expected to populate `candidate_scores`
- `rank_topk` mode may leave `candidate_scores` empty while still providing `ranked_item_ids` and `top_k_item_ids`

This is the minimal structured output needed for later Part 3 ranking evaluation work.

## Minimal Ranking Config Template

This is the recommended minimal experiment template for ranking smoke runs:

```yaml
exp_name: beauty_deepseek_candidate_ranking

input_path: data/processed/amazon_beauty/test.jsonl
output_root: outputs

task_type: candidate_ranking
ranking_mode: score_list
prompt_path: prompts/candidate_score_list.txt

model_config: configs/model/deepseek.yaml

max_samples: 20
overwrite: true
seed: 42
```

Top-k variant:

```yaml
task_type: candidate_ranking
ranking_mode: rank_topk
prompt_path: prompts/candidate_rank_topk.txt
```

## Smoke Tests Already Completed

### A. Dummy-backend smoke for legacy `pointwise_yesno`

Completed.

Confirmed:

- `task_type` defaults to `pointwise_yesno`
- the legacy pointwise prompt is still used
- the yes/no parser still returns `recommend/confidence/reason`
- pointwise JSONL output is still saved successfully

### B. Dummy-backend smoke for `candidate_ranking`

Completed.

Confirmed:

- `task_type=candidate_ranking` really switches the code path
- flattened pointwise rows are grouped into per-user candidate sets
- ranking prompt builder runs
- ranking parser runs
- structured ranking JSONL output is saved successfully

### C. Parser smoke

Completed.

Confirmed:

- `score_list` parsing works on structured JSON
- `rank_topk` parsing works on structured JSON

## Recommended Reproduction Commands

### 1. Verify legacy `pointwise_yesno` still runs

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --input_path data/processed/amazon_beauty/test.jsonl `
  --output_path outputs/version3_part2_smoke/test_pointwise_raw.jsonl `
  --max_samples 5 `
  --overwrite
```

Expected output file:

- `outputs/version3_part2_smoke/test_pointwise_raw.jsonl`

### 2. Verify `candidate_ranking` in `score_list` mode

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --task_type candidate_ranking `
  --ranking_mode score_list `
  --prompt_path prompts/candidate_score_list.txt `
  --input_path data/processed/amazon_beauty/test.jsonl `
  --output_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --max_samples 5 `
  --overwrite
```

Expected output file:

- `outputs/version3_part2_smoke/test_ranking_raw.jsonl`

### 3. Verify `candidate_ranking` in `rank_topk` mode

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --task_type candidate_ranking `
  --ranking_mode rank_topk `
  --prompt_path prompts/candidate_rank_topk.txt `
  --input_path data/processed/amazon_beauty/test.jsonl `
  --output_path outputs/version3_part2_smoke/test_ranking_topk_raw.jsonl `
  --max_samples 5 `
  --overwrite
```

Expected output file:

- `outputs/version3_part2_smoke/test_ranking_topk_raw.jsonl`

## Acceptance Mapping

### 1. Does `task_type` really switch through config/CLI?

Yes.

`main_infer.py` now branches on:

- `pointwise_yesno`
- `candidate_ranking`

without requiring hand edits in code.

### 2. Do ranking prompt and parser form a closed loop?

Yes.

The closed loop is:

`candidate_ranking sample assembly -> build_candidate_ranking_prompt -> backend.generate -> parse_ranking_response -> structured JSONL output`

### 3. Does old yes/no still coexist?

Yes.

The legacy path still uses:

- `prompts/pointwise_yesno.txt`
- `build_pointwise_prompt(...)`
- `parse_response(...)`
- `run_pointwise_inference(...)`

### 4. Was eval kept minimal?

Yes.

`main_eval.py` was intentionally left out of Part 2. Ranking output was only made structured enough to serve as the basic input for later Part 3 evaluation adaptation.

## Explicit Non-Goals for Part 2

Part 2 did **not** do the following:

- no ranking reader or ranking metrics extension in `main_eval.py`
- no ranking-side calibration
- no ranking-side reranking
- no `main_uncertainty_compare.py` changes
- no `src/uncertainty/*` changes
- no baseline-side confidence validation
- no `src/data/*` rewrites

These items are intentionally deferred to later Version 3 parts.
