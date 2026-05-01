# Reproduction

## 1. Install

```powershell
python -m venv .venv
.\.venv\bin\python.exe -m pip install -e .
```

Use the repository venv if it already exists.

## 2. Run smoke tests

```powershell
.\.venv\bin\python.exe -m pytest
```

## 3. Run Phase 1-6 smoke pipelines

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke.yaml
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_all_baselines.yaml
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase3_all.yaml
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase4_all.yaml
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase5_all.yaml
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase6_all.yaml
```

## 4. Export tables

```powershell
.\.venv\bin\python.exe scripts\export_tables.py --input outputs/runs --output outputs/tables
```

## 5. Aggregate runs

```powershell
.\.venv\bin\python.exe scripts\aggregate_runs.py --input outputs/runs --output outputs/tables
```

## 6. Where outputs are saved

Run artifacts are saved under `outputs/runs/<run_id>/`. Tables are saved under
`outputs/tables/` by default.

## 7. How to inspect predictions

Open `outputs/runs/<run_id>/predictions.jsonl`. Check `method`, `candidate_items`,
`predicted_items`, `scores`, `raw_output`, and `metadata`.

## 8. How to add a new dataset

1. Create raw interaction and item files matching `docs/data_format.md`.
2. Add a dataset YAML under `configs/datasets/`.
3. Run preprocess in dry/smoke mode first.
4. Verify processed examples contain no future history leakage.
5. Run a small smoke experiment before scaling.

## 9. How to run an API LLM experiment

1. Fill a real API config from `configs/experiments/real_llm_api_template.yaml`.
2. Keep `requires_confirm: true`.
3. Set API key through the documented environment variable.
4. Run `scripts/validate_experiment_ready.py`.
5. Get explicit user confirmation before any real API command.

## 10. How to run OursMethod smoke

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_ours_method.yaml
```

## 11. How to verify no target leakage

- Inspect prediction metadata for `target_excluded_from_prompt=true`.
- Confirm prompt candidate IDs exclude the target.
- Confirm target title is absent from raw prompt artifacts if saved.
- Run OursMethod leakage tests:

```powershell
.\.venv\bin\python.exe -m pytest tests/unit/test_ours_leakage_safeguards.py
```
