# R3b completion report

Evidence is taken only from these files (as of this update):

- `outputs/tables/r3b_conservative_gate_main.csv`
- `outputs/tables/r3b_conservative_gate_ablation.csv`
- `outputs/tables/r3b_conservative_gate_decision_stats.csv`
- `outputs/tables/r3b_observation_failures.csv`
- `outputs/tables/r3_case_studies_high_conf_wrong.csv` (R3 export; seed 13 slice)
- `outputs/tables/r3_case_studies_accept_hurts.csv`
- `outputs/tables/r3_case_studies_fallback_saves.csv`

## Verdict

**PASS — R3b conservative gate artifact gap repaired**

`outputs/tables/r3b_conservative_gate_main.csv` now includes **`ours_conservative_uncertainty_gate`** with `metrics.json` + `predictions.jsonl` from a completed cache-only child run under `outputs/runs/r3b_movielens_1m_conservative_gate_cache_replay_*`.

**Minor scope note (not a conservative-gate blocker):** `configs/experiments/r3b_movielens_1m_conservative_gate_cache_replay.yaml` lists **`ours_ablation_no_uncertainty`** and **`llm_generative_real`** and seeds **21/42**; those rows do **not** appear in the current R3b CSVs because no matching **complete** run directories were present at export time (same rule as before: export skips runs missing `metrics.json` or `predictions.jsonl`). Do not interpolate.

## Commands run (this update)

```text
py -3 scripts/export_r3b_tables.py --runs outputs/runs --output outputs/tables
py -3 -m pytest tests/unit/test_r3b_table_export.py -q
py -3 -m pytest -q
git diff --check
```

Repair pipeline (earlier in this workstream): targeted cache-only `run_all` with `allow_api_calls: false` and `cache.require_hit: true` to complete the incomplete `ours_conservative_uncertainty_gate` shard; method YAML `ablation.variant` corrected from invalid `conservative_gate` to `full` so the runner can instantiate OursMethod without changing the conservative gate policy (thresholds / policy remain in method config).

## Test results

- `tests/unit/test_r3b_table_export.py` — pass  
- Full `pytest` — pass (300 tests at last run)

## R3b conservative gate results (from `r3b_conservative_gate_main.csv`)

Listed rows use **`seeds` = `13` only** in these CSVs (no 21/42 rows exported yet).

| method | Recall@10 | NDCG@10 | MRR@10 | validity | hallucination | parse_success | grounding_success | mean_confidence | ECE | Brier | high_conf_wrong (mean) | low_conf_correct (mean) | delta Recall@10 vs BM25 | top10 ≡ BM25 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| bm25 | 0.059106 | 0.029196 | 0.020301 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ours_fallback_only | 0.059106 | 0.029196 | 0.020301 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **1.0** |
| ours_conservative_uncertainty_gate | 0.059106 | 0.029196 | 0.020301 | 1.0 | 0.0 | 0.99851 | 0.997682 | 0.850596 | 0.850596 | 0.725007 | 6031.0 | 0.0 | **0.0** | **1.0** |
| ours_uncertainty_guided_real | 0.052318 | 0.025585 | 0.017641 | 1.0 | 0.0 | 0.99851 | 0.997682 | 0.850596 | 0.850596 | 0.725007 | 6031.0 | 0.0 | **-0.006788** | 0.91457 |

Cache / replay accounting (same file, means): `ours_uncertainty_guided_real` — `cache_hit_requests_mean` **12080.0**, `live_provider_requests_mean` **0.0**, `replay_latency_seconds_sum_mean` **5059.0783**, `effective_cost_usd_mean` **11.804446**. `ours_conservative_uncertainty_gate` — `replay_latency_seconds_sum_mean` **5271.2791**, same cache hit / cost means as Ours full on this row.

### Decision counts (`r3b_conservative_gate_decision_stats.csv`, seed 13)

| method | decision | count |
| --- | --- | ---:|
| ours_fallback_only | fallback | 6040 |
| ours_conservative_uncertainty_gate | fallback | 6040 |
| ours_uncertainty_guided_real | accept | 516 |
| ours_uncertainty_guided_real | fallback | 5409 |
| ours_uncertainty_guided_real | rerank | 115 |

### Observation / failure columns (`r3b_observation_failures.csv`, seed 13)

| method | parse_failures | high_confidence_wrong | low_confidence_correct | hallucination_rate |
| --- | ---:| ---:| ---:| ---:|
| bm25 | 0 | 0 | 0 | 0.0 |
| ours_fallback_only | 6040 | 0 | 0 | 0.0 |
| ours_conservative_uncertainty_gate | 9 | 6031 | 0 | 0.0 |
| ours_uncertainty_guided_real | 9 | 6031 | 0 | 0.0 |

The `parse_failures` **6040** for `ours_fallback_only` is what the table records (metadata convention for non-generative rows); do not read it as “LLM parse failed 6040 times” without inspecting `predictions.jsonl`.

### R3 case-study CSVs (line counts; header + 50 rows each)

- `r3_case_studies_high_conf_wrong.csv`: **51** lines  
- `r3_case_studies_accept_hurts.csv`: **51** lines  
- `r3_case_studies_fallback_saves.csv`: **51** lines  

Paths in those files point at **`r3_movielens_1m_real_llm_full_candidate500_*`** runs (R3 artifacts), not `r3b_*` run ids.

## Observation-first evidence (CSV-backed only)

1. **Ranking:** `ours_uncertainty_guided_real` Recall@10 **0.052318** is **below** BM25 / `ours_fallback_only` / **`ours_conservative_uncertainty_gate`** (**0.059106**); mean delta vs BM25 **-0.006788** for Ours full only.
2. **Calibration / confidence pathology (Ours full and conservative gate rows):** Mean confidence **0.850596**, ECE **0.850596**, Brier **0.725007**, **6031** high-confidence-wrong (main table); failures table lists **9** parse_failures for both generative methods.
3. **Overrides:** **516** `accept` and **115** `rerank` vs **5409** `fallback` for Ours full. **Conservative gate:** **6040** `fallback`, **0** accept/rerank in decision stats.
4. **Fallback-only vs conservative gate (ranking row):** Recall@10 / NDCG@10 / MRR@10 **match** BM25 and `ours_fallback_only`; **top10 ≡ BM25 rate 1.0**; **delta Recall@10 vs BM25 0.0**.

## Conservative gate vs fallback-only (R3b, seed 13)

From **`r3b_conservative_gate_main.csv` only:** aggregate Recall@10, NDCG@10, and MRR@10 for **`ours_conservative_uncertainty_gate`** equal **`ours_fallback_only`** and BM25. **Do not** claim “fallback-safe” beyond what these rows show: ranking alignment with fallback and BM25 on this protocol; decision stats show **only** fallback for all 6040 examples.

## Docs updated (this pass)

- `docs/r3b_completion_report.md` (this file)
- `docs/next_experiment_decision.md`
- `docs/paper_outline.md` (subsection 2.2 caveat tightened to completed-conservative fact)

## Recommended paper framing

**Observation-first (primary)** — supported by worse Ours-full Recall@10 than BM25, strong high-confidence-wrong mass, and non-trivial accept/rerank counts with sub-BM25 aggregate ranking; conservative gate reproduces BM25/fallback ranking on the exported row while taking no accept/rerank decisions in decision stats.

## Next recommended action

**Run multi-dataset observation study** (observation track, same metric style—not method-scaling to claim a new ranker).
