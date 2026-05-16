# TRUCE-Rec

This repository is TRUCE-Rec unless the user states otherwise. It is a
research-grade LLM4Rec codebase for uncertainty-aware generative
recommendation, with a shared data, baseline, OursMethod, evaluation, export,
and reproducibility scaffold.

Current repository identity:

- GitHub: `https://github.com/appleweiping/TRUCE-Rec.git`
- Historical remote alias in this checkout: `https://github.com/appleweiping/uncertainty-llm4rec.git`
- Local path: `D:\Research\TRUCE-Rec`
- Active branch: `main`
- Current stage: Gate R1 server-first four-domain buildout with reused
  Pony/Uncertainty official-qwen3base baseline evidence. No paper-result claim
  is allowed until TRUCE Ours runs, ablations, and remaining audits are
  imported and evaluated under the same protocol.

## Evidence Labels

Use these labels consistently:

- Smoke/mock: fixture-data or MockLLM runs that verify code paths only.
- Pilot: small approved real-data run used to debug the protocol.
- Diagnostic: prompt, grounding, candidate, or artifact QA; not a paper
  conclusion.
- Controlled adapter pilot: TRUCE-side implementation that uses the shared
  protocol and Qwen3-8B base model but has not yet passed official baseline
  fidelity audit.
- Official-native controlled baseline: official project algorithm with shared
  TRUCE protocol, Qwen3-8B base-model substitution, LoRA adaptation, and the
  baseline's official/default or reported-optimal hyperparameters where
  feasible.
- Paper result: approved real experiment with tracked code, saved config, logs,
  raw outputs where applicable, predictions, metrics, and artifact checklist.

Strict rules:

- Smoke outputs are not paper evidence.
- MockLLM outputs are not paper evidence.
- Pilot/API diagnostic outputs are not paper conclusions.
- Controlled adapter pilots are not final official baseline results.
- Ignored local diagnostics under `outputs/` or `data/processed/` are not
  paper evidence unless explicitly promoted by a later approved protocol.
- Formal paper results must come from approved real experiment configs, tracked
  code, saved configs, logs, raw outputs, predictions, and metrics.
- Reference papers and official projects may be read carefully for
  reproduction fidelity and inspiration, but the TRUCE/CURE method must not be
  a stitched, copied, or renamed version of those systems.

## Current Status

Implemented and smoke-tested:

- dataset preprocessing and tiny fixture data;
- random, popularity, BM25, MF, sequential Markov, and LLM mock baselines;
- MockLLM, OpenAI-compatible provider interface, HF provider scaffold, and
  LoRA dry-run scaffold;
- unified prediction schema and shared evaluator;
- ranking, validity, confidence, calibration, coverage, diversity, novelty,
  long-tail, efficiency, slicing, aggregation, and table export support;
- Phase 6 OursMethod:
  `Calibrated Uncertainty-Guided Generative Recommendation`;
- Phase 7 paper support, real experiment templates, reproduction docs, and
  safe preflight helpers.
- Pony/Uncertainty official-qwen3base same-candidate baseline evidence is now
  the paper-facing external baseline source. TRUCE imports/copies the evidence
  packages and tracks eligibility in
  `configs/baselines/pony_official_external_baselines.yaml`.
- The old TRUCE-side controlled-adapter server suite for TALLRec/OpenP5-style/
  DEALRec/LC-Rec remains legacy pilot infrastructure, not the current main
  baseline route.
- The current project route is now organized as:
  `observation -> CURE/TRUCE framework -> official baselines -> four-domain
  same-candidate recommendation system`.
- Ours/TRUCE Qwen adapter preparation and import/evaluation scaffolds exist for
  server-side training: `scripts/prepare_ours_qwen_adapter_training.py` and
  `scripts/import_evaluate_ours_adapter.py`.
- Ours adapter supervision has been upgraded to
  `truce_observation_residual_policy_sft_v2`: train/valid rows include
  candidate-normalized utility, popularity-residual utility, harm/abstain risk,
  and conservative promote/suppress/fallback policy targets; test scoring keeps
  the same `candidate_scores.csv` schema by scoring promote-action likelihood.

Not yet completed:

- no approved paper-result experiment suite;
- no claim that OursMethod is effective;
- no HF model download;
- no completed Gate R1 TRUCE Ours/ablation paper-scale run;
- no final TRUCE Ours/ablation table under the reused Pony candidate protocol;
- no completed TRUCE-side observation sweep over Ours plus reused strong
  baselines;
- no final paper conclusions.
- no completed four-domain observation sweep that checks base Qwen3-8B and the
  four senior-recommended Qwen3-8B-LoRA baselines side by side;
- no final learned TRUCE policy/adapter ablation suite proving that Ours is
  deeper than heuristic prompting or conservative rules.

## Method Lineage

The active research line is:

```text
RESEARCH_IDEA
  -> title-generation observation and grounding
  -> CU-GR / CURE-TRUCE uncertainty features
  -> CU-GR v2 candidate-normalized preference fusion
  -> full TRUCE recommendation system with official baselines
```

The active package is `src/llm4rec/`. Historical `src/storyflow/` references
are legacy scaffolding and should be treated as background unless a current
document explicitly maps them to `llm4rec` modules.

## Durable Memory

Future agents must read `docs/PROJECT_MEMORY.md` before nontrivial work. It
records the current big direction, senior baseline advice, server workflow,
multi-agent expectation, evidence labels, four-domain plan, and update
discipline. If a task changes the roadmap, baseline policy, server commands,
or current status, update `docs/PROJECT_MEMORY.md` in the same commit.

Complex tasks should use multi-agent collaboration by default when available:
implementation/exploration plus a reviewer/fairness pass. Each completed task
should end with the next concrete plan and a stage verdict: still open, blocked
by specific gates, or ready to move toward paper writing after top-conference
review.

## Official Baseline Contract

The paper-facing external baseline lane now reuses Pony/Uncertainty completed
official-qwen3base same-candidate evidence:

```text
Pony official or official-code-level baseline run
  + shared four-domain same-candidate task
  + shared Qwen3-8B text/LLM backbone policy where applicable
  + source_event_id,user_id,item_id,score
  + copied TRUCE evidence package and tracked manifest
```

Rows enter TRUCE main baseline tables only when
`artifact_class=completed_result`, `status_label=same_schema_external_baseline`,
`implementation_status=official_completed`, and a local copied evidence package
is present. Ours may tune hyperparameters only through the declared validation
protocol. The score schema is:

```text
source_event_id,user_id,item_id,score
```

The current reused official baseline families are:

| Role | Family | Current TRUCE status |
| --- | --- | --- |
| Main | LLM2Rec | Pony completed rows reused where evidence package is present |
| Main | LLM-ESR | Pony completed rows reused |
| Main | LLMEmb | Pony completed rows reused |
| Main | RLMRec | Pony completed rows reused |
| Main | IRLLRec | Pony completed rows reused |
| Main | ELMRec | Pony completed rows reused |
| Main | ProEx | Pony completed rows reused |
| Main | ProMax | Beauty reused; remaining domains pending |

See `docs/pony_official_baseline_reuse.md`.

## Key Commands

Run smoke baselines:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_all_baselines.yaml
```

Run Phase 6 smoke suite:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase6_all.yaml
```

Run all tests:

```powershell
.\.venv\bin\python.exe -m pytest
```

Import Pony/Uncertainty official baseline evidence:

```powershell
py -3 scripts\import_pony_official_baselines.py `
  --pony-root D:\Research\Uncertainty `
  --output-root outputs\pony_official_baselines `
  --manifest configs\baselines\pony_official_external_baselines.yaml
```

Build the Pony baseline comparison/status tables:

```powershell
py -3 scripts\build_pony_baseline_comparison.py `
  --manifest-json outputs\pony_official_baselines\manifest.json `
  --output-root outputs\pony_official_baselines\tables `
  --output-name pony_official_baseline_comparison
```

Validate a real experiment template without running it:

```powershell
.\.venv\bin\python.exe scripts\validate_experiment_ready.py --config configs/experiments/real_ours_method_template.yaml
```

List required artifacts for a planned run:

```powershell
.\.venv\bin\python.exe scripts\list_required_artifacts.py --config configs/experiments/real_ours_method_template.yaml
```

## Important Docs

- `AGENTS.md`: engineering and research governance rules.
- `docs/PROJECT_MEMORY.md`: durable future-agent memory and current project
  direction.
- `docs/RESEARCH_IDEA.md`: core research direction.
- `docs/experiment_protocol.md`: split, candidate, prompt, LLM, metric, and
  leakage protocol.
- `docs/real_experiment_matrix.md`: planned real experiment groups.
- `docs/reproduction.md`: local reproduction commands.
- `docs/pre_experiment_checklist.md`: real-run readiness checklist.
- `docs/result_artifact_checklist.md`: artifact contract.
- `docs/baselines.md`: baseline readiness and limitations.
- `docs/pony_official_baseline_reuse.md`: current paper-facing reused Pony
  official baseline policy and commands.
- `docs/ours_method_plan.md`: Phase 6 method plan and boundaries.
- `docs/ablation_protocol.md`: OursMethod ablation protocol.
- `docs/leakage_fairness_checklist.md`: leakage/fairness safeguards.
- `docs/server_runbook.md`: API/HF/server/LoRA safety runbook.
- `docs/submission_roadmap.md`: milestone roadmap from observation to
  four-domain submission system.
- `docs/server_execution_matrix.md`: server-first command and artifact matrix.
- `docs/top_conference_review_plan.md`: internal reviewer/literature-agent
  checklist before paper writing.
- `docs/server_execution_matrix.md`: includes the base/baseline observation
  gate and formal Ours/baseline server command ladder.
- `docs/qwen3_lora_controlled_baselines.md`: legacy controlled-adapter pilot
  protocol.
- `docs/controlled_baseline_fidelity_audit.md`: legacy fidelity checklist.
- `docs/server_next_commands.md`: current server continuation commands.
- `docs/external_project_baseline_packets.md`: external project packet matrix.
- `docs/week8_large_same_candidate_protocol.md`: larger same-candidate
  books/electronics/movies protocol.

Current four-domain same-candidate artifact slugs are
`beauty_supplementary_smallerN_100neg`, `books_large10000_100neg`,
`electronics_large10000_100neg`, and `movies_large10000_100neg`. This artifact
lane is not model weights or a paper result by itself; do not edit its
`candidate_items.csv` or `ranking_valid/test.jsonl`, and export cross-project
scores as `source_event_id,user_id,item_id,score`.

## Package Layout

The active implementation package is `src/llm4rec/`. Older `storyflow`
references in historical local notes are not the active Phase 6/7 package
contract.

## Real Experiment Rule

Before any real pilot, fill a `configs/experiments/real_*_template.yaml`,
validate it with `scripts/validate_experiment_ready.py`, confirm datasets and
resources, and keep the safety flags blocking API calls, downloads, and
training until the user explicitly approves the run.
