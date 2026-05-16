# Pony Official Baseline Reuse

This document is now the source of truth for TRUCE-Rec paper-facing external
baselines.

## Policy

TRUCE-Rec reuses the official-qwen3base same-candidate baseline evidence from
the sibling Pony/Uncertainty project instead of rerunning the same official
baseline suite inside TRUCE. The projects share the same author, the same
four-domain same-candidate data lane, and the same candidate score contract.

The reused score schema is:

```text
source_event_id,user_id,item_id,score
```

Main baseline tables may include only rows satisfying all gates:

- `artifact_class=completed_result`;
- `status_label=same_schema_external_baseline`;
- `implementation_status=official_completed`;
- a copied TRUCE evidence package is present under
  `outputs/pony_official_baselines/evidence_packages/`.

Rows with completed metrics but missing local evidence packages remain
`pending_import`. Rows for unfinished domains remain `pending_running`.

## Baseline Pool

The paper-facing external baseline pool is:

- LLM2Rec official Qwen3-8B + SASRec;
- LLM-ESR official Qwen3-8B + LLMESR-SASRec;
- LLMEmb official Qwen3-8B;
- RLMRec official Qwen3-8B GraphCL;
- IRLLRec official Qwen3-8B IntentRep;
- ELMRec official Qwen3-8B graph bridge;
- ProEx official Qwen3-8B profile;
- ProMax official Qwen3-8B profile.

The old TRUCE-side TALLRec/OpenP5/DEALRec/LC-Rec/LLaRA/LLM-ESR controlled
adapter lane is retained as legacy/pilot infrastructure only. It is not the
current paper-facing baseline source unless the user explicitly reopens that
route.

## Local Artifact Layout

Generated/ignored artifacts:

```text
outputs/pony_official_baselines/
├── evidence_packages/*.tar.gz
├── manifest.json
└── tables/
    ├── pony_official_baseline_comparison.csv
    ├── pony_official_baseline_comparison.md
    ├── pony_official_baseline_comparison_status.csv
    └── pony_official_baseline_comparison_status.md
```

Tracked manifest:

```text
configs/baselines/pony_official_external_baselines.yaml
```

The tarballs live under `outputs/`, which is ignored by git. Commit only the
manifest, scripts, docs, and tests.

## Commands

Import/copy Pony evidence packages and write manifests:

```powershell
py -3 scripts\import_pony_official_baselines.py `
  --pony-root D:\Research\Uncertainty `
  --output-root outputs\pony_official_baselines `
  --manifest configs\baselines\pony_official_external_baselines.yaml
```

Build TRUCE-side comparison/status tables:

```powershell
py -3 scripts\build_pony_baseline_comparison.py `
  --manifest-json outputs\pony_official_baselines\manifest.json `
  --output-root outputs\pony_official_baselines\tables `
  --output-name pony_official_baseline_comparison
```

Current local import status after the first TRUCE import:

- 28 evidence packages copied and eligible.
- 28 rows enter the TRUCE Pony main-baseline comparison table.
- `llm2rec_official_qwen3base_sasrec` on Beauty is `pending_import` because the
  metrics row exists but no matching local evidence tarball was found.
- ProMax on books/electronics/movies is `pending_running`; ProMax Beauty is
  copied and eligible.

## Evidence Boundary

Reusing Pony evidence does not make TRUCE Ours complete. TRUCE still needs its
own Ours/TRUCE adapter, ablations, observation analysis, statistical tests,
efficiency artifacts, and failure cases under the same candidate protocol.
