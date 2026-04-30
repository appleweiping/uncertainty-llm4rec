# Expansion Approval Gates

This document records the local approval-gate helper for the next real
Storyflow / TRUCE-Rec expansion. It is not an experiment result and it does not
authorize execution by itself.

## Purpose

The project now has several real expansion paths:

- another large-model API/provider observation;
- Qwen3-8B server observation;
- Amazon Reviews 2023 full-category preparation beyond Beauty;
- a trained ranking baseline artifact entering the title-grounded baseline
  observation path.

Each path can spend money, require server hardware, process full data, or
create artifacts that reviewers may mistake for paper evidence. The helper
therefore writes an ignored manifest and report that list exactly what must be
approved before execution.

## Command

Generate all tracks:

```powershell
python scripts/build_expansion_approval_checklist.py
```

Generate only selected tracks:

```powershell
python scripts/build_expansion_approval_checklist.py --track qwen3_server --track baseline_artifact
```

Default ignored output:

```text
outputs/approval_gates/next_expansion/
```

The manifest records:

- `api_called=false`;
- `server_executed=false`;
- `model_training=false`;
- `data_downloaded=false`;
- `full_data_processed=false`;
- `is_experiment_result=false`.

## Run Packet

After choosing one expansion track, generate a non-executing run packet before
asking for execution approval:

```powershell
python scripts/build_expansion_run_packet.py --track api_provider --run-label deepseek_next_smoke --provider-config configs/providers/deepseek.yaml --provider deepseek --model deepseek-v4-flash --input-jsonl <input-jsonl> --sample-size 5 --rate-limit 10 --max-concurrency 1 --budget-label <budget-label>
```

Default ignored output:

```text
outputs/run_packets/<track>/<run_label>/
```

The packet records:

- provided and missing confirmations;
- safe local preflight commands;
- the command shape that still requires explicit approval;
- expected artifacts;
- forbidden claims;
- `api_called=false`;
- `server_executed=false`;
- `model_training=false`;
- `data_downloaded=false`;
- `full_data_processed=false`;
- `is_experiment_result=false`.

The packet can say that all required fields are present, but it still does not
authorize execution by itself. Real API calls, server commands, full data
preparation, training, and trained-baseline adaptation still require explicit
user approval in the current task.

## Tracks

### API Provider

Requires explicit confirmation of provider, model, endpoint/base URL, budget
label, sample size, rate limit, max concurrency, environment variable, and
`--execute-api`.

Default recommendation: keep DeepSeek as the only active provider until a new
provider/model/budget/rate gate is approved.

### Qwen3 Server

Requires explicit confirmation of server environment, GPU, model source,
input slice, catalog path, output artifact-return policy, and
`--execute-server`.

Default recommendation: only generate plan-only artifacts locally.

### Amazon Full Prepare

Requires explicit confirmation of dataset/category, license/access, raw review
and metadata JSONL paths, machine/disk budget, local/server execution mode, and
`--allow-full`.

Default recommendation: Beauty remains first; Video_Games or Books are the
next title-rich candidates after raw placement/server approval.

### Baseline Artifact

Requires explicit confirmation of baseline family, training provenance,
train/evaluation split declaration, ranking JSONL path, run manifest path,
leakage guards, and artifact-return policy.

Default recommendation: validate one trained SASRec-like ranking artifact
before adding more baseline families.

## Claim Policy

These helpers do not run an API, execute a server command, train a model,
download data, process full data, or adapt a baseline artifact. Their outputs
are readiness/approval artifacts only and must not be used as paper evidence.
