# Result Artifact Checklist

Every real experiment should preserve the following artifacts.

## Required run artifacts

- `resolved_config.yaml`
- `environment.json`
- `git_info.json` if available
- `logs.txt`
- `predictions.jsonl`
- raw LLM outputs if any
- `metrics.json`
- `metrics.csv`
- `cost_latency.json`
- `artifacts/`
- checkpoints or adapters when training is performed

## Required aggregate artifacts

- aggregate tables
- reliability diagram data
- risk coverage data
- confidence by popularity bucket data
- ablation table
- failure cases
- case-study packet where applicable

## Required provenance

- run command
- seed list
- commit hash
- dataset config
- candidate protocol
- split protocol
- provider/model config for LLM runs
- budget/rate-limit confirmation for API runs
- server/GPU details for local HF or training runs

## Required claim labels

Each run should state whether it is:

- smoke/mock;
- dry-run;
- pilot;
- full real experiment;
- API run;
- local HF run;
- training run.

No paper claim should be made from a run missing required artifacts.

## Main table export plan

Main paper tables should be exported only from completed real `metrics.json`
files. The table should include dataset, method, candidate protocol, seeds,
Recall@K, NDCG@K, MRR@K, validity, hallucination, coverage, diversity, novelty,
long-tail, latency, token usage, and artifact path. TBD: fill from real metrics
files.

## Ablation table export plan

Ablation tables should compare Ours full, fallback-only, and each disabled
component under the same dataset, split, seed list, candidate protocol, and
evaluator. Include disabled component, decision distribution, validity,
calibration, ranking metrics, and cost/latency. TBD: fill from real metrics
files.

## Figure artifact plan

Figures should be generated from saved CSV/JSON artifacts, not copied manually
from notebooks. Required plot-data artifacts include reliability diagram data,
risk-coverage data, confidence by popularity bucket, long-tail confidence
summary, diversity/novelty summary, and selected failure cases. TBD: fill from
real metrics files.
