# Pre-Experiment Checklist

Complete this before any real experiment.

## Scope

- [ ] Dataset selected.
- [ ] Method/baselines selected.
- [ ] Candidate protocol selected.
- [ ] Seed list selected.
- [ ] Metrics selected.
- [ ] Output directory selected.

## Data

- [ ] Raw data path exists or is marked TBD.
- [ ] Data license/access requirements are satisfied.
- [ ] Preprocessing config exists.
- [ ] Split strategy is documented.
- [ ] Candidate generation protocol is documented.

## Leakage

- [ ] Target title excluded from prompts.
- [ ] Target item ID excluded from prompts.
- [ ] Future interactions excluded from histories/features.
- [ ] Train popularity only.
- [ ] Shared evaluator and prediction schema.
- [ ] Comparable candidate sets across compared methods.

## Execution safety

- [ ] `dry_run: true` or `requires_confirm: true` for templates.
- [ ] API key environment variable documented, if needed.
- [ ] API budget/rate limit confirmed, if needed.
- [ ] HF model path local or download explicitly approved.
- [ ] LoRA/QLoRA training resources confirmed, if needed.

## Artifacts

- [ ] Run command recorded.
- [ ] Commit hash recorded.
- [ ] Environment capture enabled.
- [ ] Raw LLM outputs enabled for LLM runs.
- [ ] Cost/latency tracking enabled where applicable.
- [ ] Failure-case collection planned.

## Claims

- [ ] No paper result will be written before metrics exist.
- [ ] No method-effectiveness claim will be made from smoke/mock runs.
