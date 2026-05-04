# CU-GR v2 Reviewer Checklist

- No target leakage: prompts use anonymous panel labels and do not reveal the held-out item as a target.
- Target included in candidate set: configured for both MovieLens 1M and Amazon Beauty.
- Candidate protocol consistent: sampled, target-included candidate sets are shared across compared methods within each dataset gate.
- Train/validation/test separation: seed13 trains/selects candidate fusion behavior, seed21 validates/tunes fusion/safety thresholds, seed42 is held out.
- No test tuning: selected fusion weights are recorded as validation-selected before seed42 reporting.
- Raw outputs saved: `preference_signals.jsonl` and raw LLM output artifacts exist for real CU-GR v2 runs.
- Cost/latency saved: per-seed cost latency CSV/JSON artifacts exist for both CU-GR v2 gates.
- Failed v1 evidence retained: MovieLens R3/R3b tables and case studies are kept and referenced as motivation, not deleted or hidden.
