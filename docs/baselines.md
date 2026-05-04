# Baselines

This document describes implemented and planned baselines. It contains no
experimental conclusions.

## Implemented baselines

### Random

- Input signals: candidate item IDs and seed.
- Training data used: none.
- Forbidden signals: target labels, future interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable sanity baseline.

### Popularity

- Input signals: train-split item frequencies.
- Training data used: train examples only.
- Forbidden signals: validation/test popularity, target correctness.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable baseline; paper-ready only after real protocol runs.

### BM25

- Input signals: history item text and catalog item text.
- Training data used: catalog text; no held-out target text except when present
  in allowed history.
- Forbidden signals: target title as query, future interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable baseline.

### MF

- Input signals: train interactions/examples.
- Training data used: train split only.
- Forbidden signals: held-out interactions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: minimal MF is smoke-capable unless strengthened later.

### Sequential Markov

- Input signals: train transition counts and evaluation history.
- Training data used: train examples only.
- Forbidden signals: future sequence transitions.
- Candidate protocol: shared candidate set.
- Output schema: unified prediction schema.
- Status: smoke-capable sequential fallback/baseline.

### SASRec via RecBole adapter

- Input signals: exported canonical train interactions and evaluation histories.
- Training data used: train examples only.
- Forbidden signals: held-out targets and future interactions.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export implemented; training not completed because RecBole is not installed in the current environment. No SASRec paper metric should be reported yet.

### LightGCN via RecBole adapter

- Input signals: exported canonical train interactions.
- Training data used: train examples only.
- Forbidden signals: held-out targets, validation/test popularity leakage, and external evaluator metrics as final paper numbers.
- Candidate protocol: shared candidate set.
- Output schema: imported into unified prediction schema after external scoring.
- Status: adapter/config/export implemented; training not completed because RecBole is not installed in the current environment. No LightGCN paper metric should be reported yet.

### LLM generative / rerank / confidence observation

- Input signals: target-excluding prompt context and visible candidates.
- Training data used: none for API/mock inference.
- Forbidden signals: target title, target ID, future interactions.
- Candidate protocol: shared candidate set where applicable.
- Output schema: unified prediction schema with raw output and metadata.
- Status: MockLLM outputs are not paper evidence; real API/HF results require
  explicit config and user confirmation.

## Non-baseline components

- `skeleton` is not a baseline. It is a Phase 1 pipeline sanity check.
- OursMethod is a method under test, not a baseline.
- Fallback-only is an ablation/control for OursMethod routing.

## Limitations

Baseline strength varies by implementation maturity. Minimal smoke-capable
baselines should not be presented as paper-grade strong baselines until the real
experiment protocol, multi-seed runs, and comparable candidate sets are
complete.

RecBole-backed baselines are optional. Install with `py -3 -m pip install -e .[baselines]` in a compatible environment before running SASRec or LightGCN.
