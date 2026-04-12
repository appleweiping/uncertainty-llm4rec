# Beauty Freeze Checklist

This checklist defines when the Beauty main domain can be treated as complete enough for paper-first work.

## Experiment Completeness

- [x] Clean inference, evaluation, calibration, and reranking are all available.
- [x] Five-model clean comparison exists on Beauty.
- [x] Multi-estimator comparison exists on Beauty.
- [x] Fused uncertainty is parameterized and included in the compare stack.
- [x] Robustness exists as both a single noisy baseline and a multi-level curve.
- [x] Reproducibility has at least one repeated-run check on Beauty.

## Summary Layer

- [x] `final_results.csv` is stable and schema-aligned.
- [x] `model_results.csv` is stable and schema-aligned.
- [x] `beauty_estimator_results.csv` is stable and schema-aligned.
- [x] `robustness_brief.csv` exists for concise robustness discussion.
- [x] `reproducibility_delta.csv` exists for appendix-level stability evidence.
- [x] Beauty paper-facing exports are generated under `outputs/summary/`.

## Writing Layer

- [x] `docs/tables.md` maps summary files to paper usage.
- [x] `docs/paper_outline.md` maps claims to Beauty-centered evidence.
- [ ] A Beauty experiment-section draft is written in full prose.
- [ ] Final figure choices are fixed.
- [ ] Main-text versus appendix placement is frozen.

## Interpretation Layer

- [x] The main claim is no longer just “confidence exists,” but “confidence can be diagnosed, calibrated, compared, fused, and stress-tested.”
- [x] Beauty is treated as the main narrative domain.
- [x] Other domains are treated as supporting generality evidence rather than the main writing axis.

## Exit Condition

Beauty can be considered paper-ready when the remaining unchecked writing items are completed without requiring additional major code or pipeline changes.
