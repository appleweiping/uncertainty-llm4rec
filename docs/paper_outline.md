# Paper Outline

## 1. Working title

Calibrated Uncertainty-Guided Generative Recommendation.

## 2. Abstract placeholder

TBD: fill from real metrics files. The abstract should summarize the problem,
implemented framework, datasets, baselines, main empirical findings, and
limitations only after real experiments are complete.

## 3. Introduction outline

- Motivation: LLMs can generate plausible item titles, but plausibility is not
  reliability.
- Gap: recommendation-specific uncertainty includes grounding, hallucination,
  popularity bias, long-tail behavior, and history similarity.
- Contribution candidates: observation framework, shared evaluator, uncertainty
  policy, ablation protocol, and reproducible artifact plan.
- Claims are TBD until real metrics exist.

## 4. Problem formulation

Define next-item generative recommendation:

```text
history -> generated title -> catalog grounding -> ranked prediction
```

Correctness, validity, grounding, candidate adherence, confidence, popularity,
and history-similarity signals are measured under a shared evaluator.

## 5. Method overview

Describe the implemented pipeline:

- generate title;
- ground to catalog;
- estimate uncertainty signals;
- route through accept/fallback/abstain/rerank;
- save unified prediction records.

## 6. Uncertainty-aware generative recommendation observation

TBD: fill from real metrics files. This section should analyze confidence,
grounding, hallucination, popularity, long-tail, and echo-risk observations.

## 7. Calibrated uncertainty-guided method

Describe OursMethod mechanics and configuration. Do not claim effectiveness
without real results. TBD: fill from real metrics files.

## 8. Experimental setup

Datasets, splits, candidates, baselines, providers, seeds, metrics,
significance tests, and artifact paths. TBD: fill dataset-specific details from
completed real runs.

## 9. Main results table placeholder

TBD: fill from real metrics files. No fake numbers.

## 10. Calibration/uncertainty analysis placeholder

TBD: fill from real metrics files. Include ECE, Brier score, reliability
diagram, risk-coverage, high-confidence wrong rate, and low-confidence correct
rate.

## 11. Popularity/long-tail analysis placeholder

TBD: fill from real metrics files. Include confidence and accuracy by
popularity bucket, long-tail coverage, and under-confidence analysis.

## 12. Echo-chamber risk analysis placeholder

TBD: fill from real metrics files. Include history similarity, category
repetition, diversity, novelty, and confidence-weighted variants.

## 13. Ablation study placeholder

TBD: fill from real metrics files. Compare full method, fallback-only, and each
disabled component under the same evaluator.

## 14. Efficiency/cost analysis placeholder

TBD: fill from real metrics files. Include latency, token counts, estimated API
cost, throughput, and GPU memory where available.

## 15. Case study placeholder

TBD: fill from real metrics files. Use the case-study template and include raw
output snippets only within copyright and privacy limits.

## 16. Limitations

Use `docs/limitations.md` as the source. Update after real failures are
observed. TBD: fill from real metrics files where quantitative limitations are
needed.

## 17. Ethics/reproducibility notes

Discuss implicit-feedback incompleteness, popularity bias, dataset licensing,
API reproducibility, prompt leakage safeguards, artifact release, and cost
tracking. TBD: fill from real run manifests.
