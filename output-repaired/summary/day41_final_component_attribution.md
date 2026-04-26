# Day41 Final Component Attribution

## Summary

Across the observation-stage experiments, the primary contributor is `calibrated_relevance_probability`. Evidence risk is useful, but its best-supported role is a secondary regularizer rather than a standalone scorer.

## Findings

1. `calibrated_relevance_probability` is the main contributor in the external backbone plug-in setting.
2. `evidence_risk` alone is usually weaker than calibrated relevance in candidate ranking.
3. `D = calibrated relevance + evidence risk` often exceeds B, especially in the full Beauty backbone results, supporting evidence risk as a secondary regularizer.
4. Day6 yes/no decision reliability is the setting where evidence risk is strongest as a direct decision-risk signal.
5. Day9/relevance/backbone plug-in should be described as calibrated posterior first, risk regularization second.

## Boundary

Do not describe evidence risk as the main scorer for candidate-level recommendation. Do not use small-domain fallback-heavy gains as fully healthy external-backbone proof.
