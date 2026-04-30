# CR-20260430 Selective Risk Analysis

## Summary

Implement AURC/selective-risk diagnostics for grounded observation analysis.

## Motivation

`docs/experiment_protocol.md` and `docs/implementation_plan.md` already list
AURC/selective risk as a planned confidence metric. The current analysis layer
reports ECE, Brier, CBU/WBC, Tail Underconfidence Gap, reliability bins, and
popularity-confidence diagnostics, but it does not yet measure how risk changes
when examples are retained from high to low confidence.

Selective risk is directly relevant to Storyflow because confidence can shape
abstention, triage, and exposure decisions. It should therefore be visible in
the observation layer before learned CURE/TRUCE calibration is claimed.

## Storyflow Alignment

Go. The metric consumes grounded predictions only and keeps correctness,
confidence, grounding, popularity/head-tail, and claim guardrails together. It
does not change the title-level generative recommendation task.

## Phase Fit

Current phase: Phase 2C / Phase 3 analysis completeness.

## Risk Checks

- Toy risk: low. The metric is generic and applies to real grounded runs.
- Stitching risk: low. It is part of the existing confidence-analysis family.
- API required: no.
- Server required: no.
- Full data required: no.
- Training required: no.
- Paper-result risk: medium if overclaimed; docs must state it is a diagnostic
  unless backed by approved real runs and manifests.

## Decision

Go.

## Minimal Scope

- Add selective-risk curve, AURC, optimal AURC, and excess AURC utilities.
- Add these metrics to observation analysis summary/report/output artifacts.
- Add tests for metric math and analysis output.
- Update README and observation-analysis docs.

## Acceptance Criteria

- Tests pass without API calls, server execution, data download, full data
  preparation, or model training.
- Analysis writes `selective_risk_curve.json`.
- Reports label selective risk as diagnostic rather than exposure-utility
  evidence by itself.
