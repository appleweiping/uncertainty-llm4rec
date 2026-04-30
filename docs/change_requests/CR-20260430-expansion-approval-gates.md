# CR-20260430 Expansion Approval Gates

## Summary

Add an API-free approval checklist helper for the next real Storyflow
expansion paths: API provider observation, Qwen3/server observation, Amazon
full-category preparation, and trained baseline artifacts.

## Storyflow Fit

Go. The change protects the Storyflow mainline by forcing every real expansion
to preserve title-level generation/selection, catalog grounding, provenance,
and claim discipline before any costly or high-risk run is attempted.

## Current Phase Fit

Go for Phase 3/4/5 readiness. The project has scaffolded observation,
baseline, server, CURE/TRUCE, simulation, and cross-category routes. The next
real expansion requires explicit approval gates rather than more silent
scaffold growth.

## Toy Risk

Low. This gate helps move beyond toy/mock work by specifying what is needed for
real provider, server, full-data, and trained-baseline execution.

## Stitching Risk

Low. The helper keeps all expansion paths tied to the existing Storyflow
contracts: grounded title outputs, claim-scoped manifests, and
exposure-counterfactual confidence framework boundaries.

## Requires API

No. The helper only writes approval manifests. API execution remains blocked
until the user explicitly approves provider/model/budget/rate/sample details
and the command includes `--execute-api`.

## Requires Server

No. The helper can describe server approval needs, but it does not execute a
server command.

## Requires Full Data

No. It can describe full-data approval needs, but it does not download or
process full data.

## Decision

Go.

## Minimal Implementation Scope

- Add `scripts/build_expansion_approval_checklist.py`.
- Add tests covering all tracks and non-result manifest flags.
- Add docs and README command entry.
- Update the decision log.

## Acceptance Criteria

- The helper writes ignored JSON/Markdown outputs with API/server/training/data
  flags set to false.
- The API, Qwen3/server, Amazon full-prepare, and baseline-artifact tracks each
  list required confirmations, preflight commands, execution templates, and
  forbidden-without-approval actions.
- Tests pass without network, API keys, data download, server execution, or
  model training.
