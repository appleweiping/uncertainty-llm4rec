# CR-20260430 Expansion Run Packets

## Summary

Add a non-executing run-packet helper for one selected expansion track after
the broader approval checklist has been generated.

## Motivation

The project now has several guarded expansion paths: real API provider
observation, Qwen3/server observation, Amazon full-category preparation, and
trained baseline artifacts. The existing approval checklist records what must
be confirmed, but the next operational step needs a tighter artifact that maps
one selected track to concrete preflight commands, the approval-required
command shape, expected outputs, missing confirmations, and forbidden claims.

## Storyflow Alignment

Go. This supports the Storyflow mainline by making the next real expansion
auditable before any API, server, full-data, or trained-baseline execution. It
preserves title-level grounding as a required postcondition before correctness
or confidence analysis.

## Phase Fit

Current phase: Phase 3 readiness / Phase 4-5 scaffold governance.

The helper does not add a new research method. It reduces execution risk for
already planned expansion tracks.

## Risk Checks

- Toy risk: low. It pushes toward real expansion readiness rather than adding
  another toy artifact.
- Stitching risk: low. It is governance/manifest code only.
- API required: no.
- Server required: no.
- Full data required: no.
- Training required: no.

## Decision

Go.

## Minimal Scope

- Add `scripts/build_expansion_run_packet.py`.
- Add pytest coverage for non-execution flags, missing confirmations, command
  rendering, and CLI output.
- Update README and execution docs.

## Acceptance Criteria

- Generated packets record all execution/result flags as false.
- Generated packets list missing confirmations and forbidden claims.
- Execute commands are present only as approval-required templates.
- Tests pass without API calls, server execution, data download, full data
  preparation, or model training.
