# CR-20260428 Amazon Local Categories

## Summary

Recognize user-provided local Amazon Reviews 2023 raw files under `data/raw/`
for Beauty, Digital_Music, Handmade_Products, and Health_and_Personal_Care.

## Storyflow Fit

Go. These are real Amazon Reviews 2023 categories and directly support the
Storyflow requirement to move beyond MovieLens sanity checks toward full
e-commerce title-level generative recommendation.

## Current Phase Fit

Go for Phase 1/2 readiness. The change adds configs and lightweight readiness
inspection only. It does not claim full preprocessing, API results, or paper
evidence.

## Toy Risk

Low. The change moves the project away from toy/mock-only work by recognizing
real local raw datasets. Synthetic fixtures remain tests only.

## Stitching Risk

Low. The categories enter the same Amazon review -> interaction, metadata ->
catalog, grounding, confidence, and observation pipeline.

## Requires API

No.

## Requires Server

No for readiness inspection. Full preprocessing may require a server or large
disk/runtime depending on category scale.

## Requires Full Data

The raw files are present locally, but this CR does not execute full processing.
Full processing remains a separate user-approved/runbook-controlled step.

## Decision

Go.

## Minimal Implementation Scope

- Update Beauty config to match actual local file names.
- Add configs for Digital_Music, Handmade_Products, and Health_and_Personal_Care.
- Update Amazon readiness inspection to report actual local raw file existence,
  sizes, and schema samples.
- Keep raw files, processed outputs, and reports ignored by git.

## Acceptance Criteria

- `scripts/inspect_amazon_reviews_2023.py` can inspect the local Beauty files
  with `--sample-records` without full preprocessing.
- Tests cover local raw path resolution and schema sampling on tiny fixtures.
- README/docs distinguish readiness/sample checks from full processed results.
