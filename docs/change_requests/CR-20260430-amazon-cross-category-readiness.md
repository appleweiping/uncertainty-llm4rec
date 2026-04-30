# CR-20260430 Amazon Cross-Category Readiness

## Summary

Add an API-free Amazon Reviews 2023 cross-category readiness matrix and complete
server-scale configs for Video_Games, Sports_and_Outdoors, and Books.

## Storyflow Fit

Go. The change directly supports the Storyflow requirement to move beyond
MovieLens and avoid overfitting the project narrative to a single Amazon Beauty
category. These categories remain title-level generative recommendation
substrates: user history titles lead to generated/selected titles, catalog
grounding, and joint confidence/correctness/popularity analysis.

## Current Phase Fit

Go for Phase 1/3 readiness. This is a configuration, audit, and runbook gate
only. It does not download data, call APIs, train models, or claim results.

## Toy Risk

Low. The change broadens the real-data route toward title-rich and robustness
domains while keeping synthetic fixtures limited to tests.

## Stitching Risk

Low. All categories enter the same Amazon review-to-interaction and
metadata-to-catalog pipeline, then the same observation, grounding, confidence,
baseline, and CURE/TRUCE feature contracts.

## Requires API

No.

## Requires Server

No for readiness inspection. Full Books, Video_Games, and Sports processing is
server-scale or manual-raw gated.

## Requires Full Data

No for this change. Full raw files are only path templates until the user places
them or runs a server download under the dataset license/access requirements.

## Decision

Go.

## Minimal Implementation Scope

- Add `configs/datasets/amazon_reviews_2023_books.yaml`.
- Complete raw path, schema, sample, and full command templates for
  Video_Games and Sports.
- Add `scripts/inspect_amazon_category_matrix.py` to write ignored readiness
  JSON/CSV/Markdown artifacts.
- Update README/docs and tests.

## Acceptance Criteria

- Matrix inspection writes artifacts with `api_called=false`,
  `server_executed=false`, `full_download_attempted=false`, and
  `is_experiment_result=false`.
- New configs expose guarded sample commands and full commands containing
  `--allow-full`.
- Docs state that this is readiness only, not paper evidence.
