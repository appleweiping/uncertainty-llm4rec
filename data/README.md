# Data Directory Policy

This directory contains local data artifacts for Storyflow / TRUCE-Rec.

Tracked files:

- this README;
- `.gitkeep` placeholders in expected data subdirectories.

Ignored local artifacts:

- `data/raw/`: downloaded or manually placed raw datasets;
- `data/interim/`: extraction manifests, temporary indexes, and reports;
- `data/processed/`: processed tables, sequences, examples, and manifests;
- `data/cache/`: resumable download metadata and API/dataset caches.

Raw, interim, processed, and cache outputs are not committed by default. Small
test fixtures belong under `tests/fixtures/`, not under this directory.
