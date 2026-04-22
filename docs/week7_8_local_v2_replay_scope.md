# Week7.8 Local-v2 Replay Scope

## Position

Week7.8 is a dedicated replay week for the teacher-requested local 8B plus LoRA execution line. It is not a rework of Week7.7, not a reduced Week8 outer-compare week, and not a rollback to the legacy yes/no recommendation framing.

## Hard Boundaries

1. Do not remove or rewrite the official API route.
2. Do not regress the task framing back to legacy yes/no as the main recommendation problem.
3. Keep pointwise as diagnosis, candidate ranking as the main decision layer, and pairwise as support/mechanism.
4. Use full-domain results as the formal Week7.8 evidence base.
5. Treat local-v2 as the default replay execution model for this week.

## Route Separation

- Official API route:
  preserved for historical observation, teacher evidence, early uncertainty diagnostics, case studies, and later outer-compare context.
- Structured risk route:
  preserved as the strongest hand-crafted baseline.
- Local-v2 replay route:
  newly elevated to the Week7.8 default execution mainline.

## Formal Data Principle

Week7.8 formal conclusions must come from:

- `amazon_beauty` full
- `amazon_books` full
- `amazon_electronics` full
- `amazon_movies` full

`small`, compact, and startup subsets may be used only for startup checks, dry runs, and fault isolation.

## Current Day1 Status

The replay shell can be added immediately, but the current repository state shows that full-domain processed inputs are not yet fully materialized into the same runnable files used by the existing small/full ranking pipeline.

Observed on April 23, 2026:

- `data/processed/amazon_beauty` already has the established full pipeline shape in the project.
- `data/processed/amazon_books`, `data/processed/amazon_electronics`, and `data/processed/amazon_movies` currently do not yet expose the full set of expected runtime files such as `test.jsonl` and `ranking_test.jsonl` in a way that matches the existing experiment entrypoints.

Therefore, Day1 implementation focuses on establishing the replay shell, replay configs, and batch manifest without breaking current routes. Actual four-domain execution remains blocked until full-domain processed runtime files are aligned with the experiment entrypoints.

## Day1 Deliverables

- `docs/week7_8_local_v2_replay_scope.md`
- `docs/week7_8_replay_manifest.md`
- `docs/from_teacher_line_to_srpd_bridge.md`
- replay-specific model configs
- replay-specific pointwise/ranking/rerank config skeletons
- replay batch skeleton

## Acceptance

Day1 is complete when:

- official API code path is untouched
- local-v2 replay shell is visible and documented
- replay configs are separated from Week7.7 and Week8 configs
- full-domain replay intent is explicit
- current blockers are explicitly documented instead of hidden
