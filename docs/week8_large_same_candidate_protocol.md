# Week8 Large Same-Candidate Protocol

This document records the large-scale evaluation data being produced in the
parallel server project. It is the intended upgrade path after the current
Amazon Beauty early controlled-baseline pipeline.

## Source Location

Server project:

```text
~/projects/pony-rec-rescue-shadow-v6
```

Data root:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/
```

Large same-candidate task directories:

```text
{domain}_large10000_100neg_{valid,test}_same_candidate/
```

Target domains:

- `books`
- `electronics`
- `movies`

## Protocol

This is a reusable large-scale evaluation protocol, not a result table.

- Up to 10,000 users per domain.
- Each event has 1 positive and 100 negatives.
- Same-candidate setting: all methods must score the same candidate items.
- Negative sampling: popularity.
- Test history mode: `train_plus_valid`.
- Seed: `20260506`.
- Shuffle seed: `42`.

Do not resample users, negatives, candidates, or histories. Any TRUCE adapter
must read these task directories and preserve event/candidate alignment.

## Expected Files

Each task directory should contain:

- `ranking_valid.jsonl` or `ranking_test.jsonl`;
- `candidate_items.csv`;
- `train_interactions.csv`;
- `item_metadata.csv`;
- `selected_users.csv`;
- `recbole/{dataset}.inter`;
- `metadata.json`.

Check available files on the server:

```bash
find ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  -path "*large10000_100neg*" -type f | sort
```

Summary outputs from the data project should appear under:

```text
~/projects/pony-rec-rescue-shadow-v6/outputs/summary/
```

## Required Output Alignment

Models and adapters must preserve these fields whenever present:

- `event_id`;
- `source_event_id`;
- `user_id`;
- `item_id`;
- `split`.

This is required for paired comparison, oracle analysis, rank fusion, and
statistical testing across methods.

## TRUCE Migration Plan

The current Amazon Beauty controlled-baseline suite is an early pipeline and
debugging dataset. Final top-conference-strength experiments should migrate the
same framework to the Week8 large same-candidate tasks.

Migration steps:

1. Add a converter that reads a Week8 task directory and writes TRUCE-compatible
   `examples.jsonl`, `candidate_sets.jsonl`, `items.csv`, and
   `preprocess_manifest.json` without changing candidates or negatives.
2. Preserve `event_id/source_event_id` in example metadata.
3. Generate project packets for the Main4 controlled baselines:
   TALLRec, OpenP5-style, DEALRec, and LC-Rec.
4. Run the same Qwen3-8B LoRA controlled suite for books/electronics/movies.
5. Import every method with TRUCE `import_external_predictions.py --split test`.
6. Evaluate with TRUCE evaluator only.
7. Run observation analysis and paired/statistical comparison across Ours and
   the controlled baselines.

## Paper Position

Amazon Beauty can support pipeline validation and early controlled comparison.
The Week8 books/electronics/movies large same-candidate tasks should become the
main paper-scale benchmark once their metadata and summaries are finalized.
