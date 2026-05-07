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

- `beauty` if the Week8 producer exports matching directories;
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
3. Validate converted artifacts with
   `scripts/validate_week8_same_candidate_processed.py`.
4. Generate project packets for official baseline families:
   TALLRec, OpenP5, DEALRec, LC-Rec, LLaRA, and LLM-ESR where feasible.
5. Run official-native Qwen3-8B base-model controlled baselines for
   beauty/books/electronics/movies.
6. Import every method with TRUCE `import_external_predictions.py --split test`.
7. Evaluate with TRUCE evaluator only.
8. Run observation analysis and paired/statistical comparison across Ours and
   the controlled baselines.

## Paper Position

Amazon Beauty can support pipeline validation and early controlled comparison.
The Week8 books/electronics/movies large same-candidate tasks should become the
main paper-scale benchmark once their metadata and summaries are finalized.

## Strict Validation

Formal Week8 conversion should use:

```bash
python scripts/convert_week8_same_candidate_to_truce.py \
  --task-dir <task_dir> \
  --output-dir <output_dir> \
  --domain <domain> \
  --split test \
  --strict-target-in-candidates
```

Then validate:

```bash
python scripts/validate_week8_same_candidate_processed.py \
  --root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test \
  --expected-users 10000 \
  --expected-candidates 101 \
  --expected-negatives 100
```

If Beauty Week8 directories are absent, do not mark the four-domain suite
complete; use the existing Beauty pipeline only as the early-domain lane.
