# Amazon Reviews 2023 Readiness

Amazon Reviews 2023 is the intended full-scale e-commerce data source after
MovieLens sanity checks. This document defines the entry point; it does not
claim that full data has been downloaded or processed.

## Beauty Config

Primary config:

```text
configs/datasets/amazon_reviews_2023_beauty.yaml
```

The config records:

- source dataset and category;
- local raw/cache/processed paths;
- expected review and metadata fields;
- title/user/item/rating/timestamp fields;
- metadata join plan;
- k-core and interaction filtering settings;
- local sample mode;
- server full mode command template.

Current local raw Beauty files detected by config:

```text
data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl
data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl
```

Compressed copies may also exist next to them. Raw and processed data remain
ignored by git.

Additional local raw Amazon category configs currently registered:

- `configs/datasets/amazon_reviews_2023_digital_music.yaml`
- `configs/datasets/amazon_reviews_2023_handmade.yaml`
- `configs/datasets/amazon_reviews_2023_health.yaml`

Server-scale Amazon category configs currently registered:

- `configs/datasets/amazon_reviews_2023_video_games.yaml`
- `configs/datasets/amazon_reviews_2023_sports.yaml`
- `configs/datasets/amazon_reviews_2023_books.yaml`

Beauty remains the first full-data gate; the extra categories are robustness
candidates after the Beauty path is stable. Books and Video_Games are
especially useful as title-rich domains, but their full processing remains
server/manual-raw gated.

Local status on 2026-04-30: after Beauty was stabilized, Digital_Music,
Handmade_Products, and Health_and_Personal_Care were also processed locally
from already placed raw JSONL files with `--allow-full`. These outputs are
data-readiness artifacts only, remain ignored by git, and are not API/model
runs or paper evidence. The current 5-core/global-chronological manifests are:

| dataset | users | catalog items | interactions | rolling examples | test | no-repeat test | repeat-only test | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Digital_Music | 20 | 29 | 157 | 97 | 10 | 0 | 10 | Repeat-heavy diagnostic domain under current filtering. |
| Handmade_Products | 89 | 95 | 546 | 279 | 28 | 2 | 26 | Repeat-heavy; no-repeat slice too small for main observation. |
| Health_and_Personal_Care | 157 | 175 | 1189 | 718 | 72 | 60 | 12 | Best current additional local no-repeat observation substrate. |

Health validation reports one duplicate normalized title group, so grounding
ambiguity should be tracked in analysis. Digital_Music and Handmade are still
useful for repeat/copy-history diagnostics, but not as primary no-repeat
domains unless preprocessing settings are revisited.

## Lightweight Inspect

Default dry-run/readiness command:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run --sample-records 3
```

This writes ignored outputs under:

```text
outputs/amazon_reviews_2023/amazon_reviews_2023_beauty/inspect/
```

It does not download full data and does not claim availability success. Optional
online checks must be requested explicitly:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --check-online
```

If network, license, login, or dependency conditions fail, the script records a
clear status and recovery command instead of silently skipping.

Cross-category readiness matrix:

```powershell
python scripts/inspect_amazon_category_matrix.py --sample-records 3
```

This command writes ignored artifacts under:

```text
outputs/amazon_reviews_2023/category_matrix/
```

The matrix records raw review/metadata file availability, guarded sample/full
command templates, and the next action for each configured Amazon category. It
does not download full data, call APIs, train models, or create paper evidence.

## Prepare Entry

Dry-run/readiness:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Full prepare command shape once raw JSONL files exist and the full run is
explicitly approved:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl --output-suffix full --allow-full
```

Local status on 2026-04-29: the command above was executed against the
user-provided Beauty raw files and validated with:

```powershell
python scripts/validate_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix full
```

Validated processed output:

- path: `data/processed/amazon_reviews_2023_beauty/full/`;
- users: 357;
- catalog items: 479;
- interactions: 3315;
- observation examples: 2244;
- split counts: train 1795, val 224, test 225;
- validation status: `ok`;
- warning: at least one repeated-item example has target appearing in history
  and should be inspected before paper claims.

Run the full processed observation audit after validation:

```powershell
python scripts/audit_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix full
```

This writes ignored artifacts under
`outputs/data_audits/amazon_reviews_2023_beauty/full/`. The audit does not
change processed data. It counts repeated target-in-history examples, duplicate
history items, chronological split boundaries, sequence-prefix alignment,
history length distribution, bucket coverage, and title-quality issues. Repeat
targets are not automatically leakage in e-commerce data, but they must be
stratified or sensitivity-checked before full observation claims.

Build repeat-aware observation input slices after audit:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy exclude
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy only
```

These inputs do not call an API and do not overwrite the default
`test_forced_json.jsonl`; they write separate repeat-free and repeat-only files
under ignored `outputs/observation_inputs/...`.

This is a full preprocessing readiness artifact only. It is not an API full
observation, not a model result, and not paper evidence.

Small sample prepare command shape for local pipeline sanity:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --sample-mode --max-records 5000 --output-suffix sample_5k --min-user-interactions 1 --user-k-core 1 --item-k-core 1 --min-history 1 --max-history 20
```

Sample prepare only scans the requested review prefix and then loads metadata
for the surviving item ids. It writes `is_sample_result=true`,
`is_full_result=false`, and `is_experiment_result=false` in the preprocess
manifest. It is not a full processed result and must not be used for paper
claims.

The script blocks accidental full preprocessing unless `--allow-full` is
provided. This prevents a local readiness command from becoming an unapproved
full-data run.

The prepare code defines:

- review row -> canonical interaction conversion;
- metadata row -> item catalog conversion;
- title cleaning;
- timestamp sorting;
- interaction-count filtering;
- k-core filtering;
- leave-last-two and rolling/global chronological settings;
- item popularity and head/mid/tail bucket assignment.

## Sample Observation Gate

After the local `sample_5k` prepare exists, validate it before constructing any
prompt inputs:

```powershell
python scripts/validate_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k
```

Then build a small stratified input file for future observation:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity
```

Expected ignored output:

```text
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl
```

This gate only proves that the processed sample can feed the title-level
generative observation pipeline. It does not call an API, does not train a
model, and does not convert the sample into a full Amazon result.

After pilot case review or grounding diagnostics, build the v2 no-API gate that
compares free-form, catalog-constrained, and retrieval-context input variants:

```powershell
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity --candidate-count 20
```

The v2 gate writes ignored JSONL inputs and a manifest under:

```text
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/
```

Gate JSONL files are name-spaced as `test_gate30_*.jsonl` so diagnostic inputs
do not overwrite the full free-form split input `test_forced_json.jsonl`.

For the local Beauty full prepare, the full test split input was built with:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --stratify-by-popularity
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --max-examples 30 --stratify-by-popularity --candidate-count 20
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --max-examples 185 --stratify-by-popularity --candidate-count 20 --repeat-target-policy exclude
```

The full free-form input contains 225 test examples. The diagnostic gate files
contain either 30 stratified examples for quick QA or 185 repeat-free examples
for no-repeat prompt/grounding comparison, and they do not overwrite the
225-example input.

The retrieval-context variant uses history-title token overlap to select
catalog titles without including the held-out target by default. It is a
prompt/grounding readiness artifact only, not an API result and not paper
evidence.

For the local Health_and_Personal_Care full prepare, the full test split and
repeat-aware inputs were built with:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_health --processed-suffix full --split test --stratify-by-popularity
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_health --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy exclude
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_health --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy only
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_health --processed-suffix full --split test --max-examples 60 --stratify-by-popularity --candidate-count 20 --repeat-target-policy exclude
```

The Health full free-form input contains 72 test examples; the repeat-free
input and the target-excluding forced-JSON/retrieval-context/catalog-constrained
gate variants each contain 60 no-repeat examples. These are observation inputs
only; no DeepSeek/API call has been made for Health.

Local status on 2026-04-29: the repeat-free retrieval-context diagnostic input
above was run through DeepSeek after readiness passed. The completed ignored
run has 185 grounded rows and zero failed cases. It shows that candidate
context can make the model output catalog-groundable titles on this Beauty
slice, but it must not be reported as recommendation accuracy because the
held-out target is excluded from the diagnostic candidate list.

The analysis layer now also writes candidate diagnostics for that run. The
ignored diagnostic summary reports `162/185` grounded outputs selected a
provided candidate title, `0/185` targets appeared in the candidate list, and
selected candidate buckets were head=49, mid=73, tail=40. These numbers are
prompt/candidate QA signals for the Beauty gate, not paper evidence.

Local status on 2026-04-29: the repeat-free catalog-constrained diagnostic
input was also run through DeepSeek after readiness, 5-example dry-run, and
5-example smoke gates. The completed ignored run has 185 rows, zero failed API
cases, GroundHit `0.784`, `140/185` generated titles selected from the provided
candidate set, and `0/185` targets in the candidate list. It produced more
ungrounded low-confidence cases than retrieval-context (`40/185`), so it is a
useful prompt/candidate QA comparison, not a recommendation-accuracy result.

## Server Guidance

Full Amazon Beauty download and preprocessing should run on a server or local
machine with enough disk, network, and runtime budget. Codex has run local full
preprocessing from already placed raw JSONL files, but has not run any server
pipeline and cannot claim server completion without logs/artifacts.

For Video_Games, Sports_and_Outdoors, and Books, first run the cross-category
readiness matrix, then place the raw review and metadata JSONL files at the
configured paths before any sample/full prepare. Full prepare still requires
`--allow-full`, a machine with enough storage/runtime, and a recorded manifest.
For the local raw categories already processed, do not treat the manifests as
paper evidence; use them to choose the next approved observation slice.
