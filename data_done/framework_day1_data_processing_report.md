# Framework-Day1 Data Processing Report

## 1. Motivation

Observation-stage artifacts mixed old processed splits, small/medium variants, and experiment outputs. Framework-stage work needs a clean, reproducible data foundation for Qwen-LoRA, local evidence generation, ID-based backbones, and external baselines. Framework-Day1 therefore writes new artifacts under `data_done/` without overwriting `data/processed/`.

## 2. Literature / Protocol Audit

Local OpenP5 and LLM-ESR notes confirm that Amazon-style sequential recommendation commonly relies on preprocessed sequential data and cold-start filtering, but the visible local READMEs do not specify every filtering/evaluation detail. Our implemented project protocol is chronological leave-one-out with one valid and one test positive per user. Prefix-style examples are deferred to LoRA training data generation.

## 3. Chosen Protocol

The recommended first foundation is `user_min4 + chronological leave-one-out + max 10,000 users/domain + warm negative sampling`. This preserves enough history for train/valid/test, keeps scale controlled, and makes ID-based sequential backbones healthier than all-item cold sampling. We still report user_min5 and 5-core strategy comparisons.

## 4. Domain Statistics

| domain | raw-derived interactions | raw-derived users | raw-derived items | users >=4 | users >=5 |
|---|---:|---:|---:|---:|---:|
| beauty | 6583 | 992 | 1184 | 622 | 438 |
| books | 12442971 | 1711729 | 963824 | 1104817 | 779589 |
| electronics | 17515103 | 3010273 | 533532 | 1852743 | 1251322 |
| movies | 7895387 | 1174881 | 258113 | 754113 | 527705 |

## 5. Split Statistics

| domain | train users | valid rows | test rows | items | avg history len | sampled users |
|---|---:|---:|---:|---:|---:|---|
| beauty | 622 | 3732 | 3732 | 1158 | 6.80 | False |
| books | 10000 | 60000 | 60000 | 76416 | 7.67 | True |
| electronics | 10000 | 60000 | 60000 | 44896 | 5.47 | True |
| movies | 10000 | 60000 | 60000 | 46366 | 6.69 | True |

## 6. Cold/Warm Diagnostics

Each domain includes `cold_rate_diagnostics.csv` with valid/test cold rates under train_candidate_vocab, train_history_vocab, and train_backbone_vocab. Warm negative sampling should make negative cold rate close to zero under train_backbone_vocab. Positive cold rate may remain non-zero because chronological held-out positives can be item cold-start cases.

## 7. Limitations

- Beauty may have fewer than 10,000 eligible users; we do not fabricate users.
- If positive cold rate is high for a domain, ID-based backbone evaluation should be marked caution.
- The first split uses 1 positive + 5 negatives, so HR@10 is trivial. Primary metrics should be NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.
- A 20-negative evaluation split can be generated later from the same `data_done` foundation.
- Some Movies metadata rows have missing title/description; Framework-Day1 fills deterministic `Unknown item <item_id>` / `Item ID: <item_id>` placeholders so the schema remains complete for later Qwen-LoRA data generation.
- Raw Amazon JSON files are huge; Framework-Day1 uses existing normalized CSVs derived from those raw files and records the raw paths explicitly.

## 8. Next Steps

- Framework-Day2: run baseline pipeline sanity on `data_done` without API-heavy experiments.
- Framework-Day3: derive LoRA evidence-generator training pairs from train sequences.
- Framework-Day4: train or connect Qwen-LoRA evidence generator on the server.
