# Framework-Day10 Candidate Order Diagnostics Report

## Scope

Day10 creates Beauty-only candidate-order-shuffled LoRA instruction data. This does not train, call APIs, or implement confidence/evidence/CEP framework.

## Outputs

- train: listwise `622`, pointwise `3732`, pool mean `6`
- valid: listwise `622`, pointwise `3732`, pool mean `6`
- test: listwise `622`, pointwise `3732`, pool mean `6`

## Positive Position Check

- old listwise test position-1 rate: `0.13987138263665594`
- old pointwise test position-1 rate: `1.0`
- shuffled listwise test position-1 rate: `0.16720257234726688`
- shuffled pointwise test position-1 rate: `0.16720257234726688`

Day9.5 identified the pointwise route as order-biased because old pointwise positives are fixed at position 1. The shuffled versions should spread positives across positions 1-6 under seed 42.

## Boundary

The shuffled JSONL files are generated artifacts and should not be committed. Commit scripts/configs/reports only.
