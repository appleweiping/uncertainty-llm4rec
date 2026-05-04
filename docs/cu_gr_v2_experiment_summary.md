# CU-GR v2 Experiment Summary

## MovieLens Results

Seed42 CU-GR v2 fusion improves NDCG@10 from 0.049339 to 0.10606. The held-out harmful swap rate is 0.035.

## Amazon Beauty Results

Seed42 CU-GR v2 fusion improves NDCG@10 from 0.142486 to 0.154217. The held-out harmful swap rate is 0.015.

## v1 Failure Motivation

MovieLens R3 artifacts retain negative evidence for free-form generation and confidence-heavy CU-GR v1 policies, including high-confidence wrong recommendations and weak/negative ranking movement versus fallback.

## Uncertainty Observation

MovieLens uncertainty artifacts show high verbalized confidence with poor calibration for direct/confidence-observation methods. Amazon Beauty only has CU-GR v2 parser confidence for this gate, not a direct-generation uncertainty run.

## Ablation Status

The ablation table includes fallback, listwise-only, fixed fusion, train-best fusion, safe fusion, offline no-term replays, and panel-size oracle feasibility rows where available.

## Cost / Latency

CU-GR v2 seed42 effective cost per 200 examples: MovieLens 0.059083, Amazon Beauty 0.076183. Full details are in `outputs/tables/paper_cost_latency.csv`.
