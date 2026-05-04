# CU-GR v2 Experiment Summary

## MovieLens Results

Seed42 CU-GR v2 fusion improves NDCG@10 from 0.049339 to 0.10606. The held-out harmful swap rate is 0.035.

## Amazon Beauty Results

Seed42 CU-GR v2 fusion improves NDCG@10 from 0.142486 to 0.154217. The held-out harmful swap rate is 0.015.

## Amazon Video Games Results

Amazon Video Games was selected as the third local domain. Panel feasibility passed through positive oracle-gain evidence, but the held-out seed42 gate did not pass: fallback NDCG@10 was 0.096066, fusion_train_best NDCG@10 was 0.104206, and the delta was +0.008140, below the required +0.01 threshold. Harmful swap rate was 0.020 and parser success was 0.975.

## v1 Failure Motivation

MovieLens R3 artifacts retain negative evidence for free-form generation and confidence-heavy CU-GR v1 policies, including high-confidence wrong recommendations and weak/negative ranking movement versus fallback.

## Uncertainty Observation

MovieLens uncertainty artifacts show high verbalized confidence with poor calibration for direct/confidence-observation methods. Amazon Beauty only has CU-GR v2 parser confidence for this gate, not a direct-generation uncertainty run.

## Ablation Status

The ablation table includes fallback, listwise-only, fixed fusion, train-best fusion, safe fusion, offline no-term replays, and panel-size oracle feasibility rows where available.

## Cost / Latency

CU-GR v2 seed42 effective cost per 200 examples: MovieLens 0.059083, Amazon Beauty 0.076183, and Amazon Video Games 0.061430. Full details are in `outputs/tables/paper_cost_latency.csv` and `outputs/tables/cu_gr_v2_amazon_video_games_cost_latency.csv`.

## Strong Baseline Status

The RecBole external-baseline adapter now exports canonical TRUCE data and writes SASRec/LightGCN configs, but RecBole is not installed in the current environment. SASRec and LightGCN metrics are therefore not paper-candidate evidence yet.
