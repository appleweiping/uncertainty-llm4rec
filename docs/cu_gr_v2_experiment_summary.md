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

The RecBole external-baseline adapter now exports canonical TRUCE data, trains SASRec/BERT4Rec/GRU4Rec/LightGCN, scores the shared TRUCE candidate sets, imports predictions into the TRUCE schema, and evaluates with the TRUCE evaluator. MovieLens 1M completed end to end on CPU: SASRec reached Recall@10 0.184934, NDCG@10 0.108555, MRR@10 0.085336; BERT4Rec reached Recall@10 0.199172, NDCG@10 0.107392, MRR@10 0.079387; GRU4Rec reached Recall@10 0.160099, NDCG@10 0.088435, MRR@10 0.066755; LightGCN reached Recall@10 0.212086, NDCG@10 0.107865, MRR@10 0.076519. Amazon Beauty completed end to end on CPU after a clean rerun: SASRec reached Recall@10 0.013333, NDCG@10 0.005158, MRR@10 0.002667; BERT4Rec, GRU4Rec, and LightGCN reached 0.000000 on all three metrics under the current adapter/config.

Paper framing should change accordingly: CU-GR v2 is not uniformly stronger than strong recommenders. On MovieLens, SASRec, BERT4Rec, and LightGCN slightly exceed CU-GR v2 fusion by NDCG@10; on Amazon Beauty, CU-GR v2 remains stronger than the RecBole baselines and the BM25/fallback row. The more defensible claim is that CU-GR v2 is complementary to strong recommenders, with gains over fallback/LLM controls in the tested domains.
