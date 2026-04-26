# beauty data_done Processing Report

- Source interactions: `data\raw\amazon_beauty\reviews_Beauty.jsonl` via normalized `data\processed\amazon_beauty\interactions.csv`
- Source item metadata: `data\raw\amazon_beauty\meta_Beauty.jsonl` via normalized `data\processed\amazon_beauty\items.csv`
- Strategy: `user_min4`
- Seed: `42`
- Users: `622`
- Valid rows: `3732`
- Test rows: `3732`
- Negative sampling: warm train-vocab first, with recorded fallback modes if needed.
- Candidate pool size is 6, so HR@10 is trivial and should not be a primary metric.


## Framework-Day2 pool-size annotation

- `candidate_pool_setting = 5neg`
- `candidates_per_user_per_split = 6`
- `hr10_trivial_flag = true`
- Recommended primary metrics: NDCG@10, MRR, HR@1, HR@3, NDCG@3, NDCG@5.
