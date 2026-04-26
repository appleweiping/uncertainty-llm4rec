# Day38 Movies Small Fallback Sensitivity Report

## 1. Motivation

Day37 movies_small gives positive cross-domain sanity results, but ID-backbone fallback is not low. This analysis checks whether CEP gains are only a fallback artifact.

## 2. Fallback Distribution

- sasrec: fallback `0.4100`, positive fallback `0.8520`, negative fallback `0.3216`.
- gru4rec: fallback `0.5710`, positive fallback `0.8980`, negative fallback `0.5056`.
- bert4rec: fallback `0.5710`, positive fallback `0.8980`, negative fallback `0.5056`.

## 3. Warm-Positive Subset

- sasrec: warm-positive users `74`; A NDCG `0.5872` / MRR `0.4570`, D NDCG `0.6240` / MRR `0.5034`.
- gru4rec: warm-positive users `51`; A NDCG `0.7317` / MRR `0.6448`, D NDCG `0.7579` / MRR `0.6781`.
- bert4rec: warm-positive users `51`; A NDCG `0.5612` / MRR `0.4252`, D NDCG `0.5985` / MRR `0.4719`.

## 4. Cold-Positive Subset

- sasrec: cold-positive users `426`; A NDCG `0.5331` / MRR `0.3813`, D NDCG `0.6126` / MRR `0.4862`.
- gru4rec: cold-positive users `449`; A NDCG `0.4798` / MRR `0.3151`, D NDCG `0.5644` / MRR `0.4244`.
- bert4rec: cold-positive users `449`; A NDCG `0.4777` / MRR `0.3119`, D NDCG `0.5723` / MRR `0.4350`.

## 5. Signal Sanity

- bert4rec / random_score_same_distribution_B: rel NDCG mean `0.0441`, rel MRR mean `0.0870`.
- bert4rec / random_score_same_distribution_D: rel NDCG mean `0.0536`, rel MRR mean `0.1060`.
- bert4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0325`, rel MRR mean `0.0645`.
- bert4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.0403`, rel MRR mean `0.0802`.
- gru4rec / random_score_same_distribution_B: rel NDCG mean `0.0423`, rel MRR mean `0.0800`.
- gru4rec / random_score_same_distribution_D: rel NDCG mean `0.0423`, rel MRR mean `0.0800`.
- gru4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0418`, rel MRR mean `0.0792`.
- gru4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.0418`, rel MRR mean `0.0792`.
- sasrec / random_score_same_distribution_B: rel NDCG mean `0.0218`, rel MRR mean `0.0435`.
- sasrec / random_score_same_distribution_D: rel NDCG mean `0.0253`, rel MRR mean `0.0497`.
- sasrec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0196`, rel MRR mean `0.0388`.
- sasrec / shuffled_calibrated_relevance_D: rel NDCG mean `0.0265`, rel MRR mean `0.0513`.

## 6. Fallback Indicator Baseline

- bert4rec: best fallback-indicator baseline rel NDCG `-0.0059`, rel MRR `-0.0111` at lambda `0.05` / `zscore`.
- gru4rec: best fallback-indicator baseline rel NDCG `-0.0026`, rel MRR `-0.0049` at lambda `0.05` / `zscore`.
- sasrec: best fallback-indicator baseline rel NDCG `-0.0015`, rel MRR `-0.0025` at lambda `0.05` / `zscore`.

## 7. Conclusion

If warm-positive improvements are positive, movies_small supports cross-domain directionality beyond pure positive fallback. If improvements are concentrated in cold-positive users, the correct claim is that CEP helps compensate for weak/cold backbone scores in this small-domain sanity setting. In all cases, do not describe movies_small as a fully healthy external-backbone benchmark.

## 8. Relation To Main Evidence

Beauty full three-backbone multi-seed remains the primary performance evidence. Movies_small is cross-domain sanity / continuity evidence and is useful because it reproduces the direction with a different domain while explicitly exposing fallback sensitivity.
