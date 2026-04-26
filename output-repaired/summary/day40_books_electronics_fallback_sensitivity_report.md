# Day40 Books/Electronics Small Fallback Sensitivity Report

## 1. Motivation

Day39 books/electronics small-domain plug-in results were directionally positive, but fallback caveats were clear. This local-only Day40 analysis mirrors Day38 movies_small to check whether gains are fallback artifacts, warm-positive directionality, or cold/fallback compensation.

## 2. Fallback Distribution

- books_small / sasrec: fallback `0.4440`, positive fallback `0.9700`, negative fallback `0.3388`.
- books_small / gru4rec: fallback `0.6363`, positive fallback `0.9800`, negative fallback `0.5676`.
- books_small / bert4rec: fallback `0.6363`, positive fallback `0.9800`, negative fallback `0.5676`.
- electronics_small / sasrec: fallback `0.4813`, positive fallback `0.9480`, negative fallback `0.3880`.
- electronics_small / gru4rec: fallback `0.6707`, positive fallback `0.9680`, negative fallback `0.6112`.
- electronics_small / bert4rec: fallback `0.6707`, positive fallback `0.9680`, negative fallback `0.6112`.

## 3. Warm-Positive Subset

- books_small / sasrec: warm-positive users `15`, D relative NDCG `0.1741`, D relative MRR `0.2796`.
- books_small / gru4rec: warm-positive users `10`, D relative NDCG `0.2827`, D relative MRR `0.4403`.
- books_small / bert4rec: warm-positive users `10`, D relative NDCG `0.3373`, D relative MRR `0.5859`.
- electronics_small / sasrec: warm-positive users `26`, D relative NDCG `0.1627`, D relative MRR `0.2758`.
- electronics_small / gru4rec: warm-positive users `16`, D relative NDCG `0.2285`, D relative MRR `0.3815`.
- electronics_small / bert4rec: warm-positive users `16`, D relative NDCG `0.0493`, D relative MRR `0.0691`.

## 4. Cold-Positive Subset

- books_small / sasrec: cold-positive users `485`, D relative NDCG `0.3985`, D relative MRR `0.7520`.
- books_small / gru4rec: cold-positive users `490`, D relative NDCG `0.5142`, D relative MRR `1.0936`.
- books_small / bert4rec: cold-positive users `490`, D relative NDCG `0.4898`, D relative MRR `1.0372`.
- electronics_small / sasrec: cold-positive users `474`, D relative NDCG `0.2854`, D relative MRR `0.5486`.
- electronics_small / gru4rec: cold-positive users `484`, D relative NDCG `0.3249`, D relative MRR `0.6660`.
- electronics_small / bert4rec: cold-positive users `484`, D relative NDCG `0.3235`, D relative MRR `0.6741`.

## 5. Shuffled/Random Sanity

- books_small / bert4rec / random_score_same_distribution_B: rel NDCG mean `0.1151`, rel MRR mean `0.2465`.
- books_small / bert4rec / random_score_same_distribution_D: rel NDCG mean `0.1204`, rel MRR mean `0.2570`.
- books_small / bert4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.1020`, rel MRR mean `0.2184`.
- books_small / bert4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.1123`, rel MRR mean `0.2395`.
- books_small / gru4rec / random_score_same_distribution_B: rel NDCG mean `0.1135`, rel MRR mean `0.2424`.
- books_small / gru4rec / random_score_same_distribution_D: rel NDCG mean `0.1196`, rel MRR mean `0.2540`.
- books_small / gru4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.1337`, rel MRR mean `0.2842`.
- books_small / gru4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.1426`, rel MRR mean `0.3024`.
- books_small / sasrec / random_score_same_distribution_B: rel NDCG mean `0.0572`, rel MRR mean `0.1138`.
- books_small / sasrec / random_score_same_distribution_D: rel NDCG mean `0.0628`, rel MRR mean `0.1240`.
- books_small / sasrec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0558`, rel MRR mean `0.1095`.
- books_small / sasrec / shuffled_calibrated_relevance_D: rel NDCG mean `0.0636`, rel MRR mean `0.1240`.
- electronics_small / bert4rec / random_score_same_distribution_B: rel NDCG mean `0.0963`, rel MRR mean `0.1975`.
- electronics_small / bert4rec / random_score_same_distribution_D: rel NDCG mean `0.1136`, rel MRR mean `0.2306`.
- electronics_small / bert4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0918`, rel MRR mean `0.1870`.
- electronics_small / bert4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.1060`, rel MRR mean `0.2142`.
- electronics_small / gru4rec / random_score_same_distribution_B: rel NDCG mean `0.0992`, rel MRR mean `0.2030`.
- electronics_small / gru4rec / random_score_same_distribution_D: rel NDCG mean `0.1081`, rel MRR mean `0.2204`.
- electronics_small / gru4rec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0919`, rel MRR mean `0.1887`.
- electronics_small / gru4rec / shuffled_calibrated_relevance_D: rel NDCG mean `0.1019`, rel MRR mean `0.2082`.
- electronics_small / sasrec / random_score_same_distribution_B: rel NDCG mean `0.0720`, rel MRR mean `0.1420`.
- electronics_small / sasrec / random_score_same_distribution_D: rel NDCG mean `0.0845`, rel MRR mean `0.1653`.
- electronics_small / sasrec / shuffled_calibrated_relevance_B: rel NDCG mean `0.0538`, rel MRR mean `0.1065`.
- electronics_small / sasrec / shuffled_calibrated_relevance_D: rel NDCG mean `0.0638`, rel MRR mean `0.1247`.

## 6. Fallback Indicator Baseline

- books_small / bert4rec: best fallback-indicator baseline rel NDCG `-0.0037`, rel MRR `-0.0074` at lambda `0.05` / `zscore`.
- books_small / gru4rec: best fallback-indicator baseline rel NDCG `-0.0045`, rel MRR `-0.0092` at lambda `0.05` / `zscore`.
- books_small / sasrec: best fallback-indicator baseline rel NDCG `-0.0042`, rel MRR `-0.0073` at lambda `0.05` / `zscore`.
- electronics_small / bert4rec: best fallback-indicator baseline rel NDCG `-0.0044`, rel MRR `-0.0085` at lambda `0.05` / `zscore`.
- electronics_small / gru4rec: best fallback-indicator baseline rel NDCG `-0.0017`, rel MRR `-0.0031` at lambda `0.05` / `zscore`.
- electronics_small / sasrec: best fallback-indicator baseline rel NDCG `-0.0058`, rel MRR `-0.0108` at lambda `0.05` / `zscore`.

## 7. Cross-Domain Small Summary

- movies_small / sasrec: all-users rel NDCG `0.1353`, warm rel NDCG `0.0628`, cold rel NDCG `0.1492`, interpretation `fallback_heavy_caution`.
- movies_small / gru4rec: all-users rel NDCG `0.1557`, warm rel NDCG `0.0358`, cold rel NDCG `0.1764`, interpretation `fallback_heavy_caution`.
- movies_small / bert4rec: all-users rel NDCG `0.1825`, warm rel NDCG `0.0665`, cold rel NDCG `0.1980`, interpretation `fallback_heavy_caution`.
- books_small / sasrec: all-users rel NDCG `0.3909`, warm rel NDCG `0.1741`, cold rel NDCG `0.3985`, interpretation `sample_too_small`.
- books_small / gru4rec: all-users rel NDCG `0.5072`, warm rel NDCG `0.2827`, cold rel NDCG `0.5142`, interpretation `sample_too_small`.
- books_small / bert4rec: all-users rel NDCG `0.4863`, warm rel NDCG `0.3373`, cold rel NDCG `0.4898`, interpretation `sample_too_small`.
- electronics_small / sasrec: all-users rel NDCG `0.2787`, warm rel NDCG `0.1627`, cold rel NDCG `0.2854`, interpretation `sample_too_small`.
- electronics_small / gru4rec: all-users rel NDCG `0.3210`, warm rel NDCG `0.2285`, cold rel NDCG `0.3249`, interpretation `sample_too_small`.
- electronics_small / bert4rec: all-users rel NDCG `0.3077`, warm rel NDCG `0.0493`, cold rel NDCG `0.3235`, interpretation `sample_too_small`.

## 8. Claim Boundary

Small-domain results are cross-domain sanity / continuity evidence. They should not be described as fully healthy external-backbone benchmarks when fallback is high. Beauty full three-backbone multi-seed remains the primary performance evidence. Day40 helps phrase small-domain gains as either warm-positive directionality or fallback/cold compensation, depending on each domain/backbone.
