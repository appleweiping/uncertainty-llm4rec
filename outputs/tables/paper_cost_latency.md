| Dataset | Method / gate | live requests | cache hits | tokens | estimated cost | p50 latency | p95 latency | retry / timeout / 429 count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MovieLens 1M | CU-GR v2 listwise gate seed42 | 200 | 0 | 306653 | 0.059083220000000006 | 7.37451009999495 | 7.950626899983035 | 0/0/0 |
| MovieLens 1M | CU-GR v2 listwise gate seedaggregate | 400 | 200 | 918723 | 0.17688580000000004 | 4.983327999992373 | 5.431632633320987 | 0/0/0 |
| Amazon Beauty | CU-GR v2 listwise gate seed42 | 200 | 0 | 429127 | 0.07618338000000001 | 7.7298008000070695 | 8.830691399984062 | 0/0/0 |
| Amazon Beauty | CU-GR v2 listwise gate seedaggregate | 600 | 0 | 1287713 | 0.22858864 | 7.551838866677524 | 8.395559666658906 | 0/0/0 |
| MovieLens 1M | llm_generative_real | 5840 | 12280 | 122027928 | 17.372053 | 1.040225 | 1.666662 |  |
| MovieLens 1M | llm_confidence_observation_real | 0 | 18120 | 122027928 | 17.372053 | 0.165517 | 0.505609 |  |
| MovieLens 1M | ours_uncertainty_guided_real | 17920 | 200 | 122027536 | 17.371944 | 2.203485 | 2.674855 |  |
