# CU-GR v2 Limitations

- Only two datasets/domains have passed so far.
- Amazon Video Games was evaluated as a third local domain but did not pass the held-out +0.01 NDCG@10 gate under the current CU-GR v2 panel/fusion design.
- Evaluation uses a sampled candidate protocol, not full ranking.
- LLM calls use subset_size=200 per seed.
- Amazon Beauty local catalog size is 479, so candidate_size=500 was infeasible there.
- Validation is limited to DeepSeek v4 flash.
- No local open-source model validation has been run yet.
- SASRec and LightGCN have adapter/config/export support, but RecBole was not installed in the current environment and no paper-grade SASRec/LightGCN metrics are available yet.
- No full production-scale inference validation has been run.
- Candidate panel construction is still heuristic.
- Harmful swaps are controlled but not zero.
