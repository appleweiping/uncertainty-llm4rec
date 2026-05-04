# CU-GR v2 Limitations

- Only two datasets/domains have passed so far.
- Evaluation uses a sampled candidate protocol, not full ranking.
- LLM calls use subset_size=200 per seed.
- Amazon Beauty local catalog size is 479, so candidate_size=500 was infeasible there.
- Validation is limited to DeepSeek v4 flash.
- No local open-source model validation has been run yet.
- No full production-scale inference validation has been run.
- Candidate panel construction is still heuristic.
- Harmful swaps are controlled but not zero.
