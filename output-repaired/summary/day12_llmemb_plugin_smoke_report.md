# Day12 LLMEmb Plug-in Smoke Report

## Status

Blocked before performance evaluation.

Reason: LLMEmb candidate score export is not present locally; no synthetic backbone scores were generated.

The local project does not currently contain a real LLMEmb score export at:

`output-repaired\backbone\llmemb_beauty_100\candidate_scores.csv`

The plug-in reranker is implemented in `main_day12_llmemb_plugin_smoke.py`, but it requires a real external backbone table with this schema:

`user_id, candidate_item_id, backbone_score, label`

Optional columns are:

`backbone_rank, split, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success`

## Evidence Source

Day9 evidence source expected by the script:

`output-repaired\beauty_deepseek_relevance_evidence_full\calibrated\relevance_evidence_posterior_test.jsonl`

Required Day9 fields:

`relevance_probability, calibrated_relevance_probability, evidence_risk, ambiguity, missing_information, abs_evidence_margin, positive_evidence, negative_evidence`

## Next Step

Clone or place LLMEmb under an external-code directory, run a 100-user Beauty evaluation/export, and write:

`output-repaired/backbone/llmemb_beauty_100/candidate_scores.csv`

Then rerun:

```powershell
py -3.12 main_day12_llmemb_plugin_smoke.py
```

No synthetic backbone scores were generated, so no external-backbone performance claim is made.
