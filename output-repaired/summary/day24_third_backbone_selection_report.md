# Day24 Third Backbone Selection Report

| candidate_backbone | repo_url_or_local_path | model_type | requires_checkpoint | requires_special_embedding | supports_beauty_or_amazon | can_export_candidate_score | score_export_difficulty | expected_fallback_risk | recommended_choice | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM-ESR Bert4Rec | external/LLM-ESR (e5dc388c12509c88c65536ecd8d231325993d4ef) | masked-transformer sequential recommender | no | no | yes | yes, Bert4Rec.predict returns candidate logits | medium | same candidate cold-item risk as GRU4Rec; expected <20% | yes | Non-GRU external sequential backbone with real candidate logits and no missing checkpoint dependency. |
| OpenP5 | external/OpenP5 (7f110389cd5ab51820e29e94a44b6db83df243fb) | generative recommendation | yes | no direct embedding, but generated data/checkpoint required | yes | possible only with generative likelihood adapter | high | blocked by missing scoring adapter/checkpoint | no | Not suitable for Day24 smoke because it would require new generative score export work. |
| LLMEmb | external/LLMEmb (3458a5e225062e94b4f1a01e41f3ec82089f0407) | LLM-enhanced sequential recommendation | yes | yes | yes | yes in principle | blocked | blocked by missing handled data/embedding/checkpoint | no | Day13 already found required artifacts missing. |
| ItemKNN/co-occurrence | local implementation possible | history-based sanity baseline | no | no | yes | yes | low | low | fallback only | Useful as sanity, but too weak/coarse to serve as strong third backbone. |

Recommended Day24 choice: **LLM-ESR Bert4Rec**.
