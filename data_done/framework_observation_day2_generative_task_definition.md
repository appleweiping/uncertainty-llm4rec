# Framework-Observation-Day2 Generative Recommendation Task Definition

Status: Beauty small smoke design. This branch pauses yes/no confidence extensions and starts generative recommendation uncertainty observation.

## Task

Input:
- User history as item titles/text snippets.
- Optional candidate pool, depending on setting.

Output:
```json
{
  "recommended_title": "...",
  "confidence": 0.0
}
```

`confidence` is raw verbalized confidence. It is an observed model output, not a calibrated probability and not the final method.

## Setting A: Open-Title Generation

The model receives only user history and freely generates a recommended title or short title-like item description.

Evaluation requires catalog retrieval:
- Normalize generated title.
- Match against catalog titles.
- Mark invalid/hallucinated generations when no reliable catalog match exists.

This setting is closest to open generative recommendation, but it has high hallucination and matching ambiguity risk. It should not be the first full run.

## Setting B: Candidate-Grounded Title Generation

The model receives user history plus a 6-item candidate pool and must generate the recommended title, not an item_id.

This remains generative because the output is a title string, but evaluation is cleaner:
- Check whether the generated title maps to a candidate title.
- Check whether the matched candidate is the target item.
- Compute HR/MRR/NDCG from candidate-pool grounding.

Day2 starts with Setting B, Beauty 100-user smoke, base Qwen3-8B first.

## Catalog Grounding

For each generated title, produce:
- `matched_item_id`
- `matched_title`
- `match_score`
- `match_rank`
- `is_valid_catalog_item`
- `hit_target`
- `target_rank_if_candidate_pool_available`
- `hallucination`

Matching stages:
1. Exact or normalized title match inside candidate pool.
2. Candidate-pool retrieval by token similarity.
3. Catalog title retrieval by normalized token similarity or TF-IDF/BM25-style retrieval.
4. Embedding retrieval can be added later if embeddings are already available.

## Metrics

Generation validity:
- `parse_success_rate`
- `schema_valid_rate`
- `valid_candidate_title_rate`
- `catalog_match_rate`
- `hallucination_rate`

Recommendation quality:
- `HR@1`
- `MRR`
- `NDCG@3`
- `NDCG@5`
- `NDCG@10`

Confidence quality:
- `confidence_mean`
- `confidence_std`
- `confidence_unique_count`
- `confidence_ge_0.9_rate`
- `ECE_for_generation_correctness`
- `Brier_for_generation_correctness`
- `AUROC_for_generation_correctness`
- `high_conf_wrong_rate`

Calibration:
- Fit on valid split and evaluate on test split.
- If only test smoke exists, report diagnostic confidence metrics only.
