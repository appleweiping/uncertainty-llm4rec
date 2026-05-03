# CU-GR v2: Candidate-Normalized Preference Calibration (CNPC-Rec)

## 1. Why CU-GR v1 failed

- **Zero positive promote-generated labels** (`n_improve = 0`) on MovieLens R3
  candidate-500 for seeds 13/21/42: the counterfactual “insert LLM grounded title
  at rank 1” never improved NDCG@10 vs BM25 top-10.
- **Free-form title generation** is weak under a fixed 500-item candidate gate:
  the grounded item rarely beats the fallback ordering for the held-out target.
- **Verbalized confidence** from the generative path is **miscalibrated** relative
  to ranking outcomes, so thresholding raw confidence does not recover a safe
  override policy.

With no positive supervision, no binary calibrator can learn
`P(override_improves)` in that action space—the limitation is **label and
action design**, not the choice of classifier.

## 2. New hypothesis

The LLM may still supply **useful relative preference** among **items that are
already valid candidates**, when the task is localized (small panel, anonymous
labels, titles/genres only). Uncertainty is reframed from “trust this generated
ID?” to “does the LLM’s local ranking signal agree with and refine the fallback
on this panel?”

## 3. Method (v2)

1. **Local candidate panel** \(P\): 10–20 items drawn deterministically from the
   fallback ranking, mid-ranks, popularity contrasts, optional sequential hints,
   and (if already candidate-adherent) the grounded generative item—**never**
   inserting out-of-catalog IDs.
2. **Pairwise / listwise LLM prompts** over anonymous labels **A,B,C,…** with
   listwise scores or pairwise winners + confidences.
3. **Candidate-normalized** LLM scores (within-panel normalization) as features.
4. **Calibrated fusion** with fallback scores (linear or learned), thresholds fit
   on **validation seed only**.
5. **Risk-controlled reranking**: only permute positions occupied by panel items
   in the full fallback list; non-panel order unchanged; final top-10 for metrics.

## 4. Allowed inputs (prompts and features)

- User history titles (and similar history-derived text already in protocol).
- Panel item **titles** and **genres/categories** (from catalog metadata).
- Fallback ranks and scores for panel members (optional visibility per config).
- LLM listwise/pairwise outputs and parse/grounding metadata.
- **Train-only** popularity and buckets.

## 5. Forbidden inputs

- Target title; target item ID presented as “the correct answer.”
- Future interactions; any label peeking into test correctness **as a model
  feature** (offline labels for training/evaluation only are allowed with clear
  separation).
- Global item IDs in the visible prompt when anonymous local labels are used.

## 6. Evaluation

Compare (same candidate protocol, same metrics code paths as other R3 tooling):

- BM25 / fallback-only, `sequential_markov`, LLM direct, Ours v1, conservative
  gate (when artifacts exist).
- **Ranking:** Recall@10, NDCG@10, MRR@10.
- **Calibration / risk:** ECE-style slices where applicable; risk–coverage for
  fusion gates.
- **Behavior:** accepted rerank rate, harmful/beneficial/neutral swap counts,
  `panel_target_coverage`, parser success rate, cost/latency when API runs.

## 7. Offline feasibility (before any new LLM calls)

Script `scripts/build_candidate_panels.py` reports:

- `target_in_panel_rate` for panel sizes 10 / 15 / 20.
- `fallback_hit@10` baseline.
- Oracle upper bound when panel slots are reordered optimally for the target
  (target moved to the earliest fallback index occupied by any panel item).
- Count of examples with **strictly positive** oracle NDCG@10 or Recall@10 gain.

**Viability gate (documentation only for this repo stage):** proceed to a small
DeepSeek preference **subgate** only if panel oracle headroom suggests the panel
often contains the target and non-trivial rerank upside; otherwise **redesign
the panel** or stay **observation-first**.

## 8. Code map

| Piece | Path |
| --- | --- |
| Panel construction | `src/llm4rec/methods/candidate_panel.py` |
| Listwise / pairwise prompts | `src/llm4rec/prompts/preference_templates.py` |
| Response parsing | `src/llm4rec/prompts/preference_parser.py` |
| Fusion + replay (after signals exist) | `src/llm4rec/methods/preference_fusion.py`, `scripts/build_preference_dataset.py`, `scripts/train_preference_fusion.py`, `scripts/replay_preference_fusion.py` |
| Offline panel stats | `scripts/build_candidate_panels.py` → `outputs/tables/cu_gr_v2_panel_coverage.csv` |

## 9. Artifact outputs (tables)

| File | Source |
| --- | --- |
| `outputs/tables/cu_gr_v2_panel_coverage.csv` | `build_candidate_panels.py` |
| `outputs/tables/cu_gr_v2_swap_analysis.csv` | `build_candidate_panels.py` |
| `outputs/tables/cu_gr_v2_vs_fallback.csv` | `build_candidate_panels.py` |
| `outputs/tables/cu_gr_v2_preference_dataset.csv` | `build_preference_dataset.py` (placeholder until LLM JSONL) |
| `outputs/tables/cu_gr_v2_preference_parser_stats.csv` | `build_preference_dataset.py` |
| `outputs/tables/cu_gr_v2_fusion_train_summary.csv` | `train_preference_fusion.py` (placeholder) |
| `outputs/tables/cu_gr_v2_feature_importance.csv` | `train_preference_fusion.py` (placeholder) |
| `outputs/tables/cu_gr_v2_fusion_results.csv` | `replay_preference_fusion.py` (placeholder) |
