# CU-GR: Calibrated Uncertainty-Gated Recommendation

## Purpose

Convert offline observations from R3 (LLM uncertainty, grounding, fallback alignment)
into a **learned decision rule** that defaults to the BM25 fallback list and only
**promotes the LLM grounded item to rank 1** when calibrated heads predict low harm
risk and positive NDCG gain in expectation. This is **not** a generic LLM reranker
and **not** a verbalized-confidence threshold policy.

## Data and leakage controls

- **Features** are built only from predictions, metadata, and **train-only**
  popularity (same processed MovieLens artifacts as other analysis modules).
- **Target item** and all **label-derived** deltas are used **only** for offline
  supervision and evaluation, never as model inputs.
- **Seeds:** train 13, validation 21, test 42. Threshold tuning uses **validation
  only**; seed 42 is held out for reported replay metrics.

## Labels (per example)

- Fallback list `F`: BM25 top-10 by score from the BM25 run row.
- Override list `O`: insert grounded item `g` at rank 1 (if parse/grounding/
  candidate-adherent and `g` in candidates), else `O = F`.
- `delta_ndcg@10 = NDCG@10(O) - NDCG@10(F)` using the same ranking helpers as
  `llm4rec.metrics.ranking` (not a second evaluator definition).
- `override_improves = 1[delta_ndcg@10 > 0]`, `override_hurts = 1[delta_ndcg@10 < 0]`.

## Models

- **ImproveCalibrator:** estimates `P(override_improves | x)`.
- **HarmCalibrator:** estimates `P(override_hurts | x)`.
- Implementation: `scikit-learn` `LogisticRegression(class_weight="balanced")`
  + sigmoid calibration on the validation seed when available; rule fallback if
  sklearn is missing (marked non-final).

## Policy

Hard gates: parse success, grounding success, candidate adherence → else
`KEEP_FALLBACK`. Otherwise promote iff
`p_improve >= tau_improve` and `p_harm <= tau_harm` with thresholds chosen on
validation to maximize mean NDCG@10 subject to
`harmful_overrides / n <= 0.01` and `accepted_override_count >= 10`. If no pair
satisfies constraints, replay **degenerates to fallback** and metadata records
`No reliable override region found`.

## Scripts and configs

| Artifact | Producer |
| --- | --- |
| `outputs/tables/cu_gr_calibrator_dataset.csv` | `scripts/build_calibrator_dataset.py` |
| `outputs/tables/cu_gr_class_balance.csv` | build script |
| `outputs/models/cu_gr_calibrator/model.pkl` | `scripts/train_override_calibrator.py` |
| `outputs/models/cu_gr_calibrator/metadata.json` | train script |
| `outputs/tables/cu_gr_threshold_sweep.csv` | train script |
| `outputs/tables/cu_gr_feature_importance.csv` | train script |
| `outputs/tables/cu_gr_policy_results.csv` | `scripts/replay_cu_gr_policy.py` |
| `outputs/tables/cu_gr_vs_fallback.csv` | replay script |
| `outputs/tables/cu_gr_error_cases.csv` | replay script |

YAML: `configs/methods/cu_gr.yaml`, `configs/experiments/r3_cu_gr_offline_replay.yaml`.

## Non-goals

- No prompt edits, no API calls, no new live LLM runs.
- No change to shared evaluator definitions; replay uses the same JSONL metric
  utilities as other tooling.
- Rerank / `USE_LLM_RERANK` is **out of scope** until parser recovery is validated.
