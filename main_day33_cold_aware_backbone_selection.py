from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_DIR = Path("output-repaired/summary")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_audit() -> pd.DataFrame:
    rows = [
        {
            "candidate_backbone": "OpenP5 generative ranker",
            "repo_or_local_path": "external/OpenP5",
            "model_type": "text-to-item / generative recommendation platform",
            "uses_item_id_embedding_only": False,
            "uses_candidate_text": True,
            "uses_metadata": True,
            "handles_cold_candidate": "potentially_yes_if_prompted_and_scored",
            "requires_pretrained_checkpoint": True,
            "requires_generated_data": True,
            "requires_llm_embedding": False,
            "can_export_candidate_score": "possible_but_requires_generation_likelihood_adapter",
            "expected_join_key": "user_id + candidate_item_id after generated-id mapping",
            "expected_fallback_risk": "medium_high",
            "integration_difficulty": "high",
            "recommended_priority": "medium",
            "notes": "README lists Movies/Electronics/Beauty and text/generative setup, but official eval requires downloaded data/checkpoints and candidate score export needs a generative scoring adapter. Not a Day33 smoke candidate.",
        },
        {
            "candidate_backbone": "LLM-ESR LLM-enhanced SASRec/GRU4Rec/Bert4Rec",
            "repo_or_local_path": "external/LLM-ESR/models/LLMESR.py",
            "model_type": "LLM-embedding-enhanced sequential recommender",
            "uses_item_id_embedding_only": False,
            "uses_candidate_text": "indirect_via_precomputed_llm_item_embeddings",
            "uses_metadata": "indirect_via_precomputed_llm_item_embeddings",
            "handles_cold_candidate": "only_if_candidate_embedding_exists",
            "requires_pretrained_checkpoint": False,
            "requires_generated_data": True,
            "requires_llm_embedding": True,
            "can_export_candidate_score": "yes_after_handled_data_and_embeddings",
            "expected_join_key": "mapped user/item ids then user_id + candidate_item_id",
            "expected_fallback_risk": "medium_high_for_new_cold_items_without_embeddings",
            "integration_difficulty": "high",
            "recommended_priority": "medium",
            "notes": "DualLLM models load itm_emb_np.pkl and pca64_itm_emb_np.pkl from handled data. Without embeddings for Movies medium candidates, it remains blocked.",
        },
        {
            "candidate_backbone": "LLMEmb",
            "repo_or_local_path": "external/LLMEmb",
            "model_type": "LLM-generated item/user embedding recommender",
            "uses_item_id_embedding_only": False,
            "uses_candidate_text": "indirect_via_generated_llm_embeddings",
            "uses_metadata": "indirect_via_generated_llm_embeddings",
            "handles_cold_candidate": "only_if_embedding_generated_for_candidate",
            "requires_pretrained_checkpoint": True,
            "requires_generated_data": True,
            "requires_llm_embedding": True,
            "can_export_candidate_score": "yes_in_principle_but_blocked",
            "expected_join_key": "mapped user/item ids then user_id + candidate_item_id",
            "expected_fallback_risk": "high_without_embeddings",
            "integration_difficulty": "high",
            "recommended_priority": "low_for_day33",
            "notes": "Day13 already found missing handled data, LLM/SRS embeddings, and checkpoint. It is a future stronger baseline, not a quick cold-aware smoke.",
        },
        {
            "candidate_backbone": "LLM-ESR GRU4Rec",
            "repo_or_local_path": "external/LLM-ESR/models/GRU4Rec.py",
            "model_type": "ID-based sequential recommender",
            "uses_item_id_embedding_only": True,
            "uses_candidate_text": False,
            "uses_metadata": False,
            "handles_cold_candidate": False,
            "requires_pretrained_checkpoint": False,
            "requires_generated_data": False,
            "requires_llm_embedding": False,
            "can_export_candidate_score": True,
            "expected_join_key": "user_id + candidate_item_id",
            "expected_fallback_risk": "high_on_movies_medium",
            "integration_difficulty": "low",
            "recommended_priority": "exclude_for_movies_medium",
            "notes": "Healthy on Beauty, but Movies medium has test candidate cold rate 0.9439 and positive cold rate 0.7876. Reusing this would repeat the SASRec mismatch.",
        },
        {
            "candidate_backbone": "LLM-ESR Bert4Rec",
            "repo_or_local_path": "external/LLM-ESR/models/Bert4Rec.py",
            "model_type": "ID-based sequential recommender",
            "uses_item_id_embedding_only": True,
            "uses_candidate_text": False,
            "uses_metadata": False,
            "handles_cold_candidate": False,
            "requires_pretrained_checkpoint": False,
            "requires_generated_data": False,
            "requires_llm_embedding": False,
            "can_export_candidate_score": True,
            "expected_join_key": "user_id + candidate_item_id",
            "expected_fallback_risk": "high_on_movies_medium",
            "integration_difficulty": "low",
            "recommended_priority": "exclude_for_movies_medium",
            "notes": "Healthy on Beauty but still an item-id embedding model; cannot score unseen candidate items honestly without fallback.",
        },
        {
            "candidate_backbone": "Minimal SASRec-style",
            "repo_or_local_path": "main_day17_sasrec_backbone_plugin_smoke.py",
            "model_type": "ID-based sequential recommender",
            "uses_item_id_embedding_only": True,
            "uses_candidate_text": False,
            "uses_metadata": False,
            "handles_cold_candidate": False,
            "requires_pretrained_checkpoint": False,
            "requires_generated_data": False,
            "requires_llm_embedding": False,
            "can_export_candidate_score": True,
            "expected_join_key": "user_id + candidate_item_id",
            "expected_fallback_risk": "observed_0.9698_on_movies_medium",
            "integration_difficulty": "already_integrated",
            "recommended_priority": "exclude_for_movies_medium",
            "notes": "Day32 diagnosed fallback_rate 0.9698; non-invasive repair only reduced it to 0.9307, still blocked.",
        },
        {
            "candidate_backbone": "SLMRec",
            "repo_or_local_path": "Day11 audit: https://github.com/WujiangXu/SLMRec",
            "model_type": "small language model distillation for sequential recommendation",
            "uses_item_id_embedding_only": "unclear",
            "uses_candidate_text": "likely",
            "uses_metadata": "likely",
            "handles_cold_candidate": "unclear_until_repo_cloned",
            "requires_pretrained_checkpoint": "likely",
            "requires_generated_data": "likely",
            "requires_llm_embedding": "possibly",
            "can_export_candidate_score": "unknown",
            "expected_join_key": "to_verify",
            "expected_fallback_risk": "unknown",
            "integration_difficulty": "medium_high",
            "recommended_priority": "medium_for_later_audit",
            "notes": "Promising conceptually, but not present locally and not a safe Day33 smoke without repo/data inspection.",
        },
        {
            "candidate_backbone": "AGRec / graph reasoning LLM sequential",
            "repo_or_local_path": "Day11 audit: https://github.com/WangXFng/AGRec",
            "model_type": "autoregressive LLM decoder plus graph reasoning",
            "uses_item_id_embedding_only": False,
            "uses_candidate_text": "likely",
            "uses_metadata": "likely",
            "handles_cold_candidate": "possible_but_unverified",
            "requires_pretrained_checkpoint": "likely",
            "requires_generated_data": "likely",
            "requires_llm_embedding": "likely",
            "can_export_candidate_score": "unknown_adapter_needed",
            "expected_join_key": "to_verify",
            "expected_fallback_risk": "unknown",
            "integration_difficulty": "high",
            "recommended_priority": "low_for_day33",
            "notes": "Potentially cold-aware but too complex for immediate Movies medium smoke.",
        },
        {
            "candidate_backbone": "Transparent TF-IDF/BM25 history-candidate similarity",
            "repo_or_local_path": "proposed local baseline for Day34",
            "model_type": "content-based retrieval baseline",
            "uses_item_id_embedding_only": False,
            "uses_candidate_text": True,
            "uses_metadata": True,
            "handles_cold_candidate": True,
            "requires_pretrained_checkpoint": False,
            "requires_generated_data": False,
            "requires_llm_embedding": False,
            "can_export_candidate_score": True,
            "expected_join_key": "user_id + candidate_item_id",
            "expected_fallback_risk": "low",
            "integration_difficulty": "low",
            "recommended_priority": "top_for_day34",
            "notes": "Not an NH SOTA backbone, but transparent, cold-aware, reproducible, and appropriate as a cross-domain carrier when ID-based sequential models fail.",
        },
    ]
    df = pd.DataFrame(rows)
    write_csv(df, SUMMARY_DIR / "day33_cold_aware_backbone_audit.csv")
    return df


def write_mismatch_explanation() -> None:
    text = """# Day33 Movies Backbone Mismatch Explanation

## 1. Movies CEP Calibration Holds

Day31 completed Movies medium_5neg_2000 valid/test relevance evidence inference with 12,000 rows per split, parse_success=1.0, and raw_response_nonempty=1.0. On test, raw relevance ECE was 0.2913 and calibrated relevance ECE was 0.0044. This supports the cross-domain CEP calibration consistency claim: raw LLM relevance probability remains miscalibrated outside Beauty, and calibrated relevance posterior repairs probability quality.

## 2. Movies SASRec Plug-in Is Blocked

Day32 diagnosed the Movies SASRec failure mode. Join coverage is 1.0, but fallback_rate is 0.9698, positive fallback is 0.8965, and negative fallback is 0.9844. Non-invasive ID repair only reduces fallback to 0.9307, still far above the <20% health threshold.

## 3. Root Cause

The blocker is candidate coldness plus history/id mapping mismatch. Movies medium stores many history entries as `Item ID: ASIN` text while the Beauty SASRec helper originally expected title-to-id mapping. More importantly, test candidate cold rate is 0.9439 and test positive cold rate is 0.7876, meaning most candidate items are absent from the train item vocabulary. ID-based item embedding backbones cannot honestly score these items without fallback.

## 4. Why Beauty Backbones Do Not Transfer Mechanically

SASRec, GRU4Rec, and Bert4Rec were healthy on Beauty because the Beauty candidate pool and train vocabulary had acceptable coverage. Movies regular-medium is different: it is a cross-domain split with many cold candidate items. Reusing ID-based sequential backbones would measure fallback behavior, not external backbone plug-in performance.

## 5. Day33 Direction

The cross-domain carrier should be cold-aware: content-based, text-aware, LLM-embedding-aware, or retrieval-style. The next backbone should score user history text against candidate_title/candidate_text, rather than relying entirely on train item-id embeddings.
"""
    (SUMMARY_DIR / "day33_movies_backbone_mismatch_explanation.md").write_text(text, encoding="utf-8")


def write_selection_report(audit: pd.DataFrame) -> None:
    top = audit[audit["recommended_priority"] == "top_for_day34"].iloc[0]
    text = f"""# Day33 Cold-Aware Backbone Selection Report

## 1. Day32 Recap

Movies CEP calibration is valid: raw relevance ECE 0.2913 drops to calibrated ECE 0.0044 on Movies medium_5neg_2000 test. The Movies SASRec plug-in is blocked because fallback_rate is 0.9698, test candidate cold rate is 0.9439, and test positive cold rate is 0.7876.

## 2. Why Beauty Backbones Are Not Automatically Suitable

The Beauty full experiments used three healthy sequential backbones because candidate coverage was acceptable. Movies medium has a different failure mode: most candidate items are unseen by the train item vocabulary. ID-based SASRec/GRU4Rec/Bert4Rec can still rank Beauty candidates, but on Movies medium they would mostly rank fallback scores. This is a backbone/data mismatch, not a CEP failure.

## 3. NH / Project Audit Under Cold-Candidate Criteria

The audit table is saved at `output-repaired/summary/day33_cold_aware_backbone_audit.csv`. In short:

- OpenP5 is text/generative and potentially cold-aware, but it requires downloaded data/checkpoints and a generation-likelihood candidate-score adapter.
- LLM-ESR LLM-enhanced models use LLM item embeddings, but require handled data and precomputed embedding files such as `itm_emb_np.pkl` and `pca64_itm_emb_np.pkl`.
- LLMEmb is conceptually suitable but remains blocked from Day13 by missing handled data, embeddings, and checkpoints.
- LLM-ESR GRU4Rec/Bert4Rec and minimal SASRec are ID-based sequential models and should not be reused for Movies medium without solving cold candidates.

## 4. Recommended Backbone For Day34

Recommended next carrier: `{top['candidate_backbone']}`.

Reason: it can score cold candidate items from `candidate_text`/metadata, exports a transparent `user_id, candidate_item_id, backbone_score, label` table, and does not require external checkpoints, LLM embeddings, generated data, or new API calls. It is not a SOTA claim; it is a cold-aware cross-domain carrier to test whether CEP plug-in still helps when ID-based sequential backbones are invalid.

## 5. Was A 100-User Smoke Run?

No. Day33 intentionally does not run a smoke test because no existing NH/project backbone in the local workspace satisfies all cold-aware requirements without missing assets or a new scoring adapter. Running ID-based GRU4Rec/Bert4Rec would repeat the Day32 failure mode.

## 6. Day34 Next Action

Implement a transparent content-based Movies medium5 100-user smoke:

- build history text from the user's history entries;
- score candidates with TF-IDF/BM25-style similarity against `candidate_text`;
- export all candidate scores, not just top-10;
- join existing Movies CEP evidence via `user_id + candidate_item_id`;
- run A/B/C/D plug-in with metric-repaired outputs and HR@10 marked trivial.
"""
    (SUMMARY_DIR / "day33_cold_aware_backbone_selection_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    audit = build_audit()
    write_mismatch_explanation()
    write_selection_report(audit)
    print("Wrote Day33 cold-aware backbone audit and selection reports.")


if __name__ == "__main__":
    main()
