# Day41 Paper Results Section Final Draft

## RQ1: Are Raw LLM Confidence/Relevance Signals Reliable?

Our starting question is whether raw LLM confidence or relevance probability can be directly used as a recommendation decision signal. The answer is no. Across the observation diagnostics, raw confidence and raw relevance are informative, but they are substantially miscalibrated. This is visible in ECE/Brier and high-confidence error behavior. Therefore, raw LLM confidence should not be interpreted as calibrated probability.

## RQ2: Does CEP Improve Calibration?

CEP improves probability quality by converting evidence-grounded relevance outputs into a calibrated relevance posterior. On Beauty and small-domain replications, calibrated relevance consistently reduces ECE/Brier relative to raw relevance probability. AUROC does not need to improve dramatically because calibration primarily fixes probability scale rather than creating a new ranker.

## RQ3: Can CEP Improve External Recommender Backbones?

On Beauty full multi-seed experiments, CEP improves three sequential backbones: SASRec-style, GRU4Rec, and Bert4Rec. The strongest evidence is the full Beauty three-backbone multi-seed table. Component attribution shows that calibrated relevance posterior provides the primary gain, while evidence risk works as a secondary regularizer. We do not present CEP as replacing recommender backbones; instead, the backbone supplies ranking ability and CEP supplies calibrated posterior/risk information.

## RQ4: Is CEP Robust Under Input Perturbations?

The Day30 controlled robustness experiment perturbs user history and candidate text on a 500-user Beauty subset. CEP degrades modestly rather than collapsing, and the combined D setting remains close to clean CEP performance. This supports robustness, but it remains a first-run robustness setting rather than a full-domain robustness claim.

## RQ5: Does The Trend Generalize Beyond Beauty?

Small-domain Movies/Books/Electronics replicate calibration consistency and directional plug-in behavior, but the ID-backbone fallback caveat is important. Fallback sensitivity shows that gains are not explained by fallback flags alone, yet many small-domain gains are best interpreted as fallback/cold compensation or sample-limited directionality. Thus, small domains support cross-domain sanity/continuity, while Beauty full remains the primary performance evidence. Regular medium analysis further shows that realistic cross-domain settings may require content-aware/cold-aware backbones.

## Metrics Boundary

Several experiments use six candidates per user. In those settings, HR@10 is trivial and should not be used as claim-supporting evidence. Primary metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.
