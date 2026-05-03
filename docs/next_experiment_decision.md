# Next experiment direction (post R3 / R3b)

This note compares two paper directions. The recommendation table is filled only
from completed R3b exports in `outputs/tables/r3b_conservative_gate_*.csv`. Do not
invent metrics.

## Path 1: Observation-first paper (recommended if R3b confirms conservative ≈ fallback)

**Premise:** Generative LLM recommendation under fixed candidates exposes severe
calibration failure, high-confidence wrong behavior, and harmful overrides when
naively fused with a fallback ranker.

**Next experiments (allowed scaling):**

- Multi-dataset **observation** studies (MovieLens + Amazon domains): same style
  of metrics, case exports, and calibration tables—not multi-dataset **method**
  scaling to claim a new SOTA ranker.
- Confidence calibration analysis, reliability diagrams, risk–coverage.
- High-confidence wrong and grounding-failure case studies.
- Popularity / long-tail stratification.
- `llm_rerank_real` **only** after parser fix, via **cache replay** or a small
  approved rerank-only API subgate—not as a claimed winner until parses succeed.
- Conservative gate as a **safety baseline** in discussion, not as “our main
  method beats BM25”.

## Path 2: Method-improvement paper (only if new evidence appears)

**Not viable** under current R3 evidence: accepted overrides hurt; Ours full is
worse than fallback-only; direct generative / rerank paths are broken or
zero-hit under the candidate-500 protocol.

**Would require, before any ranking-improvement claim:**

- A policy or model change that **beats fallback on validation** without leakage.
- Improvement on **multiple held-out seeds** under the same protocol.
- **At least two datasets** with comparable pipelines.
- Ablations showing which **uncertainty component** carries any new effect.
- Calibration or selective-risk improvement with pre-registered thresholds.

## Rerank parser follow-up (Part E)

- Parser recovery for truncated JSON `ranked_items` is already in-tree with
  tests; **do not** rerun live rerank API in this gate.
- Next rerank validation: replay from **raw outputs / cache** when possible.
- If raw outputs cannot reconstruct scores, schedule a **small rerank-only** API
  subgate later under `docs/server_runbook.md`.

## Recommendation (filled from CSVs; no API)

| Field | Value |
| --- | --- |
| Selected path | **Observation-first** (with **hybrid** nuance: ranking table is diagnostic; primary claims are calibration / override / fallback-alignment). |
| Basis | `r3b_conservative_gate_main.csv` (seed **13** in `seeds` column): BM25, `ours_fallback_only`, and **`ours_conservative_uncertainty_gate`** Recall@10 **0.059106**; `ours_uncertainty_guided_real` **0.052318**, delta vs BM25 **-0.006788**, mean confidence **0.850596**, ECE **0.850596**, Brier **0.725007**, high-confidence-wrong **6031**, top10-identical-to-BM25 **0.91457** (Ours full). Conservative gate: same ranking means as BM25 / fallback-only, **delta Recall@10 vs BM25 0.0**, **top10-identical-to-BM25 1.0**. Decisions (`r3b_conservative_gate_decision_stats.csv`): Ours full **516** accept, **115** rerank, **5409** fallback; conservative gate **6040** fallback. R3 case CSVs (51 lines each) remain illustrative (R3 paths in file). |
| Single next action | **Run multi-dataset observation study** (observation-only expansion; same diagnostic metrics—not multi-dataset method scaling). |

**Method-improvement path:** Not supported by the numeric rows above (Ours full **under** BM25 on Recall@10).

**Conservative gate vs fallback:** On the exported seed-13 row, aggregate Recall@10 / NDCG@10 / MRR@10 **match** `ours_fallback_only` and BM25; decision stats show **only** fallback (no accept/rerank). That supports **alignment with fallback-only on these metrics**, not a claim of beating BM25 or improving over fallback.
