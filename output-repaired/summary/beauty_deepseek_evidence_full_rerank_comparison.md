# Beauty Evidence Rerank Comparison

The original Day5 grid is a diagnostic ablation, not evidence against evidence posterior. Because it used repaired_confidence as the base score and 1 - repaired_confidence as the penalty, the final score is a monotonic affine transform for non-negative lambda and cannot change ranking order.

| family | setting | lambda_penalty | base_score | uncertainty | NDCG@10 | MRR@10 | rank_change_rate | rerank_is_noop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monotonic_original | monotonic | 0 | minimal_repaired_confidence | minimal_evidence_uncertainty | 0.499333 | 0.342823 | 0 | True |
| decoupled | A | 0.1 | raw_confidence | 1_minus_repaired_confidence | 0.585745 | 0.453871 | 0.241692 | False |
| decoupled | B | 0.2 | raw_confidence | evidence_risk | 0.594429 | 0.464868 | 0.338986 | False |
| decoupled | C | 1 | repaired_confidence | evidence_risk | 0.535301 | 0.38914 | 0.59421 | False |

The decoupled grid separates relevance-like base scores from risk-like uncertainty signals. This is the decision experiment to carry forward into Day6-Day10.
