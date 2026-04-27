# Framework-Observation-Day1e Logit Score Ranking Report

## 1. Day1d Recap

Day1d showed that logit/token probability fixes scalar verbalized confidence collapse. P(true) has `unique_count=200`, test std around `0.170`, and relevance AUROC around `0.589`. Calibration reduces relevance ECE from `0.11125464997727197` to `0.034089472091289215`.

## 2. Why Hard Threshold 0.5 Is Not Enough

The 0.5 hard threshold gives a very conservative recommend decision. It produced low recommend-true rate in Day1d, but this does not mean the continuous score is useless. Day1e therefore evaluates P(true) as a threshold-free relevance/ranking score and learns thresholds on valid.

## 3. P(true) Threshold-Free Ranking

- test raw P(true) MRR / random MRR: `0.49411764705882355` / `0.4183823529411765`
- test raw P(true) NDCG@3 / random NDCG@3: `0.44572496376050846` / `0.36869348592437484`
- test raw P(true) HR@1 / random HR@1: `0.29411764705882354` / `0.1764705882352941`
- HR@10 is marked trivial because candidate pools are at most 10 items.

## 4. Valid Threshold Selection

- best test balanced-accuracy strategy: `best_threshold_by_f1`
- valid threshold: `0.1`
- test accuracy/F1/balanced accuracy: `0.64` / `0.33333333333333337` / `0.5960311835577605`
- test recommend true rate: `0.37`

## 5. Calibration Result

Valid-set calibration makes the weak P(true) signal more usable by reducing ECE/Brier. This supports continuing with token-probability confidence rather than verbalized scalar confidence.

## 6. Go/No-Go For Full Beauty

- decision: `no_go_for_full_beauty_yet`
- rationale: P(true) is not collapsed and calibration helps, but the 200/200 smoke is still weak; run a slightly larger threshold-free audit or Day1f before full Beauty.

## 7. Next Step

If the ranking lift over random is modest or ambiguous, do not full-run. The next smoke should be Day1f self-consistency 100/100 or a pair/list context audit, not more scalar confidence wording.
