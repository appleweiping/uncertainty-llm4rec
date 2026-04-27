# Framework-Observation-Day1e Go/No-Go Decision

## Decision

`no_go_for_full_beauty_yet`

P(true) is not collapsed and calibration helps, but the 200/200 smoke is still weak; run a slightly larger threshold-free audit or Day1f before full Beauty.

## Evidence

- raw P(true) test MRR / random MRR: `0.49411764705882355` / `0.4183823529411765`
- raw P(true) test NDCG@3 / random NDCG@3: `0.44572496376050846` / `0.36869348592437484`
- raw P(true) test HR@1 / random HR@1: `0.29411764705882354` / `0.1764705882352941`
- best threshold strategy on test balanced accuracy: `best_threshold_by_f1`
- best threshold test balanced accuracy: `0.5960311835577605`
- best threshold test recommend true rate: `0.37`

## Interpretation

Logit/token probability fixes scalar verbalized confidence collapse and gives a usable-but-weak miscalibrated signal. However, the hard recommend=true/false decision remains conservative and under-recommending at the default 0.5 threshold, so we should not full-run yet.
