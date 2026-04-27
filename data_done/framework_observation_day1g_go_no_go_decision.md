# Framework-Observation-Day1g Go/No-Go Decision

## Recommendation

`use_relative_ranking_score_but_not_raw_uncertainty`

## Interpretation

Day1g compares pointwise, listwise, and pairwise signals on the same Beauty 100-user subset where possible. This remains observation only: no training, no evidence fields, no CEP, no external API, and no four-domain run.

Relative candidate context is useful for local Qwen-LoRA recommendation ranking, but raw self-reported uncertainty is still unusable. The observation claim is: use relative ranking score, but not raw verbalized uncertainty.

## Test Snapshot

- pointwise logit P(true) MRR/HR@1/NDCG@3/AUROC: `0.48083333333333333` / `0.26` / `0.4422580581071478` / `0.57436`
- best relative-context method: `listwise_ranking`
- best relative-context MRR/HR@1/NDCG@3/AUROC: `0.73` / `0.46` / `0.800702066928587` / `0.892`
