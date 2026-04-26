# Day36 Small-Domain Sanity Plan

## 1. Why Consider Small Domains

Small domains are useful as lightweight cross-domain sanity / continuity experiments. They should not replace regular medium domains as the realistic cross-domain setting.

## 2. Small vs Regular Medium

Regular medium domains preserve more realistic cold-start/content-carrier behavior. Small domains can answer whether the older cross-domain pipeline and ID-based backbone sanity checks are technically feasible under lower cost.

## 3. Schema / Candidate Pool / Cold-Rate

- books: path `data/processed/amazon_books_small`, candidate pool mean `6.00`, HR@10 trivial `True`, valid all-cold `0.0247`, test all-cold `0.0253`.
- electronics: path `data/processed/amazon_electronics_small`, candidate pool mean `6.00`, HR@10 trivial `True`, valid all-cold `0.0317`, test all-cold `0.0283`.
- movies: path `data/processed/amazon_movies_small`, candidate pool mean `6.00`, HR@10 trivial `True`, valid all-cold `0.0180`, test all-cold `0.0147`.

## 4. ID-Based Backbone Suitability

Use small domains for ID-backbone sanity only if cold rates are materially lower than regular medium and candidate coverage is healthy. Otherwise, treat them as calibration sanity rather than backbone evidence.

## 5. Existing LLM Confidence Observation

- books: existing raw-confidence diagnostics `True`, existing relevance evidence `False`.
- electronics: existing raw-confidence diagnostics `True`, existing relevance evidence `False`.
- movies: existing raw-confidence diagnostics `True`, existing relevance evidence `False`.

## 6. New DeepSeek Relevance Evidence

No API was launched in Day36. Config templates were prepared only for domains with train/valid/test jsonl.

## 7. Day37 Recommendation

If the goal is a low-cost cross-domain continuity check, run one small domain first with DeepSeek relevance evidence and keep the claim as sanity-only. If the goal is realistic cross-domain behavior, continue with regular medium cold-style content carriers rather than forcing ID-based backbones.

Small domains provide a lightweight cross-domain sanity setting, while regular medium domains provide more realistic cold-start/content-carrier analysis.
