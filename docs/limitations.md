# Limitations

- Implicit feedback ground truth is incomplete and may not represent all user
  preferences.
- Generated title grounding may be imperfect, especially with ambiguous,
  duplicated, translated, or paraphrased titles.
- Confidence may be prompt-sensitive and provider-sensitive.
- API models may not expose logprobs or deterministic decoding controls.
- Real calibration analysis requires enough samples per confidence and
  popularity bucket.
- Echo-chamber proxies based on history similarity, category repetition,
  diversity, or novelty are imperfect.
- Smoke tests are not paper evidence.
- MockLLM outputs are not paper evidence.
- OursMethod thresholds are provisional.
- Dataset bias and popularity confounding remain concerns.
- Candidate-set protocols can change measured difficulty and must be reported
  separately.
- Cost, rate limits, and model availability can affect reproducibility.
- OursMethod is integrated but still needs real validation before any
  effectiveness claim.

## Evidence after R3 / R3b (MovieLens candidate-500)

Offline R3 refinement on saved artifacts showed: Ours full **worse** than
fallback-only on ranking metrics; **accepted LLM overrides hurt**; direct
`llm_generative_real` / `llm_rerank_real` can be **zero-hit** under this protocol;
verbalized confidence is **badly miscalibrated** with frequent **high-confidence
wrong** generations; a **conservative uncertainty gate** can match fallback
aggregate metrics but is **fallback-safe behavior**, not a validated ranking
improvement claim until multi-seed R3b tables and broader settings confirm it.

Do **not** scale to multi-dataset **method** experiments until observation-first
claims and safety baselines are locked; multi-dataset **observation** studies may
proceed under a separate protocol.
