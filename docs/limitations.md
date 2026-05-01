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
