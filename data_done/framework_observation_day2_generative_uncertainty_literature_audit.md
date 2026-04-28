# Framework-Observation-Day2 Literature / Inspiration Audit

Status: design audit for a new generative recommendation uncertainty branch. This is not CEP, not evidence decomposition, and not training.

## Why We Pivot From Day1

Day1-Day1i tested local Qwen-LoRA confidence under pointwise yes/no recommendation decisions. The observation was useful but narrow: verbalized scalar confidence collapsed, token-probability `P(true)` gave only a weak signal, self-consistency did not beat logit scores, and listwise behavioral uncertainty was confounded by candidate order. The next observation should therefore match the final task more closely: given user history, generate a recommended item title and evaluate whether that generation is valid, grounded, correct, and calibratable.

## A. Verbalized Confidence Calibration

[ConfTuner](https://huggingface.co/papers/2508.18847) explicitly targets LLM verbalized confidence. Its motivation lines up with our Day1 finding: LLMs can be overconfident, and prompt-only confidence expressions are often poorly calibrated. ConfTuner uses a tokenized Brier-score objective to train confidence expressions, and reports improved calibration across reasoning tasks.

Design implication for our project: verbalized confidence is a measurement target, not something to trust by default. We should record the model's generated `confidence`, but treat it as raw and uncalibrated. Unlike ConfTuner, Day2 does not fine-tune a confidence reporter; our task is recommendation generation and catalog grounding, so confidence must be evaluated against whether the generated title maps to the correct catalog item.

## B. Trustworthiness / Dynamic Benchmarking

[TrustGen](https://www.microsoft.com/en-us/research/publication/trustgen-a-platform-of-dynamic-benchmarking-on-the-trustworthiness-of-generative-foundation-models/) frames trustworthiness as dynamic and multi-dimensional rather than a single scalar. It emphasizes modular evaluation and dimensions such as truthfulness, robustness, privacy, and hallucination resistance. The project page also highlights dynamic evaluation and contextual variation as core ingredients.

Design implication for our project: a generative recommender should not be evaluated only by one confidence number. For Day2 we separate validity, catalog grounding, recommendation quality, calibration, hallucination, and robustness to candidate/open-title settings. This keeps the observation close to trustworthiness auditing without claiming a final trust framework.

## C. Generative Recommendation / Catalog Grounding

[P5 / Recommendation as Language Processing](https://arxiv.org/abs/2203.13366) motivates recommendation as a text-to-text problem: interactions, metadata, and reviews can be converted into natural language sequences, and a language model can predict recommendation outputs through prompts. This supports our pivot from binary yes/no classification to generative recommendation.

[TIGER / Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) shows another generative recommendation paradigm: the model autoregressively generates identifiers for target items, with Recall/NDCG-style evaluation. Although Day2 generates titles rather than semantic IDs, TIGER reinforces the same principle: a generated recommendation must be grounded back to catalog items before recommendation metrics are meaningful.

The [language-modeling recommendation survey](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00619/118712/Pre-train-Prompt-and-Recommendation-A) is useful as background because it organizes PLM-based recommendation and discusses evaluation metrics such as HitRate, MRR, and NDCG. For generated titles, lexical metrics alone are not enough; we need catalog matching and recommendation metrics.

## Design Principles For Day2

- Do not directly trust verbalized confidence. Record it, diagnose collapse/overconfidence, and calibrate only with valid-fit/test-evaluate.
- Generative recommendation needs catalog-grounded evaluation. A generated title is useful only if it can be mapped to a candidate or catalog item.
- Uncertainty should predict generated recommendation risk: invalid JSON, invalid candidate title, catalog hallucination, wrong catalog grounding, or wrong target item.
- Confidence sources should be plural: raw verbalized confidence, sequence logprob, self-consistency title agreement, retrieval match score, retrieval margin, and candidate-ranking margin.
- Candidate-grounded generation is the first smoke because it is evaluable and reduces free-form hallucination. Open-title generation is closer to deployment but should come after grounding diagnostics are stable.
- Calibration must be valid-fit/test-evaluate. If only test smoke exists, report diagnostics but do not make calibration claims.
