# Framework-Observation-Day2 Original Research Direction Note

This branch is not simply "LLM confidence calibration." It studies uncertainty for a recommender that generates item titles and then must be grounded back to a catalog.

Possible names:
- Generative Recommendation Uncertainty
- Catalog-Grounded Generative Recommendation Calibration
- Title-Level Recommendation Confidence Calibration

## Core Research Questions

RQ1: When a local LLM recommender generates an item title, is its confidence calibrated to whether the title can be grounded to the correct catalog item?

RQ2: Do generative validity signals, such as catalog match score, generation consistency, and title retrieval margin, provide better uncertainty than verbalized confidence?

RQ3: Can calibration transform weak generative confidence or grounding signals into useful decision risk for recommendation?

RQ4: How does closed-catalog candidate-grounded generation differ from open-title generation in hallucination and calibration?

## Proposed Observation Path

Day2:
- Design the generative recommendation observation.
- Run Beauty 100-user candidate-grounded smoke with base Qwen3-8B.
- Diagnose title validity, catalog grounding, recommendation hit rate, and raw confidence calibration.

Day3 if candidate-grounded generation works:
- Expand to valid/test 500 or Beauty full.
- Add generation logprob, retrieval margin, and title self-consistency agreement if verbalized confidence collapses.

If hallucination is high:
- Prioritize candidate-grounded generation and defer open-title full runs.

If candidate-grounded generation is strong:
- Consider later evidence observation or CEP framework integration, but do not treat Day2 itself as CEP.
