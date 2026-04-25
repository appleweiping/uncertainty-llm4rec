# Paper Motivation Snippet

The starting question is whether LLM verbalized confidence in recommendation can be directly used as a decision signal. Our multi-model diagnostics show that this is not the case. On Beauty, raw confidence is informative but substantially miscalibrated across multiple LLMs, with diagnostic ECE values of DeepSeek: 0.3025, Doubao: 0.3458, GLM: 0.3790, Kimi: 0.3163, Qwen: 0.3706. This indicates that the signal carries information about correctness, but its numerical scale should not be interpreted as a calibrated probability.

This issue persists when moving from yes/no recommendation confidence to candidate-level relevance probability. In the full Beauty relevance setting, raw relevance probability has ECE=0.2604 and Brier=0.2318, while valid-set calibration reduces these values to ECE=0.0095 and Brier=0.1347. The improvement is mainly a probability-quality repair rather than a guarantee of standalone ranking superiority, since calibration does not fundamentally retrain the ranker.

These observations motivate CEP: instead of directly using raw LLM confidence or relevance scores, we convert evidence-grounded LLM outputs into a calibrated relevance posterior before downstream decision making. In later plug-in experiments, this calibrated posterior is combined with external sequential recommendation backbones rather than replacing them.
